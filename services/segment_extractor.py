"""
Segment Decomposition Agent — dedicated revenue/profit tree extraction.

Runs as a specialist pass on financial statement and MD&A sections to build
a complete hierarchical breakdown of the business:
  Group → Division → Geography → Product Line

This catches segment breakdowns that the generic extractor misses,
and validates that segments sum to group totals.
"""

import logging
from typing import Optional

from services.llm_client import call_llm_json_async, TIER_DEFAULT, TIER_FAST
from prompts.section_prompts import SEGMENT_DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)


async def extract_segments(
    text: str,
    sector_context: str = "",
    max_tokens: int = 8192,
) -> dict:
    """
    Run the segment decomposition prompt on document text.

    Args:
        text: Document text (ideally financial statements + MD&A combined)
        sector_context: Sector-specific KPI context string
        max_tokens: Token budget

    Returns:
        Dict with group_total, segments list, and sum check.
    """
    prompt = SEGMENT_DECOMPOSITION_PROMPT.format(
        text=text[:25000],  # Cap at ~25k chars
        sector_context=sector_context or "",
    )

    try:
        result = await call_llm_json_async(prompt, max_tokens=max_tokens, tier=TIER_FAST)

        if not isinstance(result, dict):
            logger.warning("Segment extractor: unexpected result type %s", type(result))
            return {"group_total": None, "segments": [], "segments_sum_check": None}

        # Validate the sum check
        check = result.get("segments_sum_check", {})
        if check and check.get("passes") is False:
            diff = check.get("difference", 0)
            logger.warning(
                "Segment extractor: sum check FAILED (diff: %s). "
                "Segment revenues don't sum to group total.",
                diff,
            )

        segments = result.get("segments", [])
        logger.info("Segment extractor: %d segments extracted", len(segments))

        return result

    except Exception as e:
        logger.warning("Segment extraction failed: %s", str(e)[:200])
        return {"group_total": None, "segments": [], "segments_sum_check": None}


def segments_to_metrics(segment_data: dict, period_label: str = "") -> list[dict]:
    """
    Convert segment decomposition output into standard metric items
    that can be merged with the main extraction pipeline.
    """
    items = []

    # Group total
    group = segment_data.get("group_total", {})
    if group and group.get("revenue") is not None:
        items.append({
            "type": "metric",
            "metric_name": "Revenue (Group Total)",
            "metric_value": group["revenue"],
            "unit": group.get("revenue_unit", ""),
            "period": group.get("period", period_label),
            "segment": "Group",
            "extraction_method": "segment_decomposition",
            "confidence": 0.9,
            "source_snippet": "Segment decomposition group total",
        })

    if group and group.get("operating_profit") is not None:
        items.append({
            "type": "metric",
            "metric_name": "Operating Profit (Group Total)",
            "metric_value": group["operating_profit"],
            "unit": group.get("operating_profit_unit", ""),
            "period": group.get("period", period_label),
            "segment": "Group",
            "extraction_method": "segment_decomposition",
            "confidence": 0.9,
            "source_snippet": "Segment decomposition group total",
        })

    # Individual segments
    for seg in segment_data.get("segments", []):
        name = seg.get("segment_name", "Unknown")
        period = seg.get("period", period_label)
        level = seg.get("segment_level", "division")
        parent = seg.get("parent_segment")

        # Revenue
        if seg.get("revenue") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Revenue ({name})",
                "metric_value": seg["revenue"],
                "unit": seg.get("unit", ""),
                "period": period,
                "segment": name,
                "geography": name if level == "geography" else None,
                "extraction_method": "segment_decomposition",
                "confidence": 0.85,
                "source_snippet": seg.get("source_snippet", ""),
            })

        # Prior period revenue
        if seg.get("revenue_prior") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Revenue ({name})",
                "metric_value": seg["revenue_prior"],
                "unit": seg.get("unit", ""),
                "period": f"Prior {period}",
                "segment": name,
                "extraction_method": "segment_decomposition",
                "confidence": 0.8,
                "source_snippet": seg.get("source_snippet", ""),
            })

        # Growth rates
        if seg.get("revenue_growth_organic") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Organic Revenue Growth ({name})",
                "metric_value": seg["revenue_growth_organic"],
                "unit": "%",
                "period": period,
                "segment": name,
                "extraction_method": "segment_decomposition",
                "confidence": 0.85,
                "source_snippet": seg.get("source_snippet", ""),
            })
        elif seg.get("revenue_growth_reported") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Revenue Growth ({name})",
                "metric_value": seg["revenue_growth_reported"],
                "unit": "%",
                "period": period,
                "segment": name,
                "extraction_method": "segment_decomposition",
                "confidence": 0.85,
                "source_snippet": seg.get("source_snippet", ""),
            })

        # Operating profit
        if seg.get("operating_profit") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Operating Profit ({name})",
                "metric_value": seg["operating_profit"],
                "unit": seg.get("unit", ""),
                "period": period,
                "segment": name,
                "extraction_method": "segment_decomposition",
                "confidence": 0.85,
                "source_snippet": seg.get("source_snippet", ""),
            })

        # Operating margin — kept for non-financial segments. Insurance /
        # bank segments should emit their ratios via sector_kpis below.
        if seg.get("operating_margin") is not None:
            items.append({
                "type": "metric",
                "metric_name": f"Operating Margin ({name})",
                "metric_value": seg["operating_margin"],
                "unit": "%",
                "period": period,
                "segment": name,
                "extraction_method": "segment_decomposition",
                "confidence": 0.85,
                "source_snippet": seg.get("source_snippet", ""),
            })

        # Sector-specific KPIs — insurance combined/loss/expense ratios,
        # bank NII/C-I ratio/credit losses, etc. These go under their real
        # names so the financial_analyst agent + metric_store can find them
        # by the canonical KPI name instead of "Operating Margin".
        sector_kpis = seg.get("sector_kpis") or {}
        if isinstance(sector_kpis, dict):
            for key, value in sector_kpis.items():
                if value is None or value == "":
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                # Render snake_case key as "Combined Ratio", "Net Interest Income", etc.
                display = " ".join(w.capitalize() for w in str(key).replace("_", " ").split())
                unit = _guess_unit_for_kpi(key)
                items.append({
                    "type":              "metric",
                    "metric_name":       f"{display} ({name})",
                    "metric_value":      numeric,
                    "unit":               unit,
                    "period":             period,
                    "segment":            name,
                    "extraction_method":  "segment_decomposition",
                    "confidence":         0.85,
                    "source_snippet":     seg.get("source_snippet", ""),
                })

        # other_kpis — free-form sector-specific numbers that don't fit the
        # sector_kpis canonical keys. Same shape, lower confidence (0.7) so
        # downstream consumers can weight them appropriately.
        other_kpis = seg.get("other_kpis") or {}
        if isinstance(other_kpis, dict):
            for key, value in other_kpis.items():
                if value is None or value == "":
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                display = " ".join(w.capitalize() for w in str(key).replace("_", " ").split())
                items.append({
                    "type":              "metric",
                    "metric_name":       f"{display} ({name})",
                    "metric_value":      numeric,
                    "unit":               "",
                    "period":             period,
                    "segment":            name,
                    "extraction_method":  "segment_decomposition_other",
                    "confidence":         0.70,
                    "source_snippet":     seg.get("source_snippet", ""),
                })

    logger.info("Segment decomposition → %d metric items", len(items))
    return items


# Unit hints for the canonical sector_kpis keys emitted by the segment
# decomposition prompt. Insurance ratios are percentages; NPW/NPE are
# dollar amounts; charge-offs are dollar amounts, charge-off RATE is %.
_KPI_UNIT_HINTS = {
    # Insurance
    "combined_ratio":             "%",
    "current_ay_combined_ratio":  "%",
    "current_ay_ex_cat_cr":       "%",
    "loss_ratio":                 "%",
    "expense_ratio":              "%",
    "return_on_capital":          "%",
    "net_premiums_written":       "USD millions",
    "net_premiums_earned":        "USD millions",
    "underwriting_income":        "USD millions",
    "prior_period_development":   "USD millions",
    "catastrophe_losses":         "USD millions",
    # Bank
    "net_interest_income":        "USD millions",
    "fee_and_commission_income":  "USD millions",
    "cost_to_income_ratio":       "%",
    "credit_losses":              "USD millions",
    "credit_loss_ratio":          "%",
    "net_charge_offs":            "USD millions",
    "return_on_tangible_equity":  "%",
    "average_lending":            "USD millions",
    "average_deposits":           "USD millions",
    "net_interest_margin":        "%",
}


def _guess_unit_for_kpi(key: str) -> str:
    """Best-effort unit guess from the canonical KPI key."""
    k = str(key).lower().strip()
    if k in _KPI_UNIT_HINTS:
        return _KPI_UNIT_HINTS[k]
    # Fallback heuristic: anything ending in "ratio" / "margin" / "rate" / "growth" is %
    if k.endswith(("_ratio", "_margin", "_rate", "_growth", "_yield")):
        return "%"
    return ""
