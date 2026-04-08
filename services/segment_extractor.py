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

from services.llm_client import call_llm_json_async, TIER_DEFAULT
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
        result = await call_llm_json_async(prompt, max_tokens=max_tokens, tier=TIER_DEFAULT)

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

        # Operating margin
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

    logger.info("Segment decomposition → %d metric items", len(items))
    return items
