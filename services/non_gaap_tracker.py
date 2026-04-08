"""
Non-GAAP Bridge Tracker — extracts and tracks the reconciliation between
reported (GAAP/IFRS) and adjusted/underlying figures over time.

When the gap between GAAP and adjusted earnings widens, or when new
adjustments appear ("transformation costs", "strategic repositioning
charges"), that's an earnings quality red flag.

European industrials and consumer names use "adjusted" figures heavily.
This tracker catches:
  1. New adjustments that weren't in prior periods
  2. Growing adjustment magnitude (GAAP-to-adjusted gap widening)
  3. Recurring "one-offs" (same adjustment appearing period after period)
  4. Adjustment relabelling (same economic event, new name)

Runs as a dedicated extraction pass on financial statement sections,
then compares against prior period adjustments stored in the database.
"""

import logging
import re
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm_client import call_llm_json_async, TIER_DEFAULT

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Non-GAAP bridge extraction prompt
# ═══════════════════════════════════════════════════════════════════

NON_GAAP_BRIDGE_PROMPT = """\
You are an accounting quality analyst. Extract the FULL reconciliation bridge
between reported (GAAP/IFRS) and adjusted/underlying figures from this text.

Companies present this as:
  Reported operating profit: €X
  + Restructuring charges: €Y
  + Impairment: €Z
  + Other one-offs: €W
  = Adjusted operating profit: €V

Extract EVERY line item in the bridge, including:
- The reported (GAAP/IFRS) starting figure
- Each individual adjustment (positive or negative)
- The adjusted/underlying ending figure
- The same bridge for ANY metric (profit, EBITDA, EPS, margin, etc.)

CRITICAL RULES:
- Extract the EXACT adjustment labels used by the company
- Note whether each adjustment adds to or subtracts from reported
- Flag any adjustment that sounds like it could be a recurring cost
  disguised as one-off (e.g. "restructuring" appearing every period)
- If multiple bridges exist (e.g. operating profit AND net income), extract all

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "bridges": [
    {{
      "metric": "<e.g. Operating Profit, EBITDA, EPS, Net Income>",
      "reported_value": <number>,
      "adjusted_value": <number>,
      "reported_unit": "<unit>",
      "gap_absolute": <adjusted minus reported>,
      "gap_percentage": <gap as % of reported>,
      "period": "<period label>",
      "adjustments": [
        {{
          "label": "<exact label used by company>",
          "amount": <number — positive means added back>,
          "description": "<any additional context>",
          "is_likely_recurring": true | false,
          "recurrence_signal": "<why you think it's recurring, or null>"
        }}
      ]
    }}
  ]
}}

--- DOCUMENT TEXT ---
{text}
"""


# ═══════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════

async def extract_non_gaap_bridge(text: str, max_tokens: int = 4096) -> dict:
    """
    Extract non-GAAP reconciliation bridges from document text.
    Looks for sections with adjusted/underlying/non-GAAP reconciliations.
    """
    # First check if there's likely a bridge in the text
    bridge_signals = [
        r"(?i)reconciliation",
        r"(?i)adjusted\s+(?:operating|ebitda|profit|earnings|eps)",
        r"(?i)underlying\s+(?:operating|ebitda|profit|earnings|eps)",
        r"(?i)non.gaap",
        r"(?i)exceptional\s+items",
        r"(?i)one.(?:time|off)\s+items",
        r"(?i)specific\s+items",
        r"(?i)reported\s+to\s+adjusted",
        r"(?i)adjusting\s+items",
    ]

    has_bridge = any(re.search(p, text) for p in bridge_signals)
    if not has_bridge:
        logger.debug("Non-GAAP bridge: no reconciliation signals found in text")
        return {"bridges": [], "has_bridge": False}

    prompt = NON_GAAP_BRIDGE_PROMPT.format(text=text[:20000])

    try:
        result = await call_llm_json_async(prompt, max_tokens=max_tokens, tier=TIER_DEFAULT)
        if not isinstance(result, dict):
            return {"bridges": [], "has_bridge": True, "error": "unexpected response type"}

        bridges = result.get("bridges", [])
        logger.info("Non-GAAP bridge: extracted %d bridges", len(bridges))
        return {"bridges": bridges, "has_bridge": True}

    except Exception as e:
        logger.warning("Non-GAAP bridge extraction failed: %s", str(e)[:200])
        return {"bridges": [], "has_bridge": True, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════
# Cross-period comparison
# ═══════════════════════════════════════════════════════════════════

async def compare_bridges_across_periods(
    db: AsyncSession,
    company_id,
    current_period: str,
    current_bridges: list[dict],
) -> dict:
    """
    Compare current non-GAAP bridges against prior period to detect:
    1. New adjustments
    2. Gap widening
    3. Recurring "one-offs"
    4. Adjustment magnitude growth
    """
    from apps.api.models import ExtractedMetric

    # Find prior period
    prior_period = _previous_period(current_period)
    if not prior_period:
        return {"comparison_available": False}

    # Look for stored bridge data from prior period
    # Bridge data is stored as metrics with segment="non_gaap_bridge"
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == prior_period,
            ExtractedMetric.segment == "non_gaap_bridge",
        )
    )
    prior_bridge_metrics = q.scalars().all()

    if not prior_bridge_metrics:
        return {
            "comparison_available": False,
            "reason": f"No bridge data stored for {prior_period}",
        }

    # Build prior adjustment labels
    prior_labels = set()
    prior_gap_pct = None
    for m in prior_bridge_metrics:
        if m.metric_name.startswith("BRIDGE_ADJ:"):
            label = m.metric_name.replace("BRIDGE_ADJ:", "").strip().lower()
            prior_labels.add(label)
        elif m.metric_name.startswith("BRIDGE_GAP_PCT:"):
            prior_gap_pct = m.metric_value

    # Analyse current bridges
    flags = []
    new_adjustments = []
    recurring_one_offs = []
    gap_trend = None

    for bridge in current_bridges:
        current_gap_pct = bridge.get("gap_percentage")

        # Gap widening check
        if current_gap_pct is not None and prior_gap_pct is not None:
            try:
                curr = float(current_gap_pct)
                prev = float(prior_gap_pct)
                if abs(curr) > abs(prev) * 1.2:  # Gap grew by >20%
                    gap_trend = "widening"
                    flags.append({
                        "type": "gap_widening",
                        "severity": "high",
                        "metric": bridge.get("metric", ""),
                        "prior_gap_pct": prev,
                        "current_gap_pct": curr,
                        "signal": f"GAAP-to-adjusted gap widened from {prev:.1f}% to {curr:.1f}%",
                    })
                elif abs(curr) < abs(prev) * 0.8:
                    gap_trend = "narrowing"
            except (ValueError, TypeError):
                pass

        # Check each adjustment
        for adj in bridge.get("adjustments", []):
            label = (adj.get("label") or "").lower().strip()
            if not label:
                continue

            # New adjustment?
            is_new = not any(_fuzzy_label_match(label, pl) for pl in prior_labels)
            if is_new:
                new_adjustments.append({
                    "label": adj["label"],
                    "amount": adj.get("amount"),
                    "signal": f"New adjustment '{adj['label']}' not present in {prior_period}",
                })

            # Recurring one-off?
            if adj.get("is_likely_recurring"):
                recurring_one_offs.append({
                    "label": adj["label"],
                    "amount": adj.get("amount"),
                    "recurrence_signal": adj.get("recurrence_signal", ""),
                    "signal": f"'{adj['label']}' flagged as likely recurring despite being labelled one-off",
                })

    if new_adjustments:
        flags.append({
            "type": "new_adjustments",
            "severity": "medium",
            "count": len(new_adjustments),
            "items": new_adjustments,
            "signal": f"{len(new_adjustments)} new adjustment(s) appeared that weren't in {prior_period}",
        })

    if recurring_one_offs:
        flags.append({
            "type": "recurring_one_offs",
            "severity": "high",
            "count": len(recurring_one_offs),
            "items": recurring_one_offs,
            "signal": f"{len(recurring_one_offs)} adjustment(s) appear to be recurring costs labelled as one-off",
        })

    return {
        "comparison_available": True,
        "prior_period": prior_period,
        "gap_trend": gap_trend,
        "new_adjustments": new_adjustments,
        "recurring_one_offs": recurring_one_offs,
        "flags": flags,
        "total_flags": len(flags),
    }


def _fuzzy_label_match(label1: str, label2: str) -> bool:
    """Check if two adjustment labels refer to the same thing."""
    if not label1 or not label2:
        return False

    # Direct match
    if label1 == label2:
        return True

    # Check word overlap
    words1 = set(re.findall(r'\w+', label1))
    words2 = set(re.findall(r'\w+', label2))

    # Remove common filler words
    filler = {"and", "the", "of", "in", "for", "to", "a", "an", "on", "from", "costs", "charges", "items"}
    words1 -= filler
    words2 -= filler

    if not words1 or not words2:
        return False

    overlap = words1 & words2
    return len(overlap) >= max(1, min(len(words1), len(words2)) * 0.5)


def _previous_period(period: str) -> Optional[str]:
    """Compute prior period label."""
    if not period:
        return None
    period = period.strip().upper()
    m = re.match(r'^Q([1-4])\s*(\d{4})$', period)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        return f"Q4 {y - 1}" if q == 1 else f"Q{q - 1} {y}"
    m = re.match(r'^FY\s*(\d{4})$', period)
    if m:
        return f"FY {int(m.group(1)) - 1}"
    m = re.match(r'^H([12])\s*(\d{4})$', period)
    if m:
        h, y = int(m.group(1)), int(m.group(2))
        return f"H1 {y - 1}" if h == 1 else f"H1 {y}"
    return None


# ═══════════════════════════════════════════════════════════════════
# Persistence — store bridge data for future comparison
# ═══════════════════════════════════════════════════════════════════

async def persist_bridge_data(
    db: AsyncSession,
    company_id,
    document_id,
    period_label: str,
    bridges: list[dict],
) -> int:
    """
    Store bridge data as ExtractedMetric rows for cross-period comparison.
    Uses segment="non_gaap_bridge" to distinguish from regular metrics.
    """
    from apps.api.models import ExtractedMetric

    count = 0
    for bridge in bridges:
        metric_name = bridge.get("metric", "Operating Profit")

        # Store the gap percentage
        gap_pct = bridge.get("gap_percentage")
        if gap_pct is not None:
            m = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=company_id,
                document_id=document_id,
                period_label=period_label,
                metric_name=f"BRIDGE_GAP_PCT:{metric_name}",
                metric_value=gap_pct,
                unit="%",
                segment="non_gaap_bridge",
                confidence=0.9,
            )
            db.add(m)
            count += 1

        # Store the gap absolute
        gap_abs = bridge.get("gap_absolute")
        if gap_abs is not None:
            m = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=company_id,
                document_id=document_id,
                period_label=period_label,
                metric_name=f"BRIDGE_GAP_ABS:{metric_name}",
                metric_value=gap_abs,
                unit=bridge.get("reported_unit", ""),
                segment="non_gaap_bridge",
                confidence=0.9,
            )
            db.add(m)
            count += 1

        # Store each adjustment label + amount
        for adj in bridge.get("adjustments", []):
            label = adj.get("label", "unknown")
            amount = adj.get("amount")
            m = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=company_id,
                document_id=document_id,
                period_label=period_label,
                metric_name=f"BRIDGE_ADJ:{label}",
                metric_value=amount,
                unit=bridge.get("reported_unit", ""),
                segment="non_gaap_bridge",
                source_snippet=adj.get("description", "")[:500],
                confidence=0.85,
            )
            db.add(m)
            count += 1

    await db.commit()
    logger.info("Persisted %d bridge data items for %s", count, period_label)
    return count
