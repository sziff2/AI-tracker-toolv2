"""
Disappeared Metrics & Guidance Detector — flags metrics and guidance items
that were present in prior periods but are missing from the current extraction.

The most bearish signal is silence. When management stops reporting a KPI
they previously highlighted, or drops guidance without formally withdrawing
it, they're hoping nobody notices.

Runs as a post-extraction step. Queries the database for prior period data
and compares against current extraction results.

Three detection modes:
  1. Disappeared KPIs — metrics reported in prior period but absent now
  2. Silent guidance drops — guidance given last period with no update
  3. Narrowing/widening guidance ranges — directional signal on visibility
"""

import logging
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Period utilities
# ═══════════════════════════════════════════════════════════════════

def _previous_period(period: str) -> Optional[str]:
    """Compute the likely prior period label."""
    if not period:
        return None

    period = period.strip().upper()

    # Q2 2025 → Q1 2025, Q1 2025 → Q4 2024
    m = re.match(r'^Q([1-4])\s*(\d{4})$', period)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        if q == 1:
            return f"Q4 {y - 1}"
        return f"Q{q - 1} {y}"

    # FY 2025 → FY 2024
    m = re.match(r'^FY\s*(\d{4})$', period)
    if m:
        return f"FY {int(m.group(1)) - 1}"

    # H1 2025 → H1 2024, H2 2025 → H1 2025
    m = re.match(r'^H([12])\s*(\d{4})$', period)
    if m:
        h, y = int(m.group(1)), int(m.group(2))
        if h == 1:
            return f"H1 {y - 1}"
        return f"H1 {y}"

    # 2025_Q2 format
    m = re.match(r'^(\d{4})[_\-]Q([1-4])$', period)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        if q == 1:
            return f"Q4 {y - 1}"
        return f"Q{q - 1} {y}"

    return None


def _normalise_metric_name(name: str) -> str:
    """Normalise a metric name for comparison across periods."""
    if not name:
        return ""
    # Remove segment qualifiers for matching
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
    # Remove common prefixes
    name = re.sub(r'^(?:GUIDANCE:\s*|ESG:\w+:\s*)', '', name)
    return name.strip().lower()


# ═══════════════════════════════════════════════════════════════════
# Database queries
# ═══════════════════════════════════════════════════════════════════

async def _get_prior_metrics(
    db: AsyncSession,
    company_id,
    prior_period: str,
) -> list[dict]:
    """Get metric names and values from the prior period."""
    from apps.api.models import ExtractedMetric

    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == prior_period,
        )
    )
    metrics = q.scalars().all()

    return [
        {
            "metric_name": m.metric_name,
            "metric_value": m.metric_value,
            "unit": m.unit,
            "segment": m.segment,
            "is_guidance": m.segment == "guidance" or (m.metric_name or "").startswith("GUIDANCE:"),
        }
        for m in metrics
    ]


async def _get_tracked_kpis(db: AsyncSession, company_id) -> list[dict]:
    """Get the analyst's tracked KPIs for this company."""
    from apps.api.models import TrackedKPI

    q = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == company_id)
        .order_by(TrackedKPI.display_order)
    )
    kpis = q.scalars().all()

    return [{"kpi_name": k.kpi_name, "unit": k.unit} for k in kpis]


# ═══════════════════════════════════════════════════════════════════
# Detection logic
# ═══════════════════════════════════════════════════════════════════

async def detect_disappeared(
    db: AsyncSession,
    company_id,
    current_period: str,
    current_items: list[dict],
) -> dict:
    """
    Compare current extraction against prior period to detect:
    1. Disappeared KPIs
    2. Silent guidance drops
    3. Guidance range changes

    Args:
        db: Database session
        company_id: Company UUID
        current_period: Current period label (e.g. "Q4 2025")
        current_items: Currently extracted items

    Returns:
        Dict with disappeared metrics, dropped guidance, range changes, and flags.
    """
    prior_period = _previous_period(current_period)
    if not prior_period:
        logger.info("Disappearance detector: cannot determine prior period for '%s'", current_period)
        return {"disappeared_kpis": [], "dropped_guidance": [], "guidance_range_changes": []}

    # Get prior period data
    prior_metrics = await _get_prior_metrics(db, company_id, prior_period)
    tracked_kpis = await _get_tracked_kpis(db, company_id)

    if not prior_metrics:
        logger.info("Disappearance detector: no prior data for %s", prior_period)
        return {"disappeared_kpis": [], "dropped_guidance": [], "guidance_range_changes": []}

    # Build name sets for comparison
    current_names = set()
    current_guidance = {}
    for item in current_items:
        name = _normalise_metric_name(item.get("metric_name", ""))
        if name:
            current_names.add(name)

        # Track guidance ranges
        if item.get("type") == "guidance" or item.get("category") == "guidance":
            g_name = _normalise_metric_name(item.get("metric_name", ""))
            if g_name:
                current_guidance[g_name] = {
                    "low": item.get("low"),
                    "high": item.get("high"),
                    "guidance_text": item.get("guidance_text", ""),
                }

    prior_names = set()
    prior_guidance = {}
    for m in prior_metrics:
        name = _normalise_metric_name(m["metric_name"])
        if name:
            prior_names.add(name)

        if m["is_guidance"]:
            g_name = _normalise_metric_name(m["metric_name"])
            if g_name:
                prior_guidance[g_name] = {
                    "value": m["metric_value"],
                    "unit": m["unit"],
                }

    # Build tracked KPI name set
    tracked_names = {_normalise_metric_name(k["kpi_name"]) for k in tracked_kpis}

    # ── 1. Disappeared KPIs ──────────────────────────────────
    disappeared = prior_names - current_names
    disappeared_kpis = []

    for name in disappeared:
        # Find the prior metric details
        prior_detail = next(
            (m for m in prior_metrics
             if _normalise_metric_name(m["metric_name"]) == name and not m["is_guidance"]),
            None,
        )
        if not prior_detail:
            continue

        # Higher severity if it's a tracked KPI
        is_tracked = any(
            _fuzzy_match(name, tn) for tn in tracked_names
        )

        severity = "high" if is_tracked else "medium"

        disappeared_kpis.append({
            "metric_name": prior_detail["metric_name"],
            "prior_value": prior_detail["metric_value"],
            "prior_unit": prior_detail["unit"],
            "prior_period": prior_period,
            "is_tracked_kpi": is_tracked,
            "severity": severity,
            "signal": f"'{prior_detail['metric_name']}' was reported in {prior_period} but is absent from {current_period}",
        })

    # ── 2. Silent guidance drops ─────────────────────────────
    dropped_guidance_names = set(prior_guidance.keys()) - set(current_guidance.keys())
    dropped_guidance = []

    for name in dropped_guidance_names:
        prior_g = prior_guidance[name]
        dropped_guidance.append({
            "metric_name": name,
            "prior_value": prior_g["value"],
            "prior_unit": prior_g["unit"],
            "prior_period": prior_period,
            "severity": "high",
            "signal": f"Guidance on '{name}' was given in {prior_period} but not updated in {current_period} — silently dropped?",
        })

    # ── 3. Guidance range changes ────────────────────────────
    guidance_range_changes = []
    common_guidance = set(prior_guidance.keys()) & set(current_guidance.keys())

    for name in common_guidance:
        prior_g = prior_guidance[name]
        current_g = current_guidance[name]

        if current_g.get("low") is not None and current_g.get("high") is not None:
            current_width = abs(current_g["high"] - current_g["low"])

            # Try to infer prior range width (if we stored it)
            # For now, just flag if current range exists
            if prior_g.get("value") is not None:
                # Check if guidance moved up or down
                mid = (current_g["low"] + current_g["high"]) / 2
                prior_val = prior_g["value"]

                if prior_val and mid:
                    try:
                        pv = float(prior_val)
                        direction = "raised" if mid > pv else "lowered" if mid < pv else "maintained"
                        guidance_range_changes.append({
                            "metric_name": name,
                            "prior_value": pv,
                            "current_low": current_g["low"],
                            "current_high": current_g["high"],
                            "current_midpoint": mid,
                            "direction": direction,
                            "range_width": current_width,
                            "signal": f"Guidance on '{name}' {direction}: was {pv}, now {current_g['low']}-{current_g['high']}",
                        })
                    except (ValueError, TypeError):
                        pass

    # ── Summary ──────────────────────────────────────────────
    total_flags = len(disappeared_kpis) + len(dropped_guidance)

    if total_flags:
        logger.info(
            "Disappearance detector: %d disappeared KPIs, %d dropped guidance, %d range changes (comparing %s vs %s)",
            len(disappeared_kpis), len(dropped_guidance), len(guidance_range_changes),
            current_period, prior_period,
        )

    return {
        "prior_period": prior_period,
        "current_period": current_period,
        "disappeared_kpis": disappeared_kpis,
        "dropped_guidance": dropped_guidance,
        "guidance_range_changes": guidance_range_changes,
        "total_flags": total_flags,
        "summary": _build_summary(disappeared_kpis, dropped_guidance, guidance_range_changes),
    }


def _fuzzy_match(name1: str, name2: str) -> bool:
    """Simple fuzzy match for metric names."""
    if not name1 or not name2:
        return False
    # Direct substring match
    if name1 in name2 or name2 in name1:
        return True
    # Word overlap
    words1 = set(name1.split())
    words2 = set(name2.split())
    if len(words1) > 1 and len(words2) > 1:
        overlap = words1 & words2
        return len(overlap) >= min(len(words1), len(words2)) * 0.6
    return False


def _build_summary(disappeared: list, dropped: list, range_changes: list) -> str:
    """Build a human-readable summary of disappearance signals."""
    parts = []

    if disappeared:
        high = [d for d in disappeared if d["severity"] == "high"]
        if high:
            names = ", ".join(d["metric_name"] for d in high[:5])
            parts.append(f"⚠️ {len(high)} tracked KPI(s) disappeared: {names}")
        medium = [d for d in disappeared if d["severity"] == "medium"]
        if medium:
            parts.append(f"{len(medium)} other metric(s) not reported this period")

    if dropped:
        names = ", ".join(d["metric_name"] for d in dropped[:5])
        parts.append(f"🔇 {len(dropped)} guidance item(s) silently dropped: {names}")

    if range_changes:
        raised = [r for r in range_changes if r["direction"] == "raised"]
        lowered = [r for r in range_changes if r["direction"] == "lowered"]
        if raised:
            parts.append(f"↑ {len(raised)} guidance raised")
        if lowered:
            parts.append(f"↓ {len(lowered)} guidance lowered")

    return " | ".join(parts) if parts else "No disappearance signals detected"
