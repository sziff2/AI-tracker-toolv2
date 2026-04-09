"""
Context Builder — implements context fatigue reduction.

Principles:
  1. Never pass information the model does not need right now
  2. Use layered summaries instead of raw data
  3. Compress context into structured key-value facts
  4. Each pipeline step gets minimum required context

This service sits between the database and the LLM prompts,
building focused, compressed context for each reasoning step.
"""

import json
import logging
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Company, Document, ExtractedMetric, ExtractionProfile, ThesisVersion,
    EventAssessment, ResearchOutput, TrackedKPI, KPIScore,
)

logger = logging.getLogger(__name__)


async def build_thesis_context(db: AsyncSession, company_id) -> str:
    """Return only the core thesis — not the full version history."""
    q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company_id,
            ThesisVersion.active == True,
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = q.scalar_one_or_none()
    if not thesis:
        return "No thesis on file."

    parts = [f"Core thesis: {thesis.core_thesis}"]
    if thesis.key_risks:
        parts.append(f"Key risks: {thesis.key_risks}")
    if thesis.valuation_framework:
        parts.append(f"Valuation: {thesis.valuation_framework}")
    return "\n".join(parts)


async def build_kpi_summary(
    db: AsyncSession, company_id, period: str, *, max_metrics: int = 50
) -> str:
    """
    Build a compressed KPI summary — structured key-value facts.
    Prioritises: high-confidence metrics, non-guidance, deduped.
    """
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.segment != "guidance",
        ).order_by(ExtractedMetric.confidence.desc()).limit(max_metrics)
    )
    metrics = q.scalars().all()

    if not metrics:
        return "No metrics extracted for this period."

    # Deduplicate by name (keep highest confidence)
    seen = set()
    lines = []
    for m in metrics:
        key = m.metric_name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"{m.metric_name}: {val}")

    return "\n".join(lines)


async def build_guidance_summary(
    db: AsyncSession, company_id, period: str
) -> str:
    """Extract only guidance items, compressed."""
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.segment == "guidance",
        ).order_by(ExtractedMetric.confidence.desc()).limit(15)
    )
    items = q.scalars().all()

    if not items:
        return "No guidance items found."

    lines = []
    for m in items:
        name = m.metric_name.replace("GUIDANCE: ", "")
        val = m.metric_text if m.metric_text else f"{m.metric_value} {m.unit}"
        lines.append(f"{name}: {val}")
    return "\n".join(lines)


async def build_prior_period_context(
    db: AsyncSession, company_id, current_period: str
) -> str:
    """
    Get a compressed summary of the prior period for comparison.
    Uses the stored research output if available (layered summary),
    otherwise falls back to raw metrics.
    """
    from services.thesis_comparator import _previous_period
    prior = _previous_period(current_period)

    # Try stored analysis first (already a summary)
    q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company_id,
            ResearchOutput.period_label == prior,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.created_at.desc()).limit(1)
    )
    output = q.scalar_one_or_none()

    if output and output.content_json:
        try:
            data = json.loads(output.content_json)
            # Extract just the bottom line and key metrics — not the full analysis
            parts = []
            briefing = data.get("briefing") or data.get("synthesis")
            if briefing:
                if isinstance(briefing, dict):
                    if briefing.get("bottom_line"):
                        parts.append(f"Prior bottom line: {briefing['bottom_line']}")
                    if briefing.get("what_happened"):
                        parts.append(f"Prior summary: {briefing['what_happened'][:300]}")
            if parts:
                return "\n".join(parts)
        except Exception:
            pass

    # Fall back to compressed metrics
    return await build_kpi_summary(db, company_id, prior, max_metrics=15)


async def build_tracked_kpi_context(
    db: AsyncSession, company_id
) -> str:
    """Build context from tracked KPIs with their scores — what the analyst cares about."""
    q = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == company_id)
        .order_by(TrackedKPI.display_order)
    )
    kpis = q.scalars().all()
    if not kpis:
        return ""

    lines = []
    for kpi in kpis:
        scores_q = await db.execute(
            select(KPIScore).where(KPIScore.tracked_kpi_id == kpi.id)
            .order_by(KPIScore.period_label.desc()).limit(3)
        )
        scores = scores_q.scalars().all()
        history = ", ".join(
            f"{s.period_label}: {s.value}{' (score '+str(s.score)+'/5)' if s.score else ''}"
            for s in scores
        )
        lines.append(f"{kpi.kpi_name}: {history}" if history else kpi.kpi_name)

    return "Tracked KPIs:\n" + "\n".join(lines)


async def build_confidence_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Build context from extraction confidence profiles — management language signals."""
    q = await db.execute(
        select(ExtractionProfile).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
        ).order_by(ExtractionProfile.created_at.desc()).limit(3)
    )
    profiles = q.scalars().all()
    if not profiles:
        return ""

    lines = []
    for p in profiles:
        cp = p.confidence_profile
        if not cp or not isinstance(cp, dict):
            continue
        signal = cp.get("overall_signal", "unknown")
        hedge = cp.get("hedge_rate", 0)
        one_off = cp.get("one_off_rate", 0)
        lines.append(
            f"Signal: {signal} | hedge_rate: {hedge:.0%} | one_off_rate: {one_off:.0%}"
        )
        if cp.get("hedge_terms_used"):
            lines.append(f"  Hedging language: {', '.join(cp['hedge_terms_used'][:5])}")
    return "Management language analysis:\n" + "\n".join(lines) if lines else ""


async def build_segment_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Build context from segment decomposition data."""
    q = await db.execute(
        select(ExtractionProfile).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
            ExtractionProfile.segment_data.isnot(None),
        ).order_by(ExtractionProfile.created_at.desc()).limit(1)
    )
    profile = q.scalar_one_or_none()
    if not profile or not profile.segment_data:
        return ""

    sd = profile.segment_data
    if not isinstance(sd, dict):
        return ""

    segments = sd.get("segments", [])
    if not segments:
        return ""

    lines = []
    for seg in segments[:10]:
        if isinstance(seg, dict):
            name = seg.get("name", "Unknown")
            rev = seg.get("revenue", "")
            margin = seg.get("margin", "")
            parts = [name]
            if rev:
                parts.append(f"rev={rev}")
            if margin:
                parts.append(f"margin={margin}")
            lines.append(" | ".join(parts))
    return "Segment breakdown:\n" + "\n".join(lines) if lines else ""


async def build_one_off_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Summarise one-off / non-recurring items for this period."""
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.is_one_off == True,
        ).order_by(ExtractedMetric.confidence.desc()).limit(15)
    )
    items = q.scalars().all()
    if not items:
        return ""

    lines = []
    for m in items:
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"[ONE-OFF] {m.metric_name}: {val}")
    return "Non-recurring items:\n" + "\n".join(lines)


async def build_mda_narrative_context(
    db: AsyncSession, company_id, period: str, *, max_chars: int = 8000
) -> str:
    """Return the stored MD&A narrative text (already captured during extraction)."""
    q = await db.execute(
        select(ExtractionProfile.mda_narrative).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
            ExtractionProfile.mda_narrative.isnot(None),
        ).order_by(ExtractionProfile.created_at.desc()).limit(1)
    )
    row = q.scalar_one_or_none()
    if not row:
        return ""
    return row[:max_chars]


async def build_briefing_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """
    Build the minimum context needed for the briefing step.
    Returns a dict with structured, compressed context.
    """
    return {
        "thesis": await build_thesis_context(db, company_id),
        "kpis": await build_kpi_summary(db, company_id, period),
        "guidance": await build_guidance_summary(db, company_id, period),
        "prior_period": await build_prior_period_context(db, company_id, period),
        "tracked_kpis": await build_tracked_kpi_context(db, company_id),
        "confidence": await build_confidence_context(db, company_id, period),
        "segments": await build_segment_context(db, company_id, period),
        "one_offs": await build_one_off_context(db, company_id, period),
    }


async def build_comparison_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """Minimum context for the thesis comparison step."""
    return {
        "thesis": await build_thesis_context(db, company_id),
        "current_kpis": await build_kpi_summary(db, company_id, period),
        "prior_period": await build_prior_period_context(db, company_id, period),
        "confidence": await build_confidence_context(db, company_id, period),
        "one_offs": await build_one_off_context(db, company_id, period),
    }


async def build_surprise_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """Minimum context for surprise detection."""
    from services.thesis_comparator import _previous_period
    prior = _previous_period(period)
    return {
        "guidance": await build_guidance_summary(db, company_id, prior),
        "actuals": await build_kpi_summary(db, company_id, period),
    }
