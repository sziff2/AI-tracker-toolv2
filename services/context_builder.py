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
    Company, Document, ExtractedMetric, ThesisVersion,
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
    db: AsyncSession, company_id, period: str, *, max_metrics: int = 25
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
    }


async def build_comparison_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """Minimum context for the thesis comparison step."""
    return {
        "thesis": await build_thesis_context(db, company_id),
        "current_kpis": await build_kpi_summary(db, company_id, period),
        "prior_period": await build_prior_period_context(db, company_id, period),
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
