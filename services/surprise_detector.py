"""
Surprise Detection Service (§7)

Responsibilities:
  - Identify unexpected results vs prior guidance / expectations
  - Classify positive / negative surprises
  - Update EventAssessment surprise_level
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import EventAssessment, ExtractedMetric, ReviewQueueItem
from prompts import SURPRISE_DETECTOR
from schemas import SurpriseItem
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


def _build_expectations_text(guidance_metrics: list[ExtractedMetric]) -> str:
    lines = []
    for m in guidance_metrics:
        lines.append(f"- {m.metric_name}: {m.metric_text}")
    return "\n".join(lines) if lines else "No prior guidance available."


def _build_actuals_text(metrics: list[ExtractedMetric]) -> str:
    lines = []
    for m in metrics:
        if m.segment == "guidance":
            continue
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"- {m.metric_name}: {val}")
    return "\n".join(lines) if lines else "No actual results available."


async def detect_surprises(
    db: AsyncSession,
    company_id: uuid.UUID,
    document_id: uuid.UUID,
    period_label: str,
) -> list[SurpriseItem]:
    """
    Compare actuals vs guidance to find surprises.
    Updates the EventAssessment.surprise_level for the document.
    """
    # Gather guidance from prior period — try multiple candidates
    from services.thesis_comparator import _comparable_periods
    candidates = _comparable_periods(period_label)

    guidance_metrics = []
    for prior in candidates:
        guidance_q = await db.execute(
            select(ExtractedMetric).where(
                ExtractedMetric.company_id == company_id,
                ExtractedMetric.period_label == prior,
                ExtractedMetric.segment == "guidance",
            )
        )
        guidance_metrics = list(guidance_q.scalars().all())
        if guidance_metrics:
            break

    if not guidance_metrics and candidates:
        # Fall back to any metrics from prior period as expectations
        for prior in candidates:
            guidance_q = await db.execute(
                select(ExtractedMetric).where(
                    ExtractedMetric.company_id == company_id,
                    ExtractedMetric.period_label == prior,
                ).limit(20)
            )
            guidance_metrics = list(guidance_q.scalars().all())
            if guidance_metrics:
                break

    actuals_q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
        )
    )
    actual_metrics = list(actuals_q.scalars().all())

    prompt = SURPRISE_DETECTOR.format(
        expectations=_build_expectations_text(guidance_metrics),
        actuals=_build_actuals_text(actual_metrics),
    )

    # Use DB-active prompt variant if available
    try:
        from services.prompt_registry import get_active_prompt
        from apps.api.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db2:
            surprise_template = await get_active_prompt(db2, "surprise", SURPRISE_DETECTOR)
        prompt = surprise_template.format(
            expectations=_build_expectations_text(guidance_metrics),
            actuals=_build_actuals_text(actual_metrics),
        )
    except Exception:
        pass  # keep hardcoded prompt if registry fails
    raw = call_llm_json(prompt)
    if not isinstance(raw, list):
        raw = [raw]

    surprises = [SurpriseItem(**item) for item in raw]

    # Determine overall surprise level
    has_major = any(s.magnitude == "major" for s in surprises)
    overall_level = "major" if has_major else ("minor" if surprises else "none")

    # Update any EventAssessment for this document
    ea_q = await db.execute(
        select(EventAssessment).where(EventAssessment.document_id == document_id)
    )
    for ea in ea_q.scalars().all():
        ea.surprise_level = overall_level

    await db.commit()
    logger.info("Detected %d surprises (%s) for %s / %s", len(surprises), overall_level, company_id, period_label)
    return surprises
