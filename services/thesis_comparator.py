"""
Thesis Comparison Service (§7)

Responsibilities:
  - Compare current quarter vs previous quarter
  - Compare current quarter vs active thesis
  - Detect thesis strengthening / weakening signals
  - Persist event assessments
"""

import json
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Company, EventAssessment, ExtractedMetric, ReviewQueueItem, ThesisVersion,
)
from prompts import THESIS_COMPARATOR
from schemas import ThesisComparison
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


async def _get_active_thesis(db: AsyncSession, company_id: uuid.UUID) -> ThesisVersion | None:
    result = await db.execute(
        select(ThesisVersion)
        .where(ThesisVersion.company_id == company_id, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_metrics_for_period(db: AsyncSession, company_id: uuid.UUID, period: str) -> list[ExtractedMetric]:
    result = await db.execute(
        select(ExtractedMetric)
        .where(ExtractedMetric.company_id == company_id, ExtractedMetric.period_label == period)
    )
    return list(result.scalars().all())


def _metrics_to_text(metrics: list[ExtractedMetric]) -> str:
    lines = []
    for m in metrics:
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"- {m.metric_name}: {val}")
    return "\n".join(lines) if lines else "No data available."


def _previous_period(period_label: str) -> str:
    """Derive the prior quarter label, e.g. '2026_Q1' → '2025_Q4'."""
    try:
        year, q = period_label.split("_Q")
        q_int = int(q)
        if q_int == 1:
            return f"{int(year) - 1}_Q4"
        return f"{year}_Q{q_int - 1}"
    except Exception:
        return ""


async def compare_thesis(
    db: AsyncSession,
    company_id: uuid.UUID,
    document_id: uuid.UUID,
    period_label: str,
) -> ThesisComparison:
    """
    Run the thesis comparator prompt and persist an EventAssessment.
    """
    thesis = await _get_active_thesis(db, company_id)
    if thesis is None:
        raise ValueError("No active thesis found for this company.")

    current_metrics = await _get_metrics_for_period(db, company_id, period_label)
    prior_period = _previous_period(period_label)
    prior_metrics = await _get_metrics_for_period(db, company_id, prior_period) if prior_period else []

    prompt = THESIS_COMPARATOR.format(
        thesis=thesis.core_thesis,
        quarter_data=_metrics_to_text(current_metrics),
        prior_data=_metrics_to_text(prior_metrics),
    )
    data = call_llm_json(prompt)
    comparison = ThesisComparison(**data)

    # Persist assessment
    assessment = EventAssessment(
        id=uuid.uuid4(),
        company_id=company_id,
        document_id=document_id,
        event_type="earnings",
        thesis_direction=comparison.thesis_direction,
        surprise_level="minor",  # refined by surprise detector
        summary=comparison.summary,
        confidence=0.85,
        needs_review=True,
    )
    db.add(assessment)

    # Always goes to review queue
    db.add(ReviewQueueItem(
        id=uuid.uuid4(),
        entity_type="assessment",
        entity_id=assessment.id,
        queue_reason="Thesis comparison requires human review",
        priority="high",
    ))

    await db.commit()
    logger.info("Thesis comparison for %s / %s → %s", company_id, period_label, comparison.thesis_direction)

    # Write drift JSON
    from pathlib import Path
    from configs.settings import settings
    out_dir = Path(settings.storage_base_path) / "outputs"
    # Need ticker — fetch from company
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one()
    drift_dir = out_dir / company.ticker / period_label
    drift_dir.mkdir(parents=True, exist_ok=True)
    (drift_dir / "thesis_drift.json").write_text(comparison.model_dump_json(indent=2))

    return comparison
