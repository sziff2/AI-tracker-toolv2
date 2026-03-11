"""
Output Generation Service (§7)

Responsibilities:
  - Generate one-page briefing
  - Generate IR questions
  - Generate thesis drift report
  - Persist outputs and add to review queue
"""

import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Company, ExtractedMetric, EventAssessment, ResearchOutput, ReviewQueueItem, ThesisVersion,
)
from configs.settings import settings
from prompts import ONE_PAGE_BRIEFING, IR_QUESTION_GENERATOR
from schemas import BriefingSection, IRQuestion, ThesisComparison
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


def _output_dir(ticker: str, period_label: str) -> Path:
    d = Path(settings.storage_base_path) / "outputs" / ticker / period_label
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _company_by_id(db: AsyncSession, company_id: uuid.UUID) -> Company:
    result = await db.execute(select(Company).where(Company.id == company_id))
    return result.scalar_one()


async def _metrics_text(db: AsyncSession, company_id: uuid.UUID, period: str) -> str:
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
        )
    )
    lines = []
    for m in q.scalars().all():
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"- {m.metric_name}: {val}")
    return "\n".join(lines) or "No metrics extracted."


async def _thesis_text(db: AsyncSession, company_id: uuid.UUID) -> str:
    q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company_id, ThesisVersion.active == True
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    tv = q.scalar_one_or_none()
    return tv.core_thesis if tv else "No thesis on file."


async def _drift_text(db: AsyncSession, company_id: uuid.UUID, period: str) -> str:
    q = await db.execute(
        select(EventAssessment).where(
            EventAssessment.company_id == company_id,
        ).order_by(EventAssessment.created_at.desc()).limit(1)
    )
    ea = q.scalar_one_or_none()
    if ea:
        return f"Direction: {ea.thesis_direction} | Surprise: {ea.surprise_level}\n{ea.summary}"
    return "No prior comparison available."


# ─────────────────────────────────────────────────────────────────
# One-Page Briefing
# ─────────────────────────────────────────────────────────────────

async def generate_briefing(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> BriefingSection:
    company = await _company_by_id(db, company_id)
    kpis = await _metrics_text(db, company_id, period_label)
    thesis_comp = await _drift_text(db, company_id, period_label)

    prompt = ONE_PAGE_BRIEFING.format(
        company=company.name,
        ticker=company.ticker,
        period=period_label,
        kpis=kpis,
        thesis_comparison=thesis_comp,
        surprises="See thesis comparison summary.",
    )
    data = call_llm_json(prompt)
    briefing = BriefingSection(**data)

    # Write markdown
    out = _output_dir(company.ticker, period_label)
    md_path = out / "briefing.md"
    md_content = f"""# {company.name} ({company.ticker}) — {period_label} Briefing

## What Happened
{briefing.what_happened}

## What Changed
{briefing.what_changed}

## Thesis Status
{briefing.thesis_status}

## Risks
{briefing.risks}

## Follow-ups
{briefing.follow_ups}

## Bottom Line
{briefing.bottom_line}
"""
    md_path.write_text(md_content)

    # Persist research output record
    ro = ResearchOutput(
        id=uuid.uuid4(),
        company_id=company_id,
        period_label=period_label,
        output_type="briefing",
        content_path=str(md_path),
        review_status="draft",
    )
    db.add(ro)
    db.add(ReviewQueueItem(
        id=uuid.uuid4(),
        entity_type="output",
        entity_id=ro.id,
        queue_reason="New briefing requires analyst review",
        priority="normal",
    ))
    await db.commit()
    logger.info("Generated briefing for %s / %s", company.ticker, period_label)
    return briefing


# ─────────────────────────────────────────────────────────────────
# IR Questions
# ─────────────────────────────────────────────────────────────────

async def generate_ir_questions(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> list[IRQuestion]:
    company = await _company_by_id(db, company_id)
    findings = await _metrics_text(db, company_id, period_label)
    thesis = await _thesis_text(db, company_id)

    prompt = IR_QUESTION_GENERATOR.format(
        company=company.name,
        period=period_label,
        findings=findings,
        thesis=thesis,
    )
    raw = call_llm_json(prompt)
    if not isinstance(raw, list):
        raw = [raw]
    questions = [IRQuestion(**item) for item in raw]

    # Write markdown
    out = _output_dir(company.ticker, period_label)
    md_path = out / "ir_questions.md"
    lines = [f"# IR Questions — {company.name} ({company.ticker}) — {period_label}\n"]
    for i, q in enumerate(questions, 1):
        lines.append(f"## {i}. {q.topic}")
        lines.append(f"**Question:** {q.question}")
        lines.append(f"**Rationale:** {q.rationale}\n")
    md_path.write_text("\n".join(lines))

    # Persist
    ro = ResearchOutput(
        id=uuid.uuid4(),
        company_id=company_id,
        period_label=period_label,
        output_type="ir_questions",
        content_path=str(md_path),
        review_status="draft",
    )
    db.add(ro)
    await db.commit()
    logger.info("Generated %d IR questions for %s / %s", len(questions), company.ticker, period_label)
    return questions


# ─────────────────────────────────────────────────────────────────
# Thesis Drift Report  (wraps thesis_comparator output into file)
# ─────────────────────────────────────────────────────────────────

async def generate_thesis_drift_report(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> dict:
    """
    Reads the already-generated thesis_drift.json and registers
    it as a research output. If not yet generated, runs the comparator.
    """
    company = await _company_by_id(db, company_id)
    drift_path = _output_dir(company.ticker, period_label) / "thesis_drift.json"

    if not drift_path.exists():
        from services.thesis_comparator import compare_thesis
        # Find latest document for this period
        from apps.api.models import Document
        doc_q = await db.execute(
            select(Document).where(
                Document.company_id == company_id,
                Document.period_label == period_label,
            ).limit(1)
        )
        doc = doc_q.scalar_one_or_none()
        if doc is None:
            raise ValueError("No document found for this period.")
        await compare_thesis(db, company_id, doc.id, period_label)

    drift_data = json.loads(drift_path.read_text())

    # Register output
    ro = ResearchOutput(
        id=uuid.uuid4(),
        company_id=company_id,
        period_label=period_label,
        output_type="thesis_drift",
        content_path=str(drift_path),
        review_status="draft",
    )
    db.add(ro)
    await db.commit()
    return drift_data
