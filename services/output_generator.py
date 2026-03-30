"""
Output Generation Service — briefings, IR questions, thesis drift.

Uses the context_builder to assemble minimum required context per step.
Each LLM call receives only what it needs — no full document dumps.
"""

import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Company, ExtractedMetric, EventAssessment, ResearchOutput, ReviewQueueItem, ThesisVersion
from configs.settings import settings
from prompts import ONE_PAGE_BRIEFING, IR_QUESTION_GENERATOR
from schemas import BriefingSection, IRQuestion
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


def _output_dir(ticker: str, period: str) -> Path:
    d = Path(settings.storage_base_path) / "outputs" / ticker / period
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _company_by_id(db: AsyncSession, company_id: uuid.UUID) -> Company:
    result = await db.execute(select(Company).where(Company.id == company_id))
    return result.scalar_one()


# ─────────────────────────────────────────────────────────────────
# One-Page Briefing (uses structured context)
# ─────────────────────────────────────────────────────────────────

async def generate_briefing(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> BriefingSection:
    from services.context_builder import build_briefing_context
    from services.prompt_registry import get_active_prompt

    company = await _company_by_id(db, company_id)
    ctx = await build_briefing_context(db, company_id, period_label)
    briefing_template = await get_active_prompt(db, "briefing", ONE_PAGE_BRIEFING)

    prompt = briefing_template.format(
        company=company.name,
        ticker=company.ticker,
        period=period_label,
        kpis=ctx["kpis"],
        thesis_comparison=f"Thesis: {ctx['thesis']}\n\nPrior period: {ctx['prior_period']}",
        surprises=f"Guidance items: {ctx['guidance']}\n\n{ctx['tracked_kpis']}",
    )
    data = call_llm_json(prompt, feature="briefing")
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
# IR Questions (uses structured context)
# ─────────────────────────────────────────────────────────────────

async def generate_ir_questions(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> list[IRQuestion]:
    from services.context_builder import build_briefing_context
    from services.prompt_registry import get_active_prompt

    company = await _company_by_id(db, company_id)
    ctx = await build_briefing_context(db, company_id, period_label)
    ir_template = await get_active_prompt(db, "ir_questions", IR_QUESTION_GENERATOR)

    prompt = ir_template.format(
        company=company.name,
        period=period_label,
        findings=f"Key metrics:\n{ctx['kpis']}\n\nGuidance:\n{ctx['guidance']}",
        thesis=ctx["thesis"],
    )
    raw = call_llm_json(prompt, feature="ir_questions")
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
# Thesis Drift Report
# ─────────────────────────────────────────────────────────────────

async def generate_thesis_drift_report(
    db: AsyncSession, company_id: uuid.UUID, period_label: str
) -> dict:
    company = await _company_by_id(db, company_id)
    drift_path = _output_dir(company.ticker, period_label) / "thesis_drift.json"

    if not drift_path.exists():
        from services.thesis_comparator import compare_thesis
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
