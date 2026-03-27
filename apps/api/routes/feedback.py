"""
Inline Analysis Feedback — capture analyst annotations on extracted output.

Endpoints:
  POST /feedback                    — save a tag/comment on a metric, section, snippet, or surprise
  GET  /feedback                    — list feedback for a company/period
  POST /feedback/promote            — bundle session feedback into an ABExperiment record
  GET  /feedback/summary/{ticker}   — aggregate feedback stats by prompt type
"""

import json
import uuid
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, ExtractionFeedback, ABExperiment, PromptVariant

router = APIRouter(tags=["feedback"])


# ── Schemas ──────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    ticker: str
    period_label: str
    section: str                          # e.g. "metric:revenue", "briefing:bottom_line", "snippet:0", "surprise:0"
    tag: str                              # "correct" | "wrong" | "imprecise" | "missing" | "hallucinated"
    comment: Optional[str] = None
    source_snippet: Optional[str] = None  # the original text being flagged
    metric_id: Optional[str] = None       # UUID of ExtractedMetric if applicable
    document_id: Optional[str] = None
    prompt_type: Optional[str] = None     # which prompt produced this output
    author: Optional[str] = None


class FeedbackPromote(BaseModel):
    ticker: str
    period_label: str
    prompt_type: str                      # which extractor/synthesiser to file against
    overall_rating: int                   # 1-5
    session_notes: Optional[str] = None  # analyst's summary of the session
    feedback_ids: Optional[list[str]] = None  # specific IDs to include (all if omitted)


# ── Save a single tag ─────────────────────────────────────────────

@router.post("/feedback", status_code=201)
async def save_feedback(body: FeedbackCreate, db: AsyncSession = Depends(get_db)):
    """Auto-save a tag/comment on any part of the analysis output."""
    comp_q = await db.execute(select(Company).where(Company.ticker == body.ticker))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {body.ticker} not found")

    fb = ExtractionFeedback(
        id=uuid.uuid4(),
        company_id=company.id,
        period_label=body.period_label,
        section=body.section,
        tag=body.tag,
        comment=body.comment,
        source_snippet=body.source_snippet,
        metric_id=uuid.UUID(body.metric_id) if body.metric_id else None,
        document_id=uuid.UUID(body.document_id) if body.document_id else None,
        prompt_type=body.prompt_type,
        author=body.author,
        promoted=True,  # auto-promote to Prompt Lab
    )
    db.add(fb)

    # Auto-promote: create ABExperiment record so feedback flows into Prompt Lab immediately
    prompt_type = body.prompt_type or "unknown"
    snippet_ref = f" [snippet: \"{body.source_snippet[:80]}…\"]" if body.source_snippet else ""
    comment_line = f": {body.comment}" if body.comment else ""
    feedback_text = (
        f"Inline feedback for {body.ticker} {body.period_label} "
        f"(prompt_type: {prompt_type}).\n"
        f"Tag: {body.tag}.\n"
        f"Section: {body.section}.\n"
        f"- [{body.tag.upper()}] {body.section}{snippet_ref}{comment_line}"
    )

    # Find active variant for this prompt type
    variant_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == prompt_type,
            PromptVariant.is_active == True,
        ).limit(1)
    )
    active_variant = variant_q.scalar_one_or_none()

    if active_variant:
        exp = ABExperiment(
            id=uuid.uuid4(),
            company_id=company.id,
            prompt_type=prompt_type,
            period_label=body.period_label,
            variant_a_id=active_variant.id,
            variant_b_id=active_variant.id,
            output_a=feedback_text,
            output_b=None,
            winner="tie",
            rating_a=3 if body.tag in ("correct", "good") else 1,
            rating_b=None,
            analyst_feedback=feedback_text,
            status="completed",
        )
        db.add(exp)

    await db.commit()

    return {
        "id": str(fb.id),
        "status": "saved_and_promoted",
        "tag": fb.tag,
        "section": fb.section,
        "promoted": True,
    }


# ── List feedback ─────────────────────────────────────────────────

@router.get("/feedback")
async def list_feedback(
    ticker: str,
    period_label: Optional[str] = None,
    promoted: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    comp_q = await db.execute(select(Company).where(Company.ticker == ticker))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    q = select(ExtractionFeedback).where(ExtractionFeedback.company_id == company.id)
    if period_label:
        q = q.where(ExtractionFeedback.period_label == period_label)
    if promoted is not None:
        q = q.where(ExtractionFeedback.promoted == promoted)
    q = q.order_by(ExtractionFeedback.created_at.desc()).limit(200)

    result = await db.execute(q)
    items = result.scalars().all()

    return [
        {
            "id": str(f.id),
            "period_label": f.period_label,
            "section": f.section,
            "tag": f.tag,
            "comment": f.comment,
            "source_snippet": f.source_snippet,
            "prompt_type": f.prompt_type,
            "author": f.author,
            "promoted": f.promoted,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        }
        for f in items
    ]


# ── Promote session feedback → ABExperiment ───────────────────────

@router.post("/feedback/promote")
async def promote_feedback(body: FeedbackPromote, db: AsyncSession = Depends(get_db)):
    """
    Bundle unpromoted feedback from a session into an ABExperiment record.
    This wires directly into the existing Prompt Lab refinement pipeline.
    """
    comp_q = await db.execute(select(Company).where(Company.ticker == body.ticker))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {body.ticker} not found")

    # Fetch the feedback items to promote
    q = select(ExtractionFeedback).where(
        ExtractionFeedback.company_id == company.id,
        ExtractionFeedback.period_label == body.period_label,
        ExtractionFeedback.promoted == False,
    )
    if body.prompt_type:
        q = q.where(ExtractionFeedback.prompt_type == body.prompt_type)
    if body.feedback_ids:
        fids = [uuid.UUID(fid) for fid in body.feedback_ids]
        q = q.where(ExtractionFeedback.id.in_(fids))

    result = await db.execute(q)
    items = result.scalars().all()

    if not items:
        raise HTTPException(400, "No unpromoted feedback found to promote")

    # Build structured feedback text for the LLM refinement pipeline
    tag_counts = {}
    comments = []
    sections_flagged = set()
    for f in items:
        tag_counts[f.tag] = tag_counts.get(f.tag, 0) + 1
        sections_flagged.add(f.section)
        if f.comment:
            snippet_ref = f" [snippet: \"{f.source_snippet[:80]}…\"]" if f.source_snippet else ""
            comments.append(f"- [{f.tag.upper()}] {f.section}{snippet_ref}: {f.comment}")

    tag_summary = ", ".join(f"{count}x {tag}" for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]))
    sections_summary = ", ".join(sorted(sections_flagged))

    feedback_text = (
        f"Session feedback for {body.ticker} {body.period_label} "
        f"(prompt_type: {body.prompt_type}).\n"
        f"Tags: {tag_summary}.\n"
        f"Sections flagged: {sections_summary}.\n"
        f"Analyst comments:\n" + "\n".join(comments[:30])
    )
    if body.session_notes:
        feedback_text += f"\n\nAnalyst session notes: {body.session_notes}"

    # Get the active variant for this prompt type
    variant_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == body.prompt_type,
            PromptVariant.is_active == True,
        ).limit(1)
    )
    active_variant = variant_q.scalar_one_or_none()

    # Create ABExperiment record (single-variant feedback, no B side)
    exp = ABExperiment(
        id=uuid.uuid4(),
        company_id=company.id,
        prompt_type=body.prompt_type,
        period_label=body.period_label,
        variant_a_id=active_variant.id if active_variant else None,
        variant_b_id=active_variant.id if active_variant else None,  # same — single variant feedback
        output_a=feedback_text,
        output_b=None,
        winner="tie",  # not a head-to-head — marks it as inline feedback
        rating_a=body.overall_rating,
        rating_b=None,
        analyst_feedback=feedback_text,
        status="completed",
    )
    db.add(exp)

    # Mark feedback items as promoted
    for f in items:
        f.promoted = True

    await db.commit()

    return {
        "status": "promoted",
        "experiment_id": str(exp.id),
        "items_promoted": len(items),
        "tag_summary": tag_counts,
        "message": f"Feedback promoted to Prompt Lab. Use /experiments/refine to generate an improved prompt.",
    }


# ── Feedback summary stats ────────────────────────────────────────

@router.get("/feedback/summary/{ticker}")
async def feedback_summary(ticker: str, db: AsyncSession = Depends(get_db)):
    """Aggregate feedback stats by prompt type — useful for prioritising which prompt to refine."""
    comp_q = await db.execute(select(Company).where(Company.ticker == ticker))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    result = await db.execute(
        select(ExtractionFeedback).where(ExtractionFeedback.company_id == company.id)
        .order_by(ExtractionFeedback.created_at.desc()).limit(500)
    )
    all_fb = result.scalars().all()

    by_type: dict[str, dict] = {}
    for f in all_fb:
        pt = f.prompt_type or "unknown"
        if pt not in by_type:
            by_type[pt] = {"total": 0, "tags": {}, "unpromoted": 0}
        by_type[pt]["total"] += 1
        by_type[pt]["tags"][f.tag] = by_type[pt]["tags"].get(f.tag, 0) + 1
        if not f.promoted:
            by_type[pt]["unpromoted"] += 1

    return {
        "ticker": ticker,
        "total_feedback": len(all_fb),
        "by_prompt_type": by_type,
    }
