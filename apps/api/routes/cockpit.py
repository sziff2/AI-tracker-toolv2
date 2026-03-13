"""
Company Cockpit — single endpoint that returns everything
an analyst needs for one company in one call.
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company, Document, ExtractedMetric, ThesisVersion, EventAssessment,
    ResearchOutput, ReviewQueueItem, TrackedKPI, KPIScore,
    DecisionLog, AnalystNote,
)

router = APIRouter(tags=["cockpit"])


# ── Schemas ──────────────────────────────────────────────────
class DecisionCreate(BaseModel):
    action: str           # hold | add | trim | exit | initiate | watchlist
    rationale: str
    old_weight: Optional[float] = None
    new_weight: Optional[float] = None
    conviction: Optional[int] = None  # 1-5
    author: Optional[str] = None


class NoteCreate(BaseModel):
    note_type: str = "general"
    title: Optional[str] = None
    content: str
    author: Optional[str] = None


# ── Cockpit endpoint ─────────────────────────────────────────
@router.get("/companies/{ticker}/cockpit")
async def get_cockpit(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the full company cockpit: thesis, KPIs, latest results,
    research timeline, decision log, and pending review items.
    """
    # Company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    cid = company.id

    # ── Thesis ────────────────────────────────────────────────
    thesis_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == cid, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()
    thesis_data = None
    if thesis:
        thesis_data = {
            "id": str(thesis.id),
            "thesis_date": thesis.thesis_date.isoformat() if thesis.thesis_date else None,
            "core_thesis": thesis.core_thesis,
            "variant_perception": thesis.variant_perception,
            "key_risks": thesis.key_risks,
            "debate_points": thesis.debate_points,
            "capital_allocation_view": thesis.capital_allocation_view,
            "valuation_framework": thesis.valuation_framework,
        }

    # ── Latest thesis assessment ──────────────────────────────
    assess_q = await db.execute(
        select(EventAssessment).where(EventAssessment.company_id == cid)
        .order_by(EventAssessment.created_at.desc()).limit(1)
    )
    latest_assessment = assess_q.scalar_one_or_none()
    assessment_data = None
    if latest_assessment:
        assessment_data = {
            "thesis_direction": latest_assessment.thesis_direction,
            "surprise_level": latest_assessment.surprise_level,
            "summary": latest_assessment.summary,
            "confidence": float(latest_assessment.confidence) if latest_assessment.confidence else None,
        }

    # ── KPI tracker ───────────────────────────────────────────
    kpis_q = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == cid).order_by(TrackedKPI.display_order)
    )
    tracked_kpis = kpis_q.scalars().all()

    # Get all periods
    periods_q = await db.execute(
        select(ExtractedMetric.period_label).where(ExtractedMetric.company_id == cid)
        .distinct().order_by(ExtractedMetric.period_label)
    )
    periods = [p[0] for p in periods_q.all() if p[0]]

    # Get scores
    scores_q = await db.execute(select(KPIScore).where(KPIScore.company_id == cid))
    all_scores = scores_q.scalars().all()
    score_map = {}
    for s in all_scores:
        score_map[(str(s.tracked_kpi_id), s.period_label)] = {
            "value": float(s.value) if s.value is not None else None,
            "value_text": s.value_text,
            "score": s.score,
        }

    kpi_rows = []
    for kpi in tracked_kpis:
        row = {"id": str(kpi.id), "kpi_name": kpi.kpi_name, "unit": kpi.unit, "periods": {}}
        for period in periods:
            key = (str(kpi.id), period)
            if key in score_map:
                row["periods"][period] = score_map[key]
            else:
                # Auto-match
                match_q = await db.execute(
                    select(ExtractedMetric).where(
                        ExtractedMetric.company_id == cid,
                        ExtractedMetric.period_label == period,
                        ExtractedMetric.metric_name.ilike(f"%{kpi.kpi_name}%"),
                    ).limit(1)
                )
                m = match_q.scalar_one_or_none()
                if m:
                    row["periods"][period] = {
                        "value": float(m.metric_value) if m.metric_value else None,
                        "value_text": m.metric_text,
                        "score": None,
                    }
                else:
                    row["periods"][period] = {"value": None, "score": None}
        kpi_rows.append(row)

    # Overall scores per period
    overall_scores = {}
    for period in periods:
        vals = [score_map.get((str(k.id), period), {}).get("score") for k in tracked_kpis]
        valid = [v for v in vals if v is not None]
        overall_scores[period] = round(sum(valid) / len(valid), 1) if valid else None

    # ── All analyses grouped by period ─────────────────────────
    outputs_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == cid,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.created_at.desc())
    )
    all_outputs = outputs_q.scalars().all()

    # Group by period
    analyses_by_period = {}
    for o in all_outputs:
        p = o.period_label or "unknown"
        if p not in analyses_by_period:
            content = None
            if o.content_json:
                try:
                    content = json.loads(o.content_json)
                except Exception:
                    pass
            analyses_by_period[p] = {
                "period": p,
                "output_type": o.output_type,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "content": content,
            }

    # Latest briefing (for backwards compat)
    latest_briefing = None
    if analyses_by_period:
        latest_period = sorted(analyses_by_period.keys(), reverse=True)[0]
        latest_briefing = analyses_by_period[latest_period]

    # ── Documents timeline ────────────────────────────────────
    docs_q = await db.execute(
        select(Document).where(Document.company_id == cid)
        .order_by(Document.created_at.desc()).limit(20)
    )
    docs = [{
        "id": str(d.id), "title": d.title, "document_type": d.document_type,
        "period_label": d.period_label, "parsing_status": d.parsing_status,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in docs_q.scalars().all()]

    # ── Decision log ──────────────────────────────────────────
    decisions_q = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == cid)
        .order_by(DecisionLog.created_at.desc()).limit(20)
    )
    decisions = [{
        "id": str(d.id), "action": d.action, "rationale": d.rationale,
        "old_weight": float(d.old_weight) if d.old_weight else None,
        "new_weight": float(d.new_weight) if d.new_weight else None,
        "conviction": d.conviction, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in decisions_q.scalars().all()]

    # ── Analyst notes ─────────────────────────────────────────
    notes_q = await db.execute(
        select(AnalystNote).where(AnalystNote.company_id == cid)
        .order_by(AnalystNote.created_at.desc()).limit(20)
    )
    notes = [{
        "id": str(d.id), "note_type": d.note_type, "title": d.title,
        "content": d.content, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in notes_q.scalars().all()]

    # ── Review items ──────────────────────────────────────────
    review_q = await db.execute(
        select(ReviewQueueItem).where(
            ReviewQueueItem.status == "open"
        ).order_by(ReviewQueueItem.created_at.desc()).limit(10)
    )
    # Filter to this company's items by checking entity relationships
    reviews = [{
        "id": str(r.id), "entity_type": r.entity_type, "queue_reason": r.queue_reason,
        "priority": r.priority, "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in review_q.scalars().all()]

    # ── Metrics summary ───────────────────────────────────────
    metrics_count_q = await db.execute(
        select(func.count(ExtractedMetric.id)).where(ExtractedMetric.company_id == cid)
    )
    total_metrics = metrics_count_q.scalar() or 0

    # ── Thesis history ──────────────────────────────────────────
    all_theses_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == cid)
        .order_by(ThesisVersion.thesis_date.desc())
    )
    thesis_history = [{
        "id": str(t.id),
        "thesis_date": t.thesis_date.isoformat() if t.thesis_date else None,
        "core_thesis": t.core_thesis[:200] + "…" if t.core_thesis and len(t.core_thesis) > 200 else t.core_thesis,
        "active": t.active,
    } for t in all_theses_q.scalars().all()]

    # ── All thesis assessments ────────────────────────────────
    all_assessments_q = await db.execute(
        select(EventAssessment, Document.period_label)
        .join(Document, EventAssessment.document_id == Document.id)
        .where(EventAssessment.company_id == cid)
        .order_by(EventAssessment.created_at.desc())
    )
    all_assessments = [{
        "period": row[1],
        "thesis_direction": row[0].thesis_direction,
        "surprise_level": row[0].surprise_level,
        "summary": row[0].summary,
        "created_at": row[0].created_at.isoformat() if row[0].created_at else None,
    } for row in all_assessments_q.all()]

    # ── Group documents by period ─────────────────────────────
    docs_by_period = {}
    for d in docs:
        p = d["period_label"] or "unknown"
        if p not in docs_by_period:
            docs_by_period[p] = []
        docs_by_period[p].append(d)

    return {
        "company": {
            "ticker": company.ticker,
            "name": company.name,
            "sector": company.sector,
            "industry": company.industry,
            "country": company.country,
            "coverage_status": company.coverage_status,
            "primary_analyst": company.primary_analyst,
        },
        "thesis": thesis_data,
        "thesis_assessment": assessment_data,
        "thesis_history": thesis_history,
        "all_assessments": all_assessments,
        "kpi_tracker": {
            "periods": periods,
            "kpis": kpi_rows,
            "overall_scores": overall_scores,
        },
        "latest_briefing": latest_briefing,
        "analyses_by_period": analyses_by_period,
        "documents": docs,
        "docs_by_period": docs_by_period,
        "decisions": decisions,
        "notes": notes,
        "review_items": reviews,
        "stats": {
            "total_metrics": total_metrics,
            "total_documents": len(docs),
            "total_decisions": len(decisions),
            "periods_covered": len(periods),
        },
    }


# ── Decision log CRUD ────────────────────────────────────────
@router.post("/companies/{ticker}/decisions", status_code=201)
async def create_decision(ticker: str, body: DecisionCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    decision = DecisionLog(
        id=uuid.uuid4(), company_id=company.id,
        action=body.action, rationale=body.rationale,
        old_weight=body.old_weight, new_weight=body.new_weight,
        conviction=body.conviction, author=body.author,
    )
    db.add(decision)
    await db.commit()
    return {
        "id": str(decision.id), "action": decision.action,
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
    }


@router.get("/companies/{ticker}/decisions")
async def list_decisions(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    q = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == company.id)
        .order_by(DecisionLog.created_at.desc())
    )
    return [{
        "id": str(d.id), "action": d.action, "rationale": d.rationale,
        "old_weight": float(d.old_weight) if d.old_weight else None,
        "new_weight": float(d.new_weight) if d.new_weight else None,
        "conviction": d.conviction, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in q.scalars().all()]


# ── Analyst notes CRUD ───────────────────────────────────────
@router.post("/companies/{ticker}/notes", status_code=201)
async def create_note(ticker: str, body: NoteCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    note = AnalystNote(
        id=uuid.uuid4(), company_id=company.id,
        note_type=body.note_type, title=body.title,
        content=body.content, author=body.author,
    )
    db.add(note)
    await db.commit()
    return {
        "id": str(note.id), "title": note.title,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


@router.get("/companies/{ticker}/notes")
async def list_notes(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    q = await db.execute(
        select(AnalystNote).where(AnalystNote.company_id == company.id)
        .order_by(AnalystNote.created_at.desc())
    )
    return [{
        "id": str(d.id), "note_type": d.note_type, "title": d.title,
        "content": d.content, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in q.scalars().all()]
