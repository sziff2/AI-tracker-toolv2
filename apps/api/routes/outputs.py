"""
Output endpoints: history, briefings, IR questions, thesis drift.
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, Document, ExtractedMetric, EventAssessment, ResearchOutput, ThesisVersion
from schemas import ResearchOutputOut
from services.output_generator import (
    generate_briefing,
    generate_ir_questions,
    generate_thesis_drift_report,
)

router = APIRouter(tags=["outputs"])


# ─────────────────────────────────────────────────────────────────
# List all companies (for UI dropdown)
# ─────────────────────────────────────────────────────────────────
@router.get("/companies-list")
async def companies_list(db: AsyncSession = Depends(get_db)):
    """Return all companies with their thesis status for the UI dropdown."""
    result = await db.execute(select(Company).order_by(Company.name))
    companies = result.scalars().all()
    out = []
    for c in companies:
        # Check if thesis exists
        thesis_q = await db.execute(
            select(ThesisVersion).where(
                ThesisVersion.company_id == c.id, ThesisVersion.active == True
            ).limit(1)
        )
        has_thesis = thesis_q.scalar_one_or_none() is not None
        out.append({
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "industry": c.industry,
            "country": c.country,
            "primary_analyst": c.primary_analyst,
            "has_thesis": has_thesis,
        })
    return out


# ─────────────────────────────────────────────────────────────────
# Company history — all past analyses
# ─────────────────────────────────────────────────────────────────
@router.get("/companies/{ticker}/history")
async def company_history(ticker: str, db: AsyncSession = Depends(get_db)):
    """Return all past analyses for a company, grouped by period."""
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Get all research outputs
    outputs_q = await db.execute(
        select(ResearchOutput)
        .where(ResearchOutput.company_id == company.id)
        .order_by(ResearchOutput.created_at.desc())
    )
    outputs = outputs_q.scalars().all()

    # Get all documents
    docs_q = await db.execute(
        select(Document)
        .where(Document.company_id == company.id)
        .order_by(Document.created_at.desc())
    )
    docs = docs_q.scalars().all()

    # Get metric counts by period
    metrics_q = await db.execute(
        select(ExtractedMetric.period_label, __import__("sqlalchemy").func.count(ExtractedMetric.id))
        .where(ExtractedMetric.company_id == company.id)
        .group_by(ExtractedMetric.period_label)
    )
    metric_counts = dict(metrics_q.all())

    # Group by period
    periods = {}
    for doc in docs:
        p = doc.period_label or "unknown"
        if p not in periods:
            periods[p] = {"period": p, "documents": [], "analyses": [], "metrics_count": metric_counts.get(p, 0)}
        periods[p]["documents"].append({
            "id": str(doc.id),
            "title": doc.title,
            "document_type": doc.document_type,
            "parsing_status": doc.parsing_status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        })

    for out in outputs:
        p = out.period_label or "unknown"
        if p not in periods:
            periods[p] = {"period": p, "documents": [], "analyses": [], "metrics_count": 0}

        analysis_data = None
        if out.content_json:
            try:
                analysis_data = json.loads(out.content_json)
            except Exception:
                pass

        periods[p]["analyses"].append({
            "id": str(out.id),
            "output_type": out.output_type,
            "review_status": out.review_status,
            "created_at": out.created_at.isoformat() if out.created_at else None,
            "content": analysis_data,
        })

    # Get active thesis
    thesis_q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company.id, ThesisVersion.active == True
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()

    return {
        "company": {
            "ticker": company.ticker,
            "name": company.name,
            "sector": company.sector,
        },
        "thesis": {
            "core_thesis": thesis.core_thesis if thesis else None,
            "key_risks": thesis.key_risks if thesis else None,
            "valuation_framework": thesis.valuation_framework if thesis else None,
            "thesis_date": thesis.thesis_date.isoformat() if thesis else None,
        } if thesis else None,
        "periods": sorted(periods.values(), key=lambda x: x["period"], reverse=True),
    }


# ─────────────────────────────────────────────────────────────────
# Get a specific analysis by ID
# ─────────────────────────────────────────────────────────────────
@router.get("/analysis/{output_id}")
async def get_analysis(output_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ResearchOutput).where(ResearchOutput.id == output_id))
    output = result.scalar_one_or_none()
    if not output:
        raise HTTPException(404, "Analysis not found")

    content = None
    if output.content_json:
        try:
            content = json.loads(output.content_json)
        except Exception:
            pass

    return {
        "id": str(output.id),
        "period_label": output.period_label,
        "output_type": output.output_type,
        "review_status": output.review_status,
        "created_at": output.created_at.isoformat() if output.created_at else None,
        "content": content,
    }


# ─────────────────────────────────────────────────────────────────
# Existing endpoints
# ─────────────────────────────────────────────────────────────────
@router.get("/companies/{ticker}/outputs", response_model=list[ResearchOutputOut])
async def list_outputs(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchOutput)
        .join(Company)
        .where(Company.ticker == ticker.upper())
        .order_by(ResearchOutput.created_at.desc())
    )
    return result.scalars().all()


@router.post("/companies/{ticker}/generate-briefing")
async def briefing(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    b = await generate_briefing(db, company.id, period_label)
    return b.model_dump()


@router.post("/companies/{ticker}/generate-ir-questions")
async def ir_questions(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    questions = await generate_ir_questions(db, company.id, period_label)
    return [q.model_dump() for q in questions]


@router.post("/companies/{ticker}/generate-thesis-drift")
async def thesis_drift(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    try:
        drift = await generate_thesis_drift_report(db, company.id, period_label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return drift
