"""
Company CRUD endpoints (§8).
"""

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db, get_company_or_404
from apps.api.models import Company, ThesisVersion
from schemas import CompanyCreate, CompanyOut, CompanyUpdate, ThesisCreate, ThesisOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyOut])
async def list_companies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).order_by(Company.ticker))
    return result.scalars().all()


def _clean_ticker(raw: str) -> str:
    """Clean ticker: uppercase and trim outer whitespace. Keeps exchange suffix (e.g. 'ENI IM')."""
    return raw.strip().upper()


@router.get("/{ticker}", response_model=CompanyOut)
async def get_company(ticker: str, db: AsyncSession = Depends(get_db)):
    clean = _clean_ticker(ticker)
    result = await db.execute(select(Company).where(Company.ticker == clean))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company


@router.post("", response_model=CompanyOut, status_code=201)
async def create_company(body: CompanyCreate, db: AsyncSession = Depends(get_db)):
    company = Company(id=uuid.uuid4(), **body.model_dump())
    company.ticker = _clean_ticker(company.ticker)
    db.add(company)
    await db.commit()
    await db.refresh(company)
    return company


@router.patch("/{ticker}", response_model=CompanyOut)
async def update_company(ticker: str, body: CompanyUpdate, db: AsyncSession = Depends(get_db)):
    clean = _clean_ticker(ticker)
    result = await db.execute(select(Company).where(Company.ticker == clean))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(company, field, value)
    await db.commit()
    await db.refresh(company)
    return company


# ─────────────────────────────────────────────────────────────────
# Thesis
# ─────────────────────────────────────────────────────────────────
@router.post("/{ticker}/thesis", response_model=ThesisOut, status_code=201)
async def create_thesis(ticker: str, body: ThesisCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Deactivate existing active theses
    existing = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
    )
    for tv in existing.scalars().all():
        tv.active = False

    thesis = ThesisVersion(
        id=uuid.uuid4(),
        company_id=company.id,
        **body.model_dump(),
        active=True,
    )
    db.add(thesis)
    await db.commit()
    await db.refresh(thesis)
    return thesis


@router.get("/{ticker}/thesis", response_model=list[ThesisOut])
async def list_theses(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    theses = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id).order_by(ThesisVersion.thesis_date.desc())
    )
    return theses.scalars().all()


@router.post("/{ticker}/seed-thesis", status_code=201)
async def seed_heineken_thesis(ticker: str, db: AsyncSession = Depends(get_db)):
    """One-click seed of the Heineken pilot thesis."""
    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    existing = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
    )
    if existing.scalar_one_or_none():
        return {"status": "thesis already exists"}
    thesis = ThesisVersion(
        id=uuid.uuid4(), company_id=company.id, thesis_date=date(2025, 12, 1),
        core_thesis="Heineken is a premium global brewer with pricing power and volume recovery tailwinds in emerging markets.",
        key_risks="1. Consumer downtrading in Europe\n2. FX headwinds in Africa\n3. Regulatory risk",
        valuation_framework="DCF with 7.5% WACC, cross-check EV/EBITDA 10-11x and FCF yield 6%+.",
        active=True,
    )
    db.add(thesis)
    await db.commit()
    return {"status": "thesis seeded", "thesis_id": str(thesis.id)}


@router.post("/{ticker}/generate-thesis", status_code=200)
async def generate_thesis(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Use the LLM to auto-generate an investment thesis based on
    extracted data already in the system, plus web knowledge.
    """
    from services.llm_client import call_llm_json

    result = await db.execute(select(Company).where(Company.ticker == _clean_ticker(ticker)))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Gather any extracted metrics for context
    from apps.api.models import ExtractedMetric
    from sqlalchemy import func
    metrics_q = await db.execute(
        select(ExtractedMetric)
        .where(ExtractedMetric.company_id == company.id, ExtractedMetric.confidence >= 0.9)
        .order_by(ExtractedMetric.created_at.desc())
        .limit(50)
    )
    metrics = metrics_q.scalars().all()
    metrics_text = "\n".join(
        f"- {m.metric_name}: {m.metric_value} {m.unit or ''} ({m.period_label})"
        for m in metrics
    ) if metrics else "No extracted data available yet."

    prompt = f"""\
You are a senior buy-side equity analyst. Generate a structured investment thesis
for the following company based on your knowledge and any data provided.

COMPANY: {company.name} ({company.ticker})
SECTOR: {company.sector or 'Unknown'}
COUNTRY: {company.country or 'Unknown'}

AVAILABLE DATA FROM OUR SYSTEM:
{metrics_text}

Generate a comprehensive investment thesis. Be specific, opinionated, and actionable.
This is for an internal investment memo, not a marketing document.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "core_thesis": "<3-4 sentences on why this is an attractive/unattractive investment>",
  "variant_perception": "<where does the market disagree with our view, and why are we right>",
  "key_risks": "<4-6 specific risks, numbered>",
  "debate_points": "<3-5 key debates around the stock, numbered>",
  "capital_allocation_view": "<how management allocates capital, and whether we agree>",
  "valuation_framework": "<preferred valuation approach with specific parameters>"
}}
"""
    try:
        thesis_data = call_llm_json(prompt, max_tokens=4096)
        return thesis_data
    except Exception as e:
        raise HTTPException(500, f"Thesis generation failed: {str(e)[:200]}")
