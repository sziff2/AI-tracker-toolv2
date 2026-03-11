"""
Company CRUD endpoints (§8).
"""

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, ThesisVersion
from schemas import CompanyCreate, CompanyOut, CompanyUpdate, ThesisCreate, ThesisOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyOut])
async def list_companies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).order_by(Company.ticker))
    return result.scalars().all()


@router.get("/{ticker}", response_model=CompanyOut)
async def get_company(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company


@router.post("", response_model=CompanyOut, status_code=201)
async def create_company(body: CompanyCreate, db: AsyncSession = Depends(get_db)):
    company = Company(id=uuid.uuid4(), **body.model_dump())
    company.ticker = company.ticker.upper()
    db.add(company)
    await db.commit()
    await db.refresh(company)
    return company


@router.patch("/{ticker}", response_model=CompanyOut)
async def update_company(ticker: str, body: CompanyUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
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
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
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
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
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
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Check if thesis already exists
    existing = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
    )
    if existing.scalar_one_or_none():
        return {"status": "thesis already exists"}

    thesis = ThesisVersion(
        id=uuid.uuid4(),
        company_id=company.id,
        thesis_date=date(2025, 12, 1),
        core_thesis=(
            "Heineken is a premium global brewer with pricing power and "
            "volume recovery tailwinds in emerging markets. The EverGreen "
            "strategy should drive margin expansion through premiumisation "
            "and cost discipline. The stock is undervalued relative to "
            "the sector on a FCF yield basis."
        ),
        variant_perception=(
            "The market underestimates the margin improvement trajectory "
            "in Africa and Asia-Pacific, and overweights near-term input "
            "cost headwinds."
        ),
        key_risks=(
            "1. Prolonged consumer downtrading in Europe\n"
            "2. FX headwinds in Africa\n"
            "3. Regulatory risk (alcohol taxation)\n"
            "4. Integration risk in recent acquisitions"
        ),
        debate_points=(
            "1. Sustainability of premium mix shift\n"
            "2. Africa volume growth vs currency translation\n"
            "3. Capital allocation: M&A vs buybacks\n"
            "4. Input cost outlook (barley, aluminium, energy)"
        ),
        capital_allocation_view=(
            "Preference for organic growth reinvestment and selective bolt-on "
            "acquisitions. Dividend payout ~30-35% of net profit. "
            "Share buybacks only if FCF materially exceeds investment needs."
        ),
        valuation_framework=(
            "DCF with 7.5% WACC, 2% terminal growth. Cross-check with "
            "EV/EBITDA (target 10-11x forward) and FCF yield (target 6%+)."
        ),
        active=True,
    )
    db.add(thesis)
    await db.commit()
    return {"status": "thesis seeded", "thesis_id": str(thesis.id)}
