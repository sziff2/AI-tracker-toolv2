"""
Public portfolio read-only routes.

Mounted under {PREFIX}/public. Authentication is handled by the auth
middleware in apps/api/main.py — these paths bypass the session cookie
check and instead require `Authorization: Bearer <LINKEDIN_AGENT_API_TOKEN>`.

Used by external browser tools (e.g. the LinkedIn agent) that cannot share
the session cookie. Read-only: lists portfolios and their holdings.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, Portfolio, PortfolioHolding

router = APIRouter(tags=["public-portfolio"])


@router.get("/portfolios")
async def list_portfolios(db: AsyncSession = Depends(get_db)):
    """Return all portfolios with name + active-holdings count."""
    result = await db.execute(select(Portfolio).order_by(Portfolio.name))
    portfolios = result.scalars().all()
    out = []
    for p in portfolios:
        count_q = await db.execute(
            select(func.count(PortfolioHolding.id))
            .where(
                PortfolioHolding.portfolio_id == p.id,
                PortfolioHolding.status == "active",
            )
        )
        out.append({
            "id": str(p.id),
            "name": p.name,
            "description": p.description,
            "currency": p.currency,
            "holdings_count": count_q.scalar() or 0,
        })
    return out


@router.get("/portfolios/{portfolio_id}/holdings")
async def list_holdings(portfolio_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return active holdings for a portfolio as a flat list of companies."""
    p_q = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
    if not p_q.scalar_one_or_none():
        raise HTTPException(404, "Portfolio not found")

    rows = await db.execute(
        select(
            Company.ticker, Company.name, Company.sector, Company.country,
            PortfolioHolding.weight,
        )
        .join(Company, PortfolioHolding.company_id == Company.id)
        .where(
            PortfolioHolding.portfolio_id == portfolio_id,
            PortfolioHolding.status == "active",
        )
        .order_by(PortfolioHolding.weight.desc())
    )
    return [
        {
            "ticker": r.ticker,
            "name": r.name,
            "sector": r.sector,
            "country": r.country,
            "weight": float(r.weight) if r.weight else 0,
        }
        for r in rows.all()
    ]


@router.get("/holdings")
async def list_all_holdings(db: AsyncSession = Depends(get_db)):
    """Return the deduped union of active holdings across every portfolio.

    Convenience endpoint so callers can pull "everything we cover" with a
    single request, without needing to know any portfolio_id.
    """
    rows = await db.execute(
        select(
            Company.ticker, Company.name, Company.sector, Company.country,
        )
        .join(PortfolioHolding, PortfolioHolding.company_id == Company.id)
        .where(PortfolioHolding.status == "active")
        .distinct()
        .order_by(Company.ticker)
    )
    return [
        {
            "ticker": r.ticker,
            "name": r.name,
            "sector": r.sector,
            "country": r.country,
        }
        for r in rows.all()
    ]
