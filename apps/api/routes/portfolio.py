"""
Portfolio & Valuation Module — multi-portfolio support, pricing, scenarios.

Endpoints:
  Portfolio CRUD:
    GET    /portfolios                              — list all
    POST   /portfolios                              — create
    GET    /portfolios/{id}                          — get with holdings
    PUT    /portfolios/{id}                          — update
    DELETE /portfolios/{id}                          — delete

  Holdings:
    POST   /portfolios/{id}/holdings                 — add holding
    PUT    /portfolios/{id}/holdings/{holding_id}     — update weight/cost
    DELETE /portfolios/{id}/holdings/{holding_id}     — remove

  Pricing:
    PUT    /companies/{ticker}/price                  — set current price
    GET    /companies/{ticker}/price                  — get latest price
    POST   /prices/bulk                               — bulk price update

  Valuation:
    GET    /companies/{ticker}/scenarios               — get bear/base/bull
    PUT    /companies/{ticker}/scenarios               — set/update scenarios
    GET    /companies/{ticker}/valuation-summary       — computed upside/downside/expected return

  Portfolio Analytics:
    GET    /portfolios/{id}/dashboard                 — portfolio-level analytics
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company, Portfolio, PortfolioHolding, PriceRecord, ValuationScenario, ScenarioSnapshot,
)

router = APIRouter(tags=["portfolio"])


# ── Schemas ──────────────────────────────────────────────────

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    benchmark: Optional[str] = None
    currency: str = "USD"

class HoldingCreate(BaseModel):
    ticker: str
    weight: float = 0
    cost_basis: Optional[float] = None
    shares: Optional[float] = None
    status: str = "active"

class HoldingUpdate(BaseModel):
    weight: Optional[float] = None
    cost_basis: Optional[float] = None
    shares: Optional[float] = None
    status: Optional[str] = None

class PriceUpdate(BaseModel):
    price: float
    currency: str = "USD"
    source: str = "manual"

class BulkPriceRow(BaseModel):
    ticker: str
    price: float
    currency: str = "USD"
    price_date: Optional[str] = None  # ISO format, defaults to now

class ScenarioInput(BaseModel):
    scenario_type: str          # bear | base | bull
    probability: Optional[float] = None
    target_price: Optional[float] = None
    currency: str = "USD"
    methodology: Optional[str] = None
    methodology_detail: Optional[str] = None
    key_assumptions: Optional[str] = None
    time_horizon: str = "12m"
    author: Optional[str] = None

class ScenariosUpdate(BaseModel):
    scenarios: list[ScenarioInput]


# ── Helpers ──────────────────────────────────────────────────

async def _get_company(db, ticker):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company

async def _get_latest_price(db, company_id):
    q = await db.execute(
        select(PriceRecord).where(PriceRecord.company_id == company_id)
        .order_by(PriceRecord.price_date.desc()).limit(1)
    )
    return q.scalar_one_or_none()


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO CRUD
# ═══════════════════════════════════════════════════════════════

@router.get("/portfolios")
async def list_portfolios(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Portfolio).order_by(Portfolio.name))
    portfolios = result.scalars().all()
    out = []
    for p in portfolios:
        holdings_q = await db.execute(
            select(func.count(PortfolioHolding.id), func.sum(PortfolioHolding.weight))
            .where(PortfolioHolding.portfolio_id == p.id, PortfolioHolding.status == "active")
        )
        row = holdings_q.one()
        out.append({
            "id": str(p.id), "name": p.name, "description": p.description,
            "benchmark": p.benchmark, "currency": p.currency, "is_active": p.is_active,
            "holdings_count": row[0] or 0, "total_weight": float(row[1] or 0),
        })
    return out


@router.post("/portfolios", status_code=201)
async def create_portfolio(body: PortfolioCreate, db: AsyncSession = Depends(get_db)):
    p = Portfolio(id=uuid.uuid4(), name=body.name, description=body.description,
                  benchmark=body.benchmark, currency=body.currency)
    db.add(p)
    await db.commit()
    return {"id": str(p.id), "name": p.name}


@router.get("/portfolios/{portfolio_id}")
async def get_portfolio(portfolio_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Portfolio not found")

    # Use column-level select to avoid loading Company relationships
    from sqlalchemy import func
    holdings_q = await db.execute(
        select(
            PortfolioHolding.id, PortfolioHolding.weight, PortfolioHolding.cost_basis,
            PortfolioHolding.shares, PortfolioHolding.status, PortfolioHolding.company_id,
            Company.ticker, Company.name, Company.sector, Company.country,
        )
        .join(Company, PortfolioHolding.company_id == Company.id)
        .where(PortfolioHolding.portfolio_id == p.id)
        .order_by(PortfolioHolding.weight.desc())
    )
    rows = holdings_q.all()

    # Batch fetch all prices and scenarios in 2 queries instead of 2*N
    company_ids = list(set(r.company_id for r in rows))
    prices_by_co = {}
    scenarios_by_co_raw = []

    if company_ids:
        prices_q = await db.execute(
            select(PriceRecord.company_id, PriceRecord.price, PriceRecord.currency, PriceRecord.price_date)
            .where(PriceRecord.company_id.in_(company_ids))
            .order_by(PriceRecord.price_date.desc())
        )
        for pr in prices_q.all():
            if pr.company_id not in prices_by_co:
                prices_by_co[pr.company_id] = pr

        scenarios_q = await db.execute(
            select(ValuationScenario).where(ValuationScenario.company_id.in_(company_ids))
        )
        scenarios_by_co_raw = scenarios_q.scalars().all()

    # Build scenarios dict
    scenarios_by_co = {}
    for s in scenarios_by_co_raw:
        if s.company_id not in scenarios_by_co:
            scenarios_by_co[s.company_id] = {}
        scenarios_by_co[s.company_id][s.scenario_type] = {
            "target_price": float(s.target_price) if s.target_price else None,
            "probability": float(s.probability) if s.probability else None,
            "methodology": s.methodology,
        }

    holdings = []
    for r in rows:
        price_rec = prices_by_co.get(r.company_id)
        current_price = float(price_rec.price) if price_rec else None
        scenarios = scenarios_by_co.get(r.company_id, {})

        # Compute returns
        upside = downside = expected_return = None
        if current_price and current_price > 0:
            if scenarios.get("bull", {}).get("target_price"):
                upside = round((scenarios["bull"]["target_price"] / current_price - 1) * 100, 1)
            if scenarios.get("bear", {}).get("target_price"):
                downside = round((scenarios["bear"]["target_price"] / current_price - 1) * 100, 1)
            # Probability-weighted expected return
            ev = 0
            total_prob = 0
            for st in ["bear", "base", "bull"]:
                s = scenarios.get(st, {})
                if s.get("target_price") and s.get("probability"):
                    ev += s["target_price"] * s["probability"] / 100
                    total_prob += s["probability"]
            if total_prob > 0 and current_price > 0:
                expected_return = round((ev / current_price - 1) * 100, 1)

        holdings.append({
            "id": str(r.id), "ticker": r.ticker, "name": r.name,
            "sector": r.sector, "country": r.country,
            "weight": float(r.weight) if r.weight else 0,
            "cost_basis": float(r.cost_basis) if r.cost_basis else None,
            "shares": float(r.shares) if r.shares else None,
            "status": r.status,
            "current_price": current_price,
            "currency": price_rec.currency if price_rec else None,
            "scenarios": scenarios,
            "upside": upside, "downside": downside,
            "expected_return": expected_return,
        })

    total_weight = sum(h["weight"] for h in holdings)
    cash = round(100 - total_weight, 2)

    return {
        "id": str(p.id), "name": p.name, "description": p.description,
        "benchmark": p.benchmark, "currency": p.currency,
        "holdings": holdings, "total_weight": total_weight, "cash": cash,
    }


@router.delete("/portfolios/{portfolio_id}")
async def delete_portfolio(portfolio_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id))
    await db.execute(delete(Portfolio).where(Portfolio.id == portfolio_id))
    await db.commit()
    return {"status": "deleted"}


# ═══════════════════════════════════════════════════════════════
# HOLDINGS
# ═══════════════════════════════════════════════════════════════

@router.post("/portfolios/{portfolio_id}/holdings", status_code=201)
async def add_holding(portfolio_id: uuid.UUID, body: HoldingCreate, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, body.ticker)
    h = PortfolioHolding(
        id=uuid.uuid4(), portfolio_id=portfolio_id, company_id=company.id,
        weight=body.weight, cost_basis=body.cost_basis, shares=body.shares,
        status=body.status, date_added=datetime.now(timezone.utc),
    )
    db.add(h)
    await db.commit()
    from services.portfolio_analytics import flush_cache
    flush_cache()
    return {"id": str(h.id), "ticker": company.ticker}


@router.put("/portfolios/{portfolio_id}/holdings/{holding_id}")
async def update_holding(portfolio_id: uuid.UUID, holding_id: uuid.UUID, body: HoldingUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(PortfolioHolding).where(PortfolioHolding.id == holding_id))
    h = result.scalar_one_or_none()
    if not h:
        raise HTTPException(404, "Holding not found")
    if body.weight is not None: h.weight = body.weight
    if body.cost_basis is not None: h.cost_basis = body.cost_basis
    if body.shares is not None: h.shares = body.shares
    if body.status is not None: h.status = body.status
    await db.commit()
    return {"status": "updated"}


@router.delete("/portfolios/{portfolio_id}/holdings/{holding_id}")
async def remove_holding(portfolio_id: uuid.UUID, holding_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(PortfolioHolding).where(PortfolioHolding.id == holding_id))
    await db.commit()
    from services.portfolio_analytics import flush_cache
    flush_cache()
    return {"status": "deleted"}


# Bulk import holdings
class BulkHoldingRow(BaseModel):
    ticker: str
    weight: float = 0
    cost_basis: Optional[float] = None
    status: str = "active"
    name: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None

class BulkHoldingsImport(BaseModel):
    holdings: list[BulkHoldingRow]

@router.post("/portfolios/{portfolio_id}/holdings/bulk")
async def bulk_import_holdings(portfolio_id: uuid.UUID, body: BulkHoldingsImport, db: AsyncSession = Depends(get_db)):
    results = []
    for row in body.holdings:
        # Look up company; auto-create if name provided and not found
        co_q = await db.execute(select(Company).where(Company.ticker == row.ticker.upper()))
        company = co_q.scalar_one_or_none()
        if not company:
            if row.name:
                company = Company(
                    id=uuid.uuid4(),
                    ticker=row.ticker.upper(),
                    name=row.name,
                    sector=row.sector or None,
                    country=row.country or None,
                    coverage_status="active",
                )
                db.add(company)
                await db.flush()
                auto_created = True
            else:
                results.append({"ticker": row.ticker, "status": "company_not_found"})
                continue
        else:
            auto_created = False
            # Update sector/country/name if provided in import
            if row.sector:
                company.sector = row.sector
            if row.country:
                company.country = row.country
            if row.name:
                company.name = row.name

        # Upsert holding
        existing_q = await db.execute(
            select(PortfolioHolding).where(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.company_id == company.id,
            )
        )
        existing = existing_q.scalar_one_or_none()
        if existing:
            existing.weight = row.weight
            if row.cost_basis is not None: existing.cost_basis = row.cost_basis
            existing.status = row.status
            results.append({"ticker": row.ticker, "status": "company_created_and_updated" if auto_created else "updated"})
        else:
            h = PortfolioHolding(
                id=uuid.uuid4(), portfolio_id=portfolio_id, company_id=company.id,
                weight=row.weight, cost_basis=row.cost_basis, status=row.status,
                date_added=datetime.now(timezone.utc),
            )
            db.add(h)
            results.append({"ticker": row.ticker, "status": "company_created" if auto_created else "created"})

    await db.commit()
    imported = len([r for r in results if r["status"] not in ("company_not_found",)])
    return {"imported": imported, "results": results}


# ═══════════════════════════════════════════════════════════════
# PRICING
# ═══════════════════════════════════════════════════════════════

@router.put("/companies/{ticker}/price")
async def set_price(ticker: str, body: PriceUpdate, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    rec = PriceRecord(
        id=uuid.uuid4(), company_id=company.id, price=body.price,
        currency=body.currency, price_date=datetime.now(timezone.utc),
        source=body.source,
    )
    db.add(rec)
    await db.commit()
    return {"ticker": ticker, "price": body.price}


@router.get("/companies/{ticker}/price")
async def get_price(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    rec = await _get_latest_price(db, company.id)
    if not rec:
        return {"ticker": ticker, "price": None}
    return {
        "ticker": ticker, "price": float(rec.price), "currency": rec.currency,
        "price_date": rec.price_date.isoformat() if rec.price_date else None,
        "source": rec.source,
    }


@router.post("/prices/bulk")
async def bulk_price_update(body: list[BulkPriceRow], db: AsyncSession = Depends(get_db)):
    results = []
    for row in body:
        try:
            company = await _get_company(db, row.ticker)
            pd = datetime.now(timezone.utc)
            if row.price_date:
                try:
                    pd = datetime.fromisoformat(row.price_date.replace("Z", "+00:00"))
                except ValueError:
                    pass
            rec = PriceRecord(
                id=uuid.uuid4(), company_id=company.id, price=row.price,
                currency=row.currency, price_date=pd,
                source="bulk_import",
            )
            db.add(rec)
            results.append({"ticker": row.ticker, "status": "updated"})
        except HTTPException:
            results.append({"ticker": row.ticker, "status": "not_found"})
    await db.commit()
    return {"updated": len([r for r in results if r["status"] == "updated"]), "results": results}


@router.post("/prices/refresh")
async def trigger_price_refresh(ticker: str = None):
    """Refresh prices from Yahoo Finance. Optionally for one ticker only."""
    from services.price_feed import refresh_prices
    tickers = [ticker.upper()] if ticker else None
    result = await refresh_prices(tickers=tickers)
    return result


@router.post("/companies/{ticker}/backload-snapshots")
async def backload_scenario_snapshots(
    ticker: str,
    days: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """Backload scenario snapshots using Yahoo Finance historical prices.
    Creates one snapshot per scenario per day for the last N trading days."""
    company = await _get_company(db, ticker)
    from services.price_feed import bloomberg_to_yahoo
    import httpx

    yahoo_ticker = bloomberg_to_yahoo(ticker.upper())
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}",
            params={"range": f"{days * 2}d", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Yahoo returned {resp.status_code}")
        chart = resp.json().get("chart", {}).get("result", [])
        if not chart:
            raise HTTPException(502, "No chart data from Yahoo")

    timestamps = chart[0].get("timestamp", [])
    closes = chart[0]["indicators"]["quote"][0].get("close", [])
    currency = chart[0]["meta"].get("currency", "USD")

    # Get current scenarios
    scenarios_q = await db.execute(
        select(ValuationScenario).where(ValuationScenario.company_id == company.id)
    )
    scenarios = [s for s in scenarios_q.scalars().all() if s.target_price is not None]
    if not scenarios:
        raise HTTPException(400, "No scenarios set for this company")

    # Insert price records + scenario snapshots for each day
    count = 0
    for ts, close in zip(timestamps[-days:], closes[-days:]):
        if close is None:
            continue
        snap_date = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Historical price record
        db.add(PriceRecord(
            id=uuid.uuid4(), company_id=company.id,
            price=close, currency=currency, price_date=snap_date,
            source="backload",
        ))

        # Scenario snapshot per scenario
        for s in scenarios:
            db.add(ScenarioSnapshot(
                id=uuid.uuid4(), company_id=company.id,
                snapshot_date=snap_date, scenario_type=s.scenario_type,
                target_price=float(s.target_price),
                probability=float(s.probability) if s.probability else None,
                current_price=close, currency=currency,
                source="backload",
            ))
            count += 1

    await db.commit()
    return {"ticker": ticker, "days": len(timestamps[-days:]), "snapshots_created": count}


# ═══════════════════════════════════════════════════════════════
# VALUATION SCENARIOS
# ═══════════════════════════════════════════════════════════════

@router.get("/companies/{ticker:path}/scenarios")
async def get_scenarios(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    result = await db.execute(
        select(ValuationScenario).where(ValuationScenario.company_id == company.id)
    )
    return [{
        "id": str(s.id), "scenario_type": s.scenario_type,
        "probability": float(s.probability) if s.probability else None,
        "target_price": float(s.target_price) if s.target_price else None,
        "currency": s.currency, "methodology": s.methodology,
        "methodology_detail": s.methodology_detail,
        "key_assumptions": s.key_assumptions, "time_horizon": s.time_horizon,
        "author": s.author,
        "last_reviewed": s.last_reviewed.isoformat() if s.last_reviewed else None,
    } for s in result.scalars().all()]


@router.put("/companies/{ticker:path}/scenarios")
async def set_scenarios(ticker: str, body: ScenariosUpdate, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    # Get current price for snapshot
    price_rec = await _get_latest_price(db, company.id)
    current_price = float(price_rec.price) if price_rec else None
    now = datetime.now(timezone.utc)
    # Delete existing and replace
    await db.execute(delete(ValuationScenario).where(ValuationScenario.company_id == company.id))
    for s in body.scenarios:
        db.add(ValuationScenario(
            id=uuid.uuid4(), company_id=company.id, scenario_type=s.scenario_type,
            probability=s.probability, target_price=s.target_price, currency=s.currency,
            methodology=s.methodology, methodology_detail=s.methodology_detail,
            key_assumptions=s.key_assumptions, time_horizon=s.time_horizon,
            author=s.author, last_reviewed=now,
        ))
        # Save snapshot for historical tracking
        if s.target_price is not None:
            db.add(ScenarioSnapshot(
                id=uuid.uuid4(), company_id=company.id, snapshot_date=now,
                scenario_type=s.scenario_type, target_price=s.target_price,
                probability=s.probability, current_price=current_price,
                currency=s.currency or "USD", source="manual",
            ))
    await db.commit()
    return {"status": "saved", "scenarios": len(body.scenarios)}


@router.get("/companies/{ticker}/valuation-summary")
async def valuation_summary(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    price_rec = await _get_latest_price(db, company.id)
    current_price = float(price_rec.price) if price_rec else None

    scenarios_q = await db.execute(
        select(ValuationScenario).where(ValuationScenario.company_id == company.id)
    )
    scenarios = {s.scenario_type: s for s in scenarios_q.scalars().all()}

    result = {"ticker": ticker, "current_price": current_price, "currency": price_rec.currency if price_rec else None}

    for st in ["bear", "base", "bull"]:
        s = scenarios.get(st)
        if s and s.target_price:
            tp = float(s.target_price)
            ret = round((tp / current_price - 1) * 100, 1) if current_price and current_price > 0 else None
            result[st] = {
                "target_price": tp, "probability": float(s.probability) if s.probability else None,
                "return_pct": ret, "methodology": s.methodology,
                "methodology_detail": s.methodology_detail,
            }
        else:
            result[st] = None

    # Expected return
    ev = 0
    total_prob = 0
    for st in ["bear", "base", "bull"]:
        if result.get(st) and result[st].get("target_price") and result[st].get("probability"):
            ev += result[st]["target_price"] * result[st]["probability"] / 100
            total_prob += result[st]["probability"]
    result["expected_return"] = round((ev / current_price - 1) * 100, 1) if total_prob > 0 and current_price else None
    result["upside_downside_ratio"] = None
    if result.get("bull") and result.get("bear") and result["bull"].get("return_pct") and result["bear"].get("return_pct"):
        if result["bear"]["return_pct"] != 0:
            result["upside_downside_ratio"] = round(abs(result["bull"]["return_pct"] / result["bear"]["return_pct"]), 2)

    return result


@router.get("/companies/{ticker}/scenario-history")
async def scenario_history(ticker: str, db: AsyncSession = Depends(get_db)):
    """Get historical scenario snapshots for charting over time.
    Always includes current scenarios as the latest data point."""
    company = await _get_company(db, ticker)
    result = await db.execute(
        select(ScenarioSnapshot)
        .where(ScenarioSnapshot.company_id == company.id)
        .order_by(ScenarioSnapshot.snapshot_date.asc())
    )
    snapshots = result.scalars().all()
    out = [{
        "date": s.snapshot_date.isoformat() if s.snapshot_date else None,
        "scenario_type": s.scenario_type,
        "target_price": float(s.target_price) if s.target_price else None,
        "probability": float(s.probability) if s.probability else None,
        "current_price": float(s.current_price) if s.current_price else None,
    } for s in snapshots]

    # Append current scenarios as "now" only if no snapshots exist yet
    # (once snapshots exist, they already include the latest state)
    if not snapshots:
        now = datetime.now(timezone.utc).isoformat()
        price_rec = await _get_latest_price(db, company.id)
        current_price = float(price_rec.price) if price_rec else None
        scenarios_q = await db.execute(
            select(ValuationScenario).where(ValuationScenario.company_id == company.id)
        )
        for s in scenarios_q.scalars().all():
            if s.target_price is not None:
                out.append({
                    "date": now,
                    "scenario_type": s.scenario_type,
                    "target_price": float(s.target_price),
                    "probability": float(s.probability) if s.probability else None,
                    "current_price": current_price,
                })

    return out


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO DASHBOARD
# ═══════════════════════════════════════════════════════════════

@router.get("/portfolios/{portfolio_id}/dashboard")
async def portfolio_dashboard(portfolio_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Portfolio-level analytics: weighted returns, sector/country concentration, risk."""
    result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Portfolio not found")

    # Column-level select to avoid loading Company relationships
    holdings_q = await db.execute(
        select(
            PortfolioHolding.id, PortfolioHolding.weight, PortfolioHolding.company_id,
            Company.ticker, Company.name, Company.sector, Company.country,
        )
        .join(Company, PortfolioHolding.company_id == Company.id)
        .where(PortfolioHolding.portfolio_id == p.id, PortfolioHolding.status == "active")
        .order_by(PortfolioHolding.weight.desc())
    )
    rows = holdings_q.all()

    # Batch fetch prices and scenarios
    company_ids = list(set(r.company_id for r in rows))
    prices_q = await db.execute(
        select(PriceRecord.company_id, PriceRecord.price, PriceRecord.currency, PriceRecord.price_date)
        .where(PriceRecord.company_id.in_(company_ids))
        .order_by(PriceRecord.price_date.desc())
    )
    prices_by_co = {}
    scenarios_by_co = {}
    if company_ids:
        for pr in prices_q.all():
            if pr.company_id not in prices_by_co:
                prices_by_co[pr.company_id] = pr
        scenarios_q = await db.execute(
            select(ValuationScenario).where(ValuationScenario.company_id.in_(company_ids))
        )
        for s in scenarios_q.scalars().all():
            if s.company_id not in scenarios_by_co:
                scenarios_by_co[s.company_id] = {}
            scenarios_by_co[s.company_id][s.scenario_type] = s

    holdings = []
    sector_weights = {}
    country_weights = {}
    total_weight = 0
    weighted_expected_return = 0
    weighted_downside = 0

    for r in rows:
        w = float(r.weight or 0)
        total_weight += w

        sec = r.sector or "Other"
        cty = r.country or "Other"
        sector_weights[sec] = sector_weights.get(sec, 0) + w
        country_weights[cty] = country_weights.get(cty, 0) + w

        price_rec = prices_by_co.get(r.company_id)
        cp = float(price_rec.price) if price_rec else None
        scenarios = scenarios_by_co.get(r.company_id, {})

        upside = downside = expected_ret = None
        if cp and cp > 0:
            if scenarios.get("bull") and scenarios["bull"].target_price:
                upside = round((float(scenarios["bull"].target_price) / cp - 1) * 100, 1)
            if scenarios.get("bear") and scenarios["bear"].target_price:
                downside = round((float(scenarios["bear"].target_price) / cp - 1) * 100, 1)
                weighted_downside += w * downside / 100
            ev = 0
            tp = 0
            for st in ["bear", "base", "bull"]:
                s = scenarios.get(st)
                if s and s.target_price and s.probability:
                    ev += float(s.target_price) * float(s.probability) / 100
                    tp += float(s.probability)
            if tp > 0:
                expected_ret = round((ev / cp - 1) * 100, 1)
                weighted_expected_return += w * expected_ret / 100

        holdings.append({
            "ticker": r.ticker, "name": r.name, "sector": sec, "country": cty,
            "weight": w, "current_price": cp,
            "upside": upside, "downside": downside, "expected_return": expected_ret,
        })

    cash = round(100 - total_weight, 2)
    top5_weight = sum(h["weight"] for h in sorted(holdings, key=lambda x: -x["weight"])[:5])
    top_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("—", 0)
    top_country = max(country_weights.items(), key=lambda x: x[1]) if country_weights else ("—", 0)

    return {
        "portfolio": {"id": str(p.id), "name": p.name},
        "summary": {
            "holdings_count": len(holdings),
            "total_weight": round(total_weight, 2),
            "cash": cash,
            "weighted_expected_return": round(weighted_expected_return, 2),
            "weighted_downside": round(weighted_downside, 2),
            "top5_concentration": round(top5_weight, 1),
            "top_sector": {"name": top_sector[0], "weight": round(top_sector[1], 1)},
            "top_country": {"name": top_country[0], "weight": round(top_country[1], 1)},
        },
        "holdings": holdings,
        "sector_weights": dict(sorted(sector_weights.items(), key=lambda x: -x[1])),
        "country_weights": dict(sorted(country_weights.items(), key=lambda x: -x[1])),
    }


# ─────────────────────────────────────────────────────────────────
# Analytics (Phase B — real correlation + risk metrics)
# ─────────────────────────────────────────────────────────────────
@router.get("/portfolios/{portfolio_id}/correlation")
async def get_portfolio_correlation(
    portfolio_id: str,
    window: int = 60,
    min_months: int = 12,
    db: AsyncSession = Depends(get_db),
):
    """Monthly log-return correlation + covariance matrix, in USD,
    over the trailing `window` months. Tickers with fewer than
    `min_months` of overlapping history are dropped from the matrix
    and reported under `dropped`.
    """
    from services.portfolio_analytics import compute_correlation_matrix
    if window < 12 or window > 120:
        raise HTTPException(400, "window must be between 12 and 120 months")
    if min_months < 6 or min_months > window:
        raise HTTPException(400, "min_months must be between 6 and window")
    return await compute_correlation_matrix(
        db, portfolio_id, window_months=window, min_months=min_months
    )


@router.get("/companies/{ticker:path}/risk-metrics")
async def get_company_risk_metrics(
    ticker: str,
    window: int = 36,
    min_months: int = 12,
    db: AsyncSession = Depends(get_db),
):
    """Annualised realised volatility from monthly USD log returns."""
    from services.portfolio_analytics import compute_realised_vol
    if window < 12 or window > 120:
        raise HTTPException(400, "window must be between 12 and 120 months")
    company = await _get_company(db, ticker)
    return await compute_realised_vol(
        db, company.id, company.ticker,
        window_months=window, min_months=min_months,
    )


@router.post("/admin/rename-slash-tickers")
async def rename_slash_tickers(db: AsyncSession = Depends(get_db)):
    """One-shot migration: rename BP/ LN → BP LN and BT/A LN → BT-A LN
    in the companies table so every FastAPI route (which uses {ticker}
    without :path) routes cleanly. All FK-linked tables — price_records,
    scenarios, holdings, documents, etc. — reference company_id, so
    no other table needs to be touched.

    Yahoo mapping in services/price_feed.py keeps keys for BOTH old and
    new forms, so daily prices continue to work through the transition.
    """
    renames = [("BP/ LN", "BP LN"), ("BT/A LN", "BT-A LN")]
    results = []
    for old, new in renames:
        rs = await db.execute(
            text("SELECT id, ticker FROM companies WHERE ticker = :t"),
            {"t": old},
        )
        row = rs.fetchone()
        if not row:
            # Check if already renamed
            rs2 = await db.execute(
                text("SELECT id FROM companies WHERE ticker = :t"),
                {"t": new},
            )
            results.append({
                "from": old, "to": new,
                "status": "already_renamed" if rs2.scalar_one_or_none() else "not_found",
            })
            continue
        # Make sure the new ticker isn't already taken by another company
        rs2 = await db.execute(
            text("SELECT id FROM companies WHERE ticker = :t AND id != :id"),
            {"t": new, "id": str(row.id)},
        )
        if rs2.scalar_one_or_none():
            results.append({"from": old, "to": new, "status": "conflict_new_ticker_exists"})
            continue
        await db.execute(
            text("UPDATE companies SET ticker = :new WHERE id = :id"),
            {"new": new, "id": str(row.id)},
        )
        results.append({"from": old, "to": new, "status": "renamed", "company_id": str(row.id)})
    await db.commit()
    return {"renames": results}


@router.post("/admin/seed-factor-proxies")
async def seed_factor_proxies(db: AsyncSession = Depends(get_db)):
    """One-shot: insert the factor-proxy ETFs as companies with
    coverage_status='factor' so the existing backfill + daily price feed
    can keep them up to date."""
    from services.factor_analytics import FACTOR_PROXIES
    inserted, existed = [], []
    for key, meta in FACTOR_PROXIES.items():
        rs = await db.execute(text("SELECT id FROM companies WHERE ticker = :t"),
                              {"t": meta["ticker"]})
        if rs.scalar_one_or_none() is not None:
            existed.append(meta["ticker"])
            continue
        await db.execute(text("""
            INSERT INTO companies (id, ticker, name, sector, industry, country,
                                   coverage_status, primary_analyst)
            VALUES (:id, :tkr, :nm, 'Factor Proxy', 'ETF', 'US',
                    'factor', 'system')
        """), {"id": str(uuid.uuid4()), "tkr": meta["ticker"], "nm": meta["label"]})
        inserted.append(meta["ticker"])
    await db.commit()
    return {"inserted": inserted, "already_existed": existed}


@router.get("/factors/list")
async def list_factors():
    """UI helper: catalogue of supported factor shocks."""
    from services.factor_analytics import list_factor_proxies
    return {"factors": list_factor_proxies()}


@router.post("/admin/refresh-factor-betas")
async def refresh_factor_betas(
    window: int = 36,
    db: AsyncSession = Depends(get_db),
):
    """Recompute holding × factor betas for every active holding using
    the trailing N-month window. Writes to holding_factor_exposures."""
    from services.factor_analytics import refresh_all_holding_betas
    return await refresh_all_holding_betas(db, window_months=window)


@router.get("/portfolios/{portfolio_id}/factor-exposures")
async def get_portfolio_factor_exposures(
    portfolio_id: str,
    window: int = 36,
    db: AsyncSession = Depends(get_db),
):
    """Per-holding factor betas for a portfolio. Returns the weight-
    weighted portfolio-level beta to each factor as a summary."""
    from services.factor_analytics import FACTOR_KEYS
    rs = await db.execute(text("""
        SELECT c.ticker, c.sector, h.weight, hfe.betas, hfe.r_squared,
               hfe.n_months, hfe.as_of_month
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
          LEFT JOIN holding_factor_exposures hfe
            ON hfe.company_id = c.id AND hfe.window_months = :win
         WHERE h.portfolio_id = :pid AND COALESCE(h.weight,0) > 0
         ORDER BY h.weight DESC NULLS LAST
    """), {"pid": portfolio_id, "win": window})
    rows = rs.all()
    per_holding = []
    portfolio_beta = {k: 0.0 for k in FACTOR_KEYS}
    coverage_weight = 0.0
    for r in rows:
        if not r.betas:
            per_holding.append({
                "ticker": r.ticker, "sector": r.sector, "weight_pct": float(r.weight or 0),
                "betas": None, "r_squared": None, "n_months": None,
            })
            continue
        b = r.betas if isinstance(r.betas, dict) else __import__("json").loads(r.betas or "{}")
        per_holding.append({
            "ticker": r.ticker, "sector": r.sector,
            "weight_pct": float(r.weight or 0),
            "betas": {k: round(b.get(k, 0.0), 3) for k in FACTOR_KEYS},
            "r_squared": float(r.r_squared) if r.r_squared else None,
            "n_months": int(r.n_months or 0),
            "as_of_month": r.as_of_month,
        })
        w = float(r.weight or 0) / 100.0
        coverage_weight += w
        for k in FACTOR_KEYS:
            portfolio_beta[k] += w * b.get(k, 0.0)
    return {
        "portfolio_id": portfolio_id,
        "window_months": window,
        "coverage_weight_pct": round(coverage_weight * 100, 2),
        "portfolio_beta": {k: round(v, 3) for k, v in portfolio_beta.items()},
        "per_holding": per_holding,
    }


@router.post("/portfolios/{portfolio_id}/factor-shock")
async def post_factor_shock(
    portfolio_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Run a factor shock against the portfolio.
    Body: {"shocks": {"equity": -0.20, "rates": 0.05, ...}, "window": 36}"""
    from services.factor_analytics import apply_factor_shock
    shocks = body.get("shocks") or {}
    if not shocks or not isinstance(shocks, dict):
        raise HTTPException(400, "body.shocks must be a non-empty dict")
    return await apply_factor_shock(
        db, portfolio_id,
        {k: float(v) for k, v in shocks.items()},
        window_months=int(body.get("window", 36)),
    )


@router.get("/portfolios/{portfolio_id}/stress-test/presets")
async def list_stress_presets(portfolio_id: str):
    """Return the catalogue of historical stress windows available to
    replay against a portfolio. `portfolio_id` is not used today but is
    kept in the path so future user-specific custom scenarios can slot
    in without a URL change."""
    from services.stress_scenarios import list_presets
    return {"presets": list_presets()}


@router.post("/portfolios/{portfolio_id}/stress-test")
async def run_stress_test(
    portfolio_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Run one or more preset historical stress windows against the
    portfolio's current weights. Body: {"scenarios": ["gfc_crash", "covid_crash"]}.
    Returns one result per scenario under `results`.
    """
    from services.stress_scenarios import compute_historical_stress
    keys = body.get("scenarios") or []
    if not keys:
        raise HTTPException(400, "body.scenarios must be a non-empty list of preset keys")
    out = []
    for k in keys:
        out.append(await compute_historical_stress(db, portfolio_id, k))
    return {"portfolio_id": portfolio_id, "results": out}


@router.post("/portfolios/{portfolio_id}/optimise")
async def optimise_portfolio(
    portfolio_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Server-side QP optimiser (scipy SLSQP / linprog).
    Body: {
        "method": "mv" | "kelly" | "cvar",
        "max_position": 0.10,
        "sum_to_one": true,
        "sector_caps": {"Energy": 0.30, ...},
        "country_caps": {"US": 0.60, ...},
        "lambda": 5.0,                 # for mv
        "kelly_fraction": 0.5,         # for kelly
        "alpha": 0.95,                 # for cvar
        "window_months": 60,           # μ/Σ + cvar history window
        "use_realised": true           # μ/Σ from price history (else from scenarios)
    }
    """
    from services.portfolio_optimiser import (
        Constraints, solve_mv, solve_kelly, solve_cvar,
    )
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns, _aligned_returns,
    )

    method = (body.get("method") or "mv").lower()
    constraints = Constraints(
        max_position=float(body.get("max_position", 0.10)),
        sum_to_one=bool(body.get("sum_to_one", True)),
        sector_caps={k: float(v) for k, v in (body.get("sector_caps") or {}).items()},
        country_caps={k: float(v) for k, v in (body.get("country_caps") or {}).items()},
    )
    window = int(body.get("window_months", 60))
    use_realised = bool(body.get("use_realised", True))

    rs = await db.execute(text("""
        SELECT c.id AS cid, c.ticker, c.sector, c.country
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid AND COALESCE(h.weight,0) > 0
         ORDER BY h.weight DESC NULLS LAST
    """), {"pid": portfolio_id})
    holdings = [(r.cid, r.ticker, r.sector, r.country) for r in rs]
    if not holdings:
        raise HTTPException(400, "no holdings with weight > 0")

    fx = await _load_fx_by_month(db)

    # Build per-ticker monthly USD return series
    returns_by_ticker = {}
    for cid, ticker, _sec, _cty in holdings:
        series = await _load_monthly_series_usd(db, cid, window, fx)
        if len(series) >= 13:
            rets = _log_returns(series)
            if len(rets) >= 12:
                returns_by_ticker[ticker] = rets

    if len(returns_by_ticker) < 2:
        raise HTTPException(400, "insufficient price history for optimisation")

    kept, months, matrix = _aligned_returns(returns_by_ticker, min_months=12)
    if len(kept) < 2:
        raise HTTPException(400, "no overlapping price history across holdings")

    # Map kept tickers back to (sector, country) preserving order
    info = {t: (s, c) for (_cid, t, s, c) in holdings}
    sectors = [info[t][0] or "Unknown" for t in kept]
    countries = [info[t][1] or "Unknown" for t in kept]

    # μ from realised: mean monthly log-return × 12
    arr = np.asarray(matrix, dtype=float)             # shape (n_tickers, n_months)
    mus = (arr.mean(axis=1) * 12.0).tolist()
    cov = (np.cov(arr) * 12.0).tolist()

    if method == "mv":
        result = solve_mv(kept, mus, cov, sectors, countries, constraints,
                          lam=float(body.get("lambda", 5.0)))
    elif method == "kelly":
        result = solve_kelly(kept, mus, cov, sectors, countries, constraints,
                             kelly_fraction=float(body.get("kelly_fraction", 0.5)))
    elif method == "cvar":
        # CVaR uses the raw monthly returns matrix (T × N), not the cov.
        # arr is (N × T) so transpose. We use log returns as a proxy for
        # simple returns — for small monthly moves the difference is <0.5%.
        # force_full_investment defaults TRUE so the LP doesn't trivially
        # pick all-cash (always zero loss); user can pass false to allow it.
        result = solve_cvar(
            kept, arr.T.tolist(), sectors, countries, constraints,
            alpha=float(body.get("alpha", 0.95)),
            force_full_investment=bool(body.get("force_full_investment", True)),
        )
    else:
        raise HTTPException(400, f"unknown method '{method}'")

    return {
        "portfolio_id": portfolio_id,
        "method": method,
        "tickers_used": kept,
        "n_months": len(months),
        "constraints": {
            "max_position": constraints.max_position,
            "sum_to_one": constraints.sum_to_one,
            "sector_caps": constraints.sector_caps,
            "country_caps": constraints.country_caps,
        },
        **result,
    }


@router.get("/portfolios/{portfolio_id}/efficient-frontier")
async def get_portfolio_efficient_frontier(
    portfolio_id: str,
    points: int = 25,
    max_position: float = 0.10,
    sector_cap: float | None = None,
    country_cap: float | None = None,
    window: int = 60,
    db: AsyncSession = Depends(get_db),
):
    """Sweep λ on a log scale and return the upper Pareto frontier:
    [{lambda, vol, exp_ret, sharpe, weights, cash_pct}, ...].
    Sector/country caps optional — if set, applied uniformly across all
    distinct sectors/countries in the portfolio."""
    from services.portfolio_optimiser import Constraints, efficient_frontier
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns, _aligned_returns,
    )

    rs = await db.execute(text("""
        SELECT c.id AS cid, c.ticker, c.sector, c.country
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid AND COALESCE(h.weight,0) > 0
    """), {"pid": portfolio_id})
    holdings = [(r.cid, r.ticker, r.sector, r.country) for r in rs]
    if not holdings:
        raise HTTPException(400, "no holdings with weight > 0")

    fx = await _load_fx_by_month(db)
    returns_by_ticker = {}
    for cid, ticker, _sec, _cty in holdings:
        series = await _load_monthly_series_usd(db, cid, window, fx)
        if len(series) >= 13:
            rets = _log_returns(series)
            if len(rets) >= 12:
                returns_by_ticker[ticker] = rets

    if len(returns_by_ticker) < 2:
        raise HTTPException(400, "insufficient price history")
    kept, months, matrix = _aligned_returns(returns_by_ticker, min_months=12)
    info = {t: (s, c) for (_cid, t, s, c) in holdings}
    sectors = [info[t][0] or "Unknown" for t in kept]
    countries = [info[t][1] or "Unknown" for t in kept]

    arr = np.asarray(matrix, dtype=float)
    mus = (arr.mean(axis=1) * 12.0).tolist()
    cov = (np.cov(arr) * 12.0).tolist()

    sector_caps = {}
    if sector_cap is not None:
        for s in set(sectors):
            sector_caps[s] = sector_cap
    country_caps = {}
    if country_cap is not None:
        for c in set(countries):
            country_caps[c] = country_cap

    constraints = Constraints(
        max_position=max_position,
        sector_caps=sector_caps,
        country_caps=country_caps,
    )
    pts = efficient_frontier(kept, mus, cov, sectors, countries, constraints,
                             n_points=points)
    return {
        "portfolio_id": portfolio_id,
        "n_months": len(months),
        "n_tickers": len(kept),
        "constraints": {
            "max_position": max_position,
            "sector_cap": sector_cap,
            "country_cap": country_cap,
        },
        "points": pts,
    }


@router.get("/portfolios/{portfolio_id}/risk-dashboard")
async def get_portfolio_risk_dashboard(
    portfolio_id: str,
    window: int = 60,
    min_months: int = 12,
    confidence: float = 0.95,
    db: AsyncSession = Depends(get_db),
):
    """Phase E — portfolio-level tail risk from USD monthly log returns:
    realised vol, historical VaR + CVaR, max drawdown, best/worst month.
    """
    from services.portfolio_analytics import compute_portfolio_risk_dashboard
    if window < 12 or window > 120:
        raise HTTPException(400, "window must be between 12 and 120 months")
    if not (0.80 <= confidence <= 0.999):
        raise HTTPException(400, "confidence must be between 0.80 and 0.999")
    return await compute_portfolio_risk_dashboard(
        db, portfolio_id,
        window_months=window, min_months=min_months, confidence=confidence,
    )


@router.get("/portfolios/{portfolio_id}/risk-metrics")
async def get_portfolio_risk_metrics(
    portfolio_id: str,
    window: int = 36,
    min_months: int = 12,
    db: AsyncSession = Depends(get_db),
):
    """Batch realised vol for every holding in the portfolio — avoids N
    round-trips from the UI. Keyed by Bloomberg ticker."""
    from services.portfolio_analytics import compute_realised_vol
    if window < 12 or window > 120:
        raise HTTPException(400, "window must be between 12 and 120 months")
    rs = await db.execute(
        select(Company.id, Company.ticker)
        .join(PortfolioHolding, PortfolioHolding.company_id == Company.id)
        .where(PortfolioHolding.portfolio_id == portfolio_id)
    )
    holdings = [(row[0], row[1]) for row in rs]
    metrics = {}
    for cid, ticker in holdings:
        metrics[ticker] = await compute_realised_vol(
            db, cid, ticker, window_months=window, min_months=min_months,
        )
    return {
        "portfolio_id": portfolio_id,
        "window_months": window,
        "min_months": min_months,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────
# Sprint C-prep — native Claude PDF A/B arm
#
# Runs the Tier 5.6 native-PDF extraction server-side so the
# ANTHROPIC_API_KEY never leaves Railway. The local harness
# (scripts/pdf_ab_test.py, --phase native --mode remote) POSTs
# base64-encoded PDFs here; this endpoint calls Anthropic with a
# document content block and returns structured JSON. Temporary —
# delete once the A/B is finalised.
# ─────────────────────────────────────────────────────────────────

class AbPdfInput(BaseModel):
    key:         str
    filename:    str
    pdf_base64:  str

class AbPdfBatch(BaseModel):
    pdfs: list[AbPdfInput]
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 2048


_AB_NATIVE_PROMPT = """\
You are benchmarking PDF comprehension for an investment-research parser
A/B. Treat this PDF as an arbitrary financial document.

Return STRICT JSON, no preamble:
{
  "doc_pages": <int>,
  "heading_count": <int>,
  "main_headings": ["...", "..."],
  "table_count": <int>,
  "key_table_summary": "<1-2 sentences on the biggest table(s)>",
  "visible_charts": <int>,
  "chart_descriptions": ["...", "..."],
  "layout_signals": ["...", "..."],
  "likely_document_type": "<one of: 10-Q, 10-K, condensed_financials, earnings_release, transcript, presentation, tanshin, RNS, other>",
  "sample_first_1000_chars": "<verbatim from page 1>"
}
"""


@router.post("/admin/pdf-ab/native")
async def run_native_pdf_ab(payload: AbPdfBatch):
    """Server-side native Claude PDF extraction for Sprint C-prep A/B.

    Receives base64 PDFs, calls Anthropic document-block API, returns
    parsed JSON per file. Temporary admin endpoint — session auth via
    the global app middleware is sufficient (no need for a separate
    per-route password check).
    """
    import base64 as _b64
    import json as _json
    import time as _time
    from configs.settings import settings as _settings

    if not _settings.anthropic_api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set on server")

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=_settings.anthropic_api_key)

    results = []
    total_in = 0
    total_out = 0
    for item in payload.pdfs:
        t0 = _time.time()
        try:
            pdf_bytes = _b64.standard_b64decode(item.pdf_base64)
        except Exception as exc:
            results.append({"key": item.key, "status": "error",
                            "error": f"bad base64: {exc}"})
            continue

        # Re-encode because Anthropic wants base64 string; we already have it
        pdf_b64 = item.pdf_base64

        try:
            resp = await client.messages.create(
                model=payload.model,
                max_tokens=payload.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": _AB_NATIVE_PROMPT},
                    ],
                }],
            )
        except Exception as exc:
            results.append({
                "key": item.key, "status": "error",
                "error": str(exc)[:300],
                "elapsed_s": round(_time.time() - t0, 2),
                "pdf_bytes": len(pdf_bytes),
            })
            continue

        text_out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parsed = {}
        try:
            stripped = text_out.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = _json.loads(stripped)
        except Exception as exc:
            parsed = {"_parse_error": str(exc)[:200]}

        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
        total_in += input_tokens
        total_out += output_tokens

        results.append({
            "key":           item.key,
            "filename":      item.filename,
            "status":        "ok",
            "elapsed_s":     round(_time.time() - t0, 2),
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost, 4),
            "pdf_bytes":     len(pdf_bytes),
            "parsed":        parsed,
            "raw_response":  text_out,
            "raw_response_len": len(text_out),
        })

    return {
        "count":        len(results),
        "total_input_tokens":  total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round((total_in * 3.0 + total_out * 15.0) / 1_000_000, 4),
        "results":      results,
    }
