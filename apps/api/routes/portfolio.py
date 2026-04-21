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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func, delete
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
