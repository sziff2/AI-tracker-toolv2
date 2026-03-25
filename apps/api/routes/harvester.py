"""
Harvester Sources API

Endpoints:
  GET  /harvester/sources            — list all companies + source status
  PUT  /harvester/sources/{ticker}   — set ir_docs_url, override flag, notes
  POST /harvester/sources/{ticker}/discover — re-run LLM discovery
  POST /harvester/run                — manually trigger harvest run
  GET  /harvester/log                — recent harvest log
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, HarvesterSource, HarvestedDocument
from services.harvester.sources.sec_edgar import EDGAR_SOURCES

logger = logging.getLogger(__name__)
router = APIRouter(tags=["harvester"])


# ── Schemas ──────────────────────────────────────────────────────

class SourceUpdate(BaseModel):
    ir_docs_url: Optional[str] = None   # IR documents page for scraper
    ir_url:      Optional[str] = None   # IR homepage (informational)
    override:    Optional[bool] = None
    notes:       Optional[str] = None


# ── GET /harvester/sources ────────────────────────────────────────

@router.get("/harvester/sources")
async def list_harvester_sources(db: AsyncSession = Depends(get_db)):
    companies_q = await db.execute(
        select(Company).where(Company.coverage_status == "active").order_by(Company.ticker)
    )
    companies = companies_q.scalars().all()

    sources_q = await db.execute(select(HarvesterSource))
    sources = {s.company_id: s for s in sources_q.scalars().all()}

    result = []
    for co in companies:
        src = sources.get(co.id)
        # Determine which source will actually be used
        if co.ticker in EDGAR_SOURCES:
            active_source = "edgar"
        elif src and src.ir_docs_url:
            active_source = "ir_scrape"
        else:
            active_source = "none"

        result.append({
            "ticker":           co.ticker,
            "name":             co.name,
            "country":          co.country,
            "active_source":    active_source,        # what will run
            "ir_docs_url":      src.ir_docs_url if src else None,
            "ir_url":           src.ir_url if src else None,
            "override":         src.override if src else False,
            "last_checked_at":  src.last_checked_at.isoformat() if src and src.last_checked_at else None,
            "notes":            src.notes if src else None,
            "status":           _status(co.ticker, src),
        })

    return result


def _status(ticker: str, src) -> str:
    if ticker in EDGAR_SOURCES:
        return "edgar"
    if src and src.ir_docs_url:
        return "scraper" if not src.override else "scraper_locked"
    return "unconfigured"


# ── PUT /harvester/sources/{ticker} ──────────────────────────────

@router.put("/harvester/sources/{ticker}")
async def update_harvester_source(
    ticker: str,
    body: SourceUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update source config for a company.
    Primary use: set ir_docs_url so the scraper knows where to look.
    """
    co_q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = co_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    src_q = await db.execute(
        select(HarvesterSource).where(HarvesterSource.company_id == company.id)
    )
    src = src_q.scalar_one_or_none()

    if src is None:
        src = HarvesterSource(id=uuid.uuid4(), company_id=company.id)
        db.add(src)

    if body.ir_docs_url is not None:
        src.ir_docs_url = body.ir_docs_url.strip() or None
        src.discovery_method = "manual"
        src.ir_reachable = True
        src.last_checked_at = datetime.now(timezone.utc)

    if body.ir_url is not None:
        src.ir_url = body.ir_url.strip() or None

    if body.override is not None:
        src.override = body.override

    if body.notes is not None:
        src.notes = body.notes.strip() or None

    await db.commit()
    return {"status": "updated", "ticker": ticker}


# ── POST /harvester/run ───────────────────────────────────────────

@router.post("/harvester/run")
async def trigger_harvest_run(
    background_tasks: BackgroundTasks,
    ticker: Optional[str] = None,
):
    tickers = [ticker.upper()] if ticker else None
    background_tasks.add_task(_run_harvest_bg, tickers)
    return {
        "status": "harvest_started",
        "scope": tickers or "all active companies",
    }


async def _run_harvest_bg(tickers):
    from services.harvester import run_harvest
    result = await run_harvest(tickers=tickers)
    logger.info("[HARVEST] Manual run complete: %s", result)


# ── GET /harvester/log ────────────────────────────────────────────

@router.get("/harvester/log")
async def get_harvest_log(
    limit: int = 50,
    ticker: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(HarvestedDocument, Company.ticker, Company.name)
        .join(Company, HarvestedDocument.company_id == Company.id)
        .order_by(desc(HarvestedDocument.discovered_at))
        .limit(limit)
    )
    if ticker:
        query = query.where(Company.ticker == ticker.upper())

    result = await db.execute(query)
    return [
        {
            "ticker":       r.ticker,
            "company":      r.name,
            "source":       r.HarvestedDocument.source,
            "headline":     r.HarvestedDocument.headline,
            "period":       r.HarvestedDocument.period_label,
            "source_url":   r.HarvestedDocument.source_url,
            "discovered_at":r.HarvestedDocument.discovered_at.isoformat() if r.HarvestedDocument.discovered_at else None,
            "ingested":     r.HarvestedDocument.ingested,
            "error":        r.HarvestedDocument.error,
        }
        for r in result.all()
    ]
