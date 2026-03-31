"""
Harvester Sources API

Endpoints:
  GET  /harvester/sources            — list all companies + source status
  PUT  /harvester/sources/{ticker:path}   — set ir_docs_url, override flag, notes
  POST /harvester/sources/{ticker:path}/discover — re-run LLM discovery
  POST /harvester/run                — manually trigger harvest run
  GET  /harvester/log                — recent harvest log
"""

import re
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


# ── PUT /harvester/sources/{ticker:path} ──────────────────────────────

@router.put("/harvester/sources/{ticker:path}")
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
        raise HTTPException(404, f"Company {ticker:path} not found")

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


# ── POST /harvester/llm-scan — LLM-powered IR page scan ───────────

@router.post("/harvester/llm-scan/{ticker:path}")
async def llm_scan_ir(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Use the LLM to scan a company's IR page and find all documents.
    Returns the list for human review — does NOT auto-ingest.
    """
    from apps.api.models import Company, HarvesterSource
    comp_q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker:path} not found")

    src_q = await db.execute(select(HarvesterSource).where(HarvesterSource.company_id == company.id))
    src = src_q.scalar_one_or_none()
    ir_url = src.ir_docs_url if src else None
    if not ir_url:
        raise HTTPException(400, "No IR URL configured. Set one on the Documents tab first.")

    from services.harvester.sources.ir_llm_scraper import scrape_ir_with_llm
    from services.harvester.dispatcher import dispatch_candidates
    candidates = await scrape_ir_with_llm(
        ticker=company.ticker,
        company_name=company.name,
        ir_docs_url=ir_url,
    )

    # Save to harvested_documents so they appear in the Documents tab
    if candidates:
        summary = await dispatch_candidates(candidates)
    else:
        summary = {"new": 0, "skipped": 0, "failed": 0}

    return {
        "ticker": company.ticker,
        "ir_url": ir_url,
        "documents_found": len(candidates),
        "new": summary["new"],
        "skipped": summary["skipped"],
    }


# ── Period inference from headline / URL ──────────────────────────

_PERIOD_PATTERNS = [
    # Q3 2025 / q3-2025 / Q4_2024 / Q1.2025
    (re.compile(r'[Qq]([1-4])[\s\-_\.]*(\d{4})'), lambda m: f'{m.group(2)}_Q{m.group(1)}'),
    # 2025-Q3 / 2025_Q1
    (re.compile(r'(\d{4})[\s\-_\.]*[Qq]([1-4])'), lambda m: f'{m.group(1)}_Q{m.group(2)}'),
    # FY2024 / FY-2024 / FY_2024
    (re.compile(r'FY[\s\-_\.]*(\d{4})', re.IGNORECASE), lambda m: f'{m.group(1)}_FY'),
    # 2025-annual / 2024_annual / 2024-full-year
    (re.compile(r'(\d{4})[\s\-_\.]*(annual|full[\s\-_]*year)', re.IGNORECASE), lambda m: f'{m.group(1)}_FY'),
    # annual-2025 / annual_report_2024
    (re.compile(r'(annual|full[\s\-_]*year)[\s\-_\.]*(\d{4})', re.IGNORECASE), lambda m: f'{m.group(2)}_FY'),
    # H1-2025 / 1H2025 / H2_2024
    (re.compile(r'[Hh]([12])[\s\-_\.]*(\d{4})'), lambda m: f'{m.group(2)}_H{m.group(1)}'),
    (re.compile(r'([12])[Hh][\s\-_\.]*(\d{4})'), lambda m: f'{m.group(2)}_H{m.group(1)}'),
    # 2025-H1
    (re.compile(r'(\d{4})[\s\-_\.]*[Hh]([12])'), lambda m: f'{m.group(1)}_H{m.group(2)}'),
]


def _infer_period(headline: str, source_url: str) -> Optional[str]:
    """Try to extract a period label from headline or URL filename."""
    for text in [headline or '', source_url or '']:
        for pattern, fmt in _PERIOD_PATTERNS:
            m = pattern.search(text)
            if m:
                return fmt(m)
    return None


# ── DELETE /harvester/log — clear harvest log for a company ────────

@router.delete("/harvester/log")
async def clear_harvest_log(ticker: str, source: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """Clear harvested document entries for a company. Optionally filter by source."""
    from sqlalchemy import delete as sa_delete
    comp_q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = comp_q.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    q = sa_delete(HarvestedDocument).where(HarvestedDocument.company_id == company.id)
    if source:
        q = q.where(HarvestedDocument.source == source)
    result = await db.execute(q)
    await db.commit()
    return {"status": "cleared", "ticker": ticker, "deleted": result.rowcount}


# ── GET /harvester/log ────────────────────────────────────────────

@router.get("/harvester/log")
async def get_harvest_log(
    limit: int = 50,
    ticker: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    from apps.api.models import Document

    query = (
        select(HarvestedDocument, Company.ticker, Company.name)
        .join(Company, HarvestedDocument.company_id == Company.id)
        .order_by(desc(HarvestedDocument.discovered_at))
        .limit(limit)
    )
    if ticker:
        query = query.where(Company.ticker == ticker.upper())

    result = await db.execute(query)
    rows = result.all()

    # Collect linked document IDs to fetch parsing_status and metrics count
    doc_ids = [r.HarvestedDocument.document_id for r in rows if r.HarvestedDocument.document_id]
    doc_info = {}
    if doc_ids:
        from apps.api.models import DocumentSection, ExtractedMetric
        from sqlalchemy import func
        # Get parsing_status for linked documents
        doc_q = await db.execute(
            select(Document.id, Document.parsing_status, Document.document_type).where(Document.id.in_(doc_ids))
        )
        for d in doc_q.all():
            doc_info[d.id] = {"parsing_status": d.parsing_status, "document_type": d.document_type}
        # Count sections per document
        sec_q = await db.execute(
            select(DocumentSection.document_id, func.count(DocumentSection.id))
            .where(DocumentSection.document_id.in_(doc_ids))
            .group_by(DocumentSection.document_id)
        )
        for s in sec_q.all():
            if s[0] in doc_info:
                doc_info[s[0]]["sections_count"] = s[1]
        # Count extracted metrics per document
        met_q = await db.execute(
            select(ExtractedMetric.document_id, func.count(ExtractedMetric.id))
            .where(ExtractedMetric.document_id.in_(doc_ids))
            .group_by(ExtractedMetric.document_id)
        )
        for m in met_q.all():
            if m[0] in doc_info:
                doc_info[m[0]]["metrics_count"] = m[1]

    out = []
    for r in rows:
        hd = r.HarvestedDocument
        period = hd.period_label
        # Feature 2: auto-infer period from headline/URL if missing
        inferred_period = None
        if not period:
            inferred_period = _infer_period(hd.headline, hd.source_url)
            period = inferred_period

        # Linked document info
        linked = doc_info.get(hd.document_id, {})

        out.append({
            "ticker":          r.ticker,
            "company":         r.name,
            "source":          hd.source,
            "headline":        hd.headline,
            "period":          period,
            "period_inferred": inferred_period is not None,
            "source_url":      hd.source_url,
            "document_type":   linked.get("document_type"),
            "discovered_at":   hd.discovered_at.isoformat() if hd.discovered_at else None,
            "ingested":        hd.ingested,
            "error":           hd.error,
            "parsing_status":  linked.get("parsing_status"),
            "sections_count":  linked.get("sections_count", 0),
            "metrics_count":   linked.get("metrics_count", 0),
        })
    return out
