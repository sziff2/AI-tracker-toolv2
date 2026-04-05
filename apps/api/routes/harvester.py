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

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel

from apps.api.rate_limit import limiter
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, HarvesterSource, HarvestedDocument
from configs.settings import settings
from services.harvester.sources.sec_edgar import EDGAR_SOURCES
from services.harvester.sources.investegate import INVESTEGATE_SOURCES

logger = logging.getLogger(__name__)
router = APIRouter(tags=["harvester"])


# ── Schemas ──────────────────────────────────────────────────────

class SourceUpdate(BaseModel):
    ir_docs_url: Optional[str] = None   # IR documents page for scraper
    ir_url:      Optional[str] = None   # IR homepage (informational)
    override:    Optional[bool] = None
    notes:       Optional[str] = None


# ── GET /harvester/status ─────────────────────────────────────────

@router.get("/harvester/status")
async def harvester_status(db: AsyncSession = Depends(get_db)):
    """Per-company harvester status: ok/stale/failed/never_run/no_source/new_docs."""
    from sqlalchemy import func, text
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(days=7)

    # All active companies
    companies_q = await db.execute(
        select(Company.id, Company.ticker, Company.name, Company.cik)
        .where(Company.coverage_status == "active")
        .order_by(Company.ticker)
    )
    companies = companies_q.all()

    # Harvester sources keyed by company_id
    sources_q = await db.execute(select(HarvesterSource))
    sources = {s.company_id: s for s in sources_q.scalars().all()}

    # Harvested documents stats per company: latest doc date, pending count, last error
    hd_stats_q = await db.execute(text("""
        SELECT
            company_id,
            MAX(discovered_at) AS last_harvested,
            MAX(CASE WHEN ingested THEN discovered_at END) AS latest_ingested_at,
            COUNT(*) FILTER (WHERE NOT ingested AND error IS NULL) AS pending_count,
            MAX(CASE WHEN error IS NOT NULL THEN discovered_at END) AS last_error_at,
            MAX(error) AS last_error
        FROM harvested_documents
        GROUP BY company_id
    """))
    hd_stats = {}
    for row in hd_stats_q.all():
        hd_stats[row.company_id] = row

    # Latest document date per company (from documents table)
    doc_stats_q = await db.execute(text("""
        SELECT company_id, MAX(published_at) AS latest_doc_date
        FROM documents
        GROUP BY company_id
    """))
    doc_dates = {row.company_id: row.latest_doc_date for row in doc_stats_q.all()}

    result = []
    for co_id, co_ticker, co_name, co_cik in companies:
        src = sources.get(co_id)
        has_ir_url = bool(src and src.ir_docs_url)
        has_edgar = co_ticker in EDGAR_SOURCES
        has_investegate = co_ticker in INVESTEGATE_SOURCES
        has_source = has_ir_url or has_edgar or has_investegate

        stats = hd_stats.get(co_id)
        last_harvested = stats.last_harvested if stats else None
        pending_count = stats.pending_count if stats else 0
        last_error = stats.last_error if stats else None
        last_error_at = stats.last_error_at if stats else None

        latest_doc_date = doc_dates.get(co_id)

        # Status logic
        if not has_source:
            status = "no_source"
        elif last_harvested is None:
            status = "never_run"
        elif last_error_at and (not last_harvested or last_error_at >= last_harvested):
            # Most recent harvest attempt had an error
            status = "failed"
        elif last_harvested < stale_threshold:
            status = "stale"
        elif pending_count > 0:
            status = "new_docs"
        else:
            status = "ok"

        result.append({
            "ticker": co_ticker,
            "name": co_name,
            "last_harvested": last_harvested.isoformat() if last_harvested else None,
            "latest_doc_date": latest_doc_date.isoformat() if latest_doc_date else None,
            "new_docs": pending_count,
            "status": status,
            "has_ir_url": has_ir_url,
            "has_edgar": has_edgar,
            "last_error": last_error,
        })

    return result


# ── GET /harvester/sources ────────────────────────────────────────

@router.get("/harvester/sources")
async def list_harvester_sources(db: AsyncSession = Depends(get_db)):
    # Use column-level select to avoid loading all relationships
    from sqlalchemy import Column
    companies_q = await db.execute(
        select(Company.id, Company.ticker, Company.name, Company.country)
        .where(Company.coverage_status == "active")
        .order_by(Company.ticker)
    )
    companies = companies_q.all()

    sources_q = await db.execute(select(HarvesterSource))
    sources = {s.company_id: s for s in sources_q.scalars().all()}

    result = []
    for co_id, co_ticker, co_name, co_country in companies:
        src = sources.get(co_id)
        has_edgar = co_ticker in EDGAR_SOURCES
        has_investegate = co_ticker in INVESTEGATE_SOURCES
        if has_edgar and has_investegate:
            active_source = "edgar+investegate"
        elif has_edgar:
            active_source = "edgar"
        elif has_investegate:
            active_source = "investegate"
        elif src and src.ir_docs_url:
            active_source = "ir_scrape"
        else:
            active_source = "none"

        result.append({
            "ticker":           co_ticker,
            "name":             co_name,
            "country":          co_country,
            "active_source":    active_source,
            "ir_docs_url":      src.ir_docs_url if src else None,
            "ir_url":           src.ir_url if src else None,
            "override":         src.override if src else False,
            "last_checked_at":  src.last_checked_at.isoformat() if src and src.last_checked_at else None,
            "notes":            src.notes if src else None,
            "status":           _status(co_ticker, src),
        })

    return result


def _status(ticker: str, src) -> str:
    has_edgar = ticker in EDGAR_SOURCES
    has_investegate = ticker in INVESTEGATE_SOURCES
    if has_edgar and has_investegate:
        return "edgar+investegate"
    if has_edgar:
        return "edgar"
    if has_investegate:
        return "investegate"
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
    skip_llm: bool = False,
):
    tickers = [ticker.upper()] if ticker else None
    background_tasks.add_task(_run_harvest_bg, tickers, skip_llm)
    return {
        "status": "harvest_started",
        "scope": tickers or "all active companies",
        "skip_llm": skip_llm,
    }


async def _run_harvest_bg(tickers, skip_llm=False):
    from services.harvester import run_harvest
    result = await run_harvest(tickers=tickers, skip_llm=skip_llm)
    logger.info("[HARVEST] Manual run complete: new=%d skipped=%d failed=%d",
                result["new"], result["skipped"], result["failed"])


# ── POST /harvester/run-weekly — trigger weekly harvest + report ─────

@router.post("/harvester/run-weekly")
async def trigger_weekly_harvest(background_tasks: BackgroundTasks):
    """Manually trigger a weekly-style harvest with report and Teams notification."""
    background_tasks.add_task(_run_weekly_bg)
    return {"status": "weekly_harvest_started"}


async def _run_weekly_bg():
    from services.harvester.scheduler import run_and_report
    try:
        result = await run_and_report(trigger="manual")
        logger.info("[HARVEST] Weekly run complete: new=%d skipped=%d failed=%d report=%s",
                    result["new"], result["skipped"], result["failed"], result.get("report_id"))
    except Exception as exc:
        logger.error("[HARVEST] Weekly run FAILED: %s", exc, exc_info=True)


@router.post("/harvester/test-teams")
async def test_teams_webhook():
    """Diagnostic: test Teams webhook with a dummy payload."""
    from services.harvester.scheduler import post_teams_report
    dummy = {"new": 1, "skipped": 2, "failed": 0, "details": [
        {"ticker": "TEST", "name": "Test Company", "sources_tried": ["edgar"], "candidates_found": 1, "errors": [], "source_used": "edgar"}
    ]}
    try:
        sent = await post_teams_report(dummy)
        return {"sent": sent, "webhook_configured": bool(settings.teams_webhook_url)}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


@router.post("/harvester/test-alert")
async def test_alert():
    """Test the Teams alert system (budget warnings, pipeline errors)."""
    from services.alerts import send_alert
    try:
        sent = await send_alert(
            "This is a test alert from AI Tracker. "
            "Budget warnings and pipeline errors will appear here.",
            level="info",
        )
        return {"sent": sent, "webhook_configured": bool(settings.teams_webhook_url)}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


# ── GET /harvester/reports — recent harvest reports ──────────────────

@router.get("/harvester/reports")
async def list_harvest_reports(limit: int = 10, db: AsyncSession = Depends(get_db)):
    """List recent harvest reports, most recent first."""
    import json as _json
    from sqlalchemy import text
    result = await db.execute(
        text("SELECT id, run_at, trigger, summary_json, teams_sent, created_at "
             "FROM harvest_reports ORDER BY run_at DESC LIMIT :limit"),
        {"limit": limit},
    )
    rows = result.all()
    return [
        {
            "id": str(r.id),
            "run_at": r.run_at.isoformat() if r.run_at else None,
            "trigger": r.trigger,
            "summary": _json.loads(r.summary_json) if r.summary_json else None,
            "teams_sent": r.teams_sent,
        }
        for r in rows
    ]


@router.get("/harvester/reports/latest")
async def get_latest_report(db: AsyncSession = Depends(get_db)):
    """Get the latest harvest report with full per-company detail."""
    import json as _json
    from sqlalchemy import text
    result = await db.execute(
        text("SELECT id, run_at, trigger, summary_json, details_json, teams_sent "
             "FROM harvest_reports ORDER BY run_at DESC LIMIT 1")
    )
    row = result.first()
    if not row:
        raise HTTPException(404, "No harvest reports found")
    return {
        "id": str(row.id),
        "run_at": row.run_at.isoformat() if row.run_at else None,
        "trigger": row.trigger,
        "summary": _json.loads(row.summary_json) if row.summary_json else None,
        "details": _json.loads(row.details_json) if row.details_json else None,
        "teams_sent": row.teams_sent,
    }


# ── GET /harvester/test-fetch — diagnostic for page fetching ───────

@router.get("/harvester/test-fetch")
async def test_fetch(url: str):
    """Diagnostic: test fetch_page chain on a URL."""
    from services.doc_utils import fetch_page
    import time
    start = time.time()
    html = fetch_page(url, timeout=90)
    elapsed = round(time.time() - start, 1)
    import re
    pdfs = len(re.findall(r'\.pdf', html, re.IGNORECASE))
    visible = re.sub(r'<[^>]+>', ' ', html)
    visible = re.sub(r'\s+', ' ', visible).strip()
    return {
        "url": url,
        "size": len(html),
        "pdfs": pdfs,
        "visible_text_len": len(visible),
        "elapsed_seconds": elapsed,
        "first_100": html[:100] if html else "EMPTY",
    }


# ── POST /harvester/llm-scan — LLM-powered IR page scan ───────────

@router.post("/harvester/llm-scan/{ticker:path}")
@limiter.limit("10/minute")
async def llm_scan_ir(request: Request, ticker: str, db: AsyncSession = Depends(get_db)):
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
    (re.compile(r'FY[\s\-_\.]*(\d{4})', re.IGNORECASE), lambda m: f'{m.group(1)}_Q4'),
    # 2025-annual / 2024_annual / 2024-full-year
    (re.compile(r'(\d{4})[\s\-_\.]*(annual|full[\s\-_]*year)', re.IGNORECASE), lambda m: f'{m.group(1)}_Q4'),
    # annual-2025 / annual_report_2024
    (re.compile(r'(annual|full[\s\-_]*year)[\s\-_\.]*(\d{4})', re.IGNORECASE), lambda m: f'{m.group(2)}_Q4'),
    # H1-2025 / 1H2025 → Q2, H2_2024 → Q4
    (re.compile(r'[Hh]1[\s\-_\.]*(\d{4})'), lambda m: f'{m.group(1)}_Q2'),
    (re.compile(r'[Hh]2[\s\-_\.]*(\d{4})'), lambda m: f'{m.group(1)}_Q4'),
    (re.compile(r'1[Hh][\s\-_\.]*(\d{4})'), lambda m: f'{m.group(1)}_Q2'),
    (re.compile(r'2[Hh][\s\-_\.]*(\d{4})'), lambda m: f'{m.group(1)}_Q4'),
    (re.compile(r'(\d{4})[\s\-_\.]*[Hh]1'), lambda m: f'{m.group(1)}_Q2'),
    (re.compile(r'(\d{4})[\s\-_\.]*[Hh]2'), lambda m: f'{m.group(1)}_Q4'),
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
