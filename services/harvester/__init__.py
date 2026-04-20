"""
Document Harvesting Agent

Priority order per company:
  1. SEC EDGAR      — structured filings, most reliable, no scraping
  2. Investegate    — UK RNS announcements (results, trading updates)
  3. IR Scraper     — regex scraper using ir_docs_url (also runs for EDGAR/Investegate companies)
  4. LLM Scraper    — last resort, sends page HTML to Claude (skipped in auto-runs)

The ir_docs_url is set via the Harvester Sources admin panel
in the Data Hub — analysts paste the IR documents page URL
for any company that needs it.
"""

import asyncio
import logging
import time

from sqlalchemy import select

from apps.api.database import AsyncSessionLocal
from apps.api.models import Company, HarvesterSource
from services.harvester.sources.sec_edgar import fetch_sec_edgar, EDGAR_SOURCES
from services.harvester.sources.investegate import fetch_investegate, INVESTEGATE_SOURCES
from services.harvester.sources.ir_scraper import scrape_ir_page
from services.harvester.sources.ir_llm_scraper import scrape_ir_with_llm
from services.harvester.dispatcher import dispatch_candidates

logger = logging.getLogger(__name__)


async def run_harvest(
    tickers: list[str] | None = None,
    skip_llm: bool = False,
) -> dict:
    """
    Run a full harvest cycle.

    Args:
        tickers: Restrict to specific tickers. None = all active companies.
        skip_llm: If True, skip the LLM scraper (Priority 4) to contain costs.

    Returns:
        {
            "new": int, "skipped": int, "failed": int, "tickers": list,
            "details": [{"ticker", "name", "sources_tried", "candidates_found", "errors"}, ...]
        }
    """
    async with AsyncSessionLocal() as db:
        query = select(Company).where(Company.coverage_status == "active")
        if tickers:
            query = query.where(Company.ticker.in_(tickers))
        result = await db.execute(query)
        companies = result.scalars().all()

        src_result = await db.execute(select(HarvesterSource))
        sources = {s.company_id: s for s in src_result.scalars().all()}

    logger.info("[HARVEST] Starting harvest for %d companies (skip_llm=%s)", len(companies), skip_llm)

    from configs.settings import settings
    COMPANY_TIMEOUT = settings.harvester_company_timeout_seconds  # env-overridable

    all_candidates = []
    all_details = []

    async def _harvest_one(company, src, skip_llm):
        """Harvest a single company. Returns (candidates, detail)."""
        candidates = []
        used_source = None
        detail = {
            "ticker": company.ticker,
            "name": company.name,
            "sources_tried": [],
            "candidates_found": 0,
            "errors": [],
        }

        # ── Priority 1: SEC EDGAR ─────────────────────────────────
        if company.ticker in EDGAR_SOURCES:
            detail["sources_tried"].append("edgar")
            try:
                found = await fetch_sec_edgar(company.ticker)
                if found:
                    candidates.extend(found)
                    detail["candidates_found"] += len(found)
                    used_source = "edgar"
                    logger.info("[HARVEST] %s → EDGAR (%d)", company.ticker, len(found))
            except Exception as exc:
                detail["errors"].append(f"EDGAR: {exc}")
                logger.error("[HARVEST] EDGAR error for %s: %s", company.ticker, exc)

        # ── Priority 2: Investegate (UK RNS) ─────────────────────
        if company.ticker in INVESTEGATE_SOURCES:
            detail["sources_tried"].append("investegate")
            try:
                found = await fetch_investegate(company.ticker)
                if found:
                    candidates.extend(found)
                    detail["candidates_found"] += len(found)
                    if used_source is None:
                        used_source = "investegate"
                    logger.info("[HARVEST] %s → Investegate (%d)", company.ticker, len(found))
            except Exception as exc:
                detail["errors"].append(f"Investegate: {exc}")
                logger.error("[HARVEST] Investegate error for %s: %s", company.ticker, exc)

        # ── Priority 3: IR page scraper (regex) ──────────────────
        ir_docs_url = src.ir_docs_url if src else None
        if ir_docs_url:
            detail["sources_tried"].append("ir_scrape")
            try:
                found = await scrape_ir_page(ticker=company.ticker, ir_docs_url=ir_docs_url)
                if found:
                    candidates.extend(found)
                    detail["candidates_found"] += len(found)
                    if used_source is None:
                        used_source = "ir_scrape"
                    logger.info("[HARVEST] %s → IR scraper (%d)", company.ticker, len(found))
            except Exception as exc:
                detail["errors"].append(f"IR scraper: {exc}")
                logger.error("[HARVEST] IR scraper error for %s: %s", company.ticker, exc)

        # ── Priority 4: LLM-powered scraper (fallback) ──────────
        if not skip_llm and used_source is None and ir_docs_url:
            detail["sources_tried"].append("ir_llm")
            try:
                found = await scrape_ir_with_llm(
                    ticker=company.ticker, company_name=company.name, ir_docs_url=ir_docs_url)
                if found:
                    candidates.extend(found)
                    detail["candidates_found"] += len(found)
                    used_source = "ir_llm"
                    logger.info("[HARVEST] %s → LLM scraper (%d)", company.ticker, len(found))
            except Exception as exc:
                detail["errors"].append(f"LLM scraper: {exc}")
                logger.error("[HARVEST] LLM scraper error for %s: %s", company.ticker, exc)

        if used_source is None:
            logger.debug("[HARVEST] %s — no source configured", company.ticker)

        detail["source_used"] = used_source
        return candidates, detail

    for company in companies:
        src = sources.get(company.id)
        t0 = time.time()
        try:
            candidates, detail = await asyncio.wait_for(
                _harvest_one(company, src, skip_llm),
                timeout=COMPANY_TIMEOUT,
            )
            all_candidates.extend(candidates)
        except asyncio.TimeoutError:
            elapsed = round(time.time() - t0, 1)
            detail = {
                "ticker": company.ticker, "name": company.name,
                "sources_tried": [], "candidates_found": 0,
                "errors": [f"Timed out after {elapsed}s (limit {COMPANY_TIMEOUT}s)"],
                "source_used": None,
            }
            logger.warning("[HARVEST] %s timed out after %ss", company.ticker, elapsed)
        all_details.append(detail)

    # Update last_checked_at for all harvested companies (upsert — create row if missing)
    try:
        from datetime import datetime, timezone as tz
        import uuid as _uuid
        now = datetime.now(tz.utc)
        async with AsyncSessionLocal() as db:
            from sqlalchemy import text
            for company in companies:
                await db.execute(text("""
                    INSERT INTO harvester_sources (id, company_id, last_checked_at)
                    VALUES (:id, :cid, :now)
                    ON CONFLICT (company_id) DO UPDATE SET last_checked_at = :now
                """), {"id": str(_uuid.uuid4()), "cid": str(company.id), "now": now})
            await db.commit()
    except Exception as exc:
        logger.warning("[HARVEST] Failed to update last_checked_at: %s", exc)

    logger.info("[HARVEST] %d total candidates before dedup", len(all_candidates))
    summary = await dispatch_candidates(all_candidates)
    logger.info(
        "[HARVEST] Complete — new: %d, skipped: %d, failed: %d",
        summary["new"], summary["skipped"], summary["failed"],
    )
    summary["details"] = all_details
    return summary
