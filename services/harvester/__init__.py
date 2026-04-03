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

import logging

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

    all_candidates = []
    all_details = []

    for company in companies:
        src = sources.get(company.id)
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
                candidates = await fetch_sec_edgar(company.ticker)
                if candidates:
                    all_candidates.extend(candidates)
                    detail["candidates_found"] += len(candidates)
                    used_source = "edgar"
                    logger.info(
                        "[HARVEST] %s → EDGAR (%d candidates)",
                        company.ticker, len(candidates)
                    )
            except Exception as exc:
                detail["errors"].append(f"EDGAR: {exc}")
                logger.error(
                    "[HARVEST] EDGAR error for %s: %s",
                    company.ticker, exc, exc_info=True
                )

        # ── Priority 2: Investegate (UK RNS) ─────────────────────
        # Runs even if EDGAR found results — catches UK-specific
        # trading updates that don't appear as SEC filings.
        if company.ticker in INVESTEGATE_SOURCES:
            detail["sources_tried"].append("investegate")
            try:
                candidates = await fetch_investegate(company.ticker)
                if candidates:
                    all_candidates.extend(candidates)
                    detail["candidates_found"] += len(candidates)
                    if used_source is None:
                        used_source = "investegate"
                    logger.info(
                        "[HARVEST] %s → Investegate (%d candidates)",
                        company.ticker, len(candidates)
                    )
            except Exception as exc:
                detail["errors"].append(f"Investegate: {exc}")
                logger.error(
                    "[HARVEST] Investegate error for %s: %s",
                    company.ticker, exc, exc_info=True
                )

        # ── Priority 3: IR page scraper (regex) ──────────────────
        # Runs even for EDGAR/Investegate companies — catches presentations,
        # transcripts, and supplements that don't appear in SEC filings.
        ir_docs_url = src.ir_docs_url if src else None
        if ir_docs_url:
            detail["sources_tried"].append("ir_scrape")
            try:
                candidates = await scrape_ir_page(
                    ticker=company.ticker,
                    ir_docs_url=ir_docs_url,
                )
                if candidates:
                    all_candidates.extend(candidates)
                    detail["candidates_found"] += len(candidates)
                    if used_source is None:
                        used_source = "ir_scrape"
                    logger.info(
                        "[HARVEST] %s → IR scraper (%d candidates)",
                        company.ticker, len(candidates)
                    )
            except Exception as exc:
                detail["errors"].append(f"IR scraper: {exc}")
                logger.error(
                    "[HARVEST] IR scraper error for %s: %s",
                    company.ticker, exc, exc_info=True
                )

        # ── Priority 4: LLM-powered scraper (fallback) ──────────
        if not skip_llm and used_source is None and ir_docs_url:
            detail["sources_tried"].append("ir_llm")
            try:
                candidates = await scrape_ir_with_llm(
                    ticker=company.ticker,
                    company_name=company.name,
                    ir_docs_url=ir_docs_url,
                )
                if candidates:
                    all_candidates.extend(candidates)
                    detail["candidates_found"] += len(candidates)
                    used_source = "ir_llm"
                    logger.info(
                        "[HARVEST] %s → LLM scraper (%d candidates)",
                        company.ticker, len(candidates)
                    )
            except Exception as exc:
                detail["errors"].append(f"LLM scraper: {exc}")
                logger.error(
                    "[HARVEST] LLM scraper error for %s: %s",
                    company.ticker, exc, exc_info=True
                )

        if used_source is None:
            logger.debug(
                "[HARVEST] %s — no source configured. "
                "Add CIK to EDGAR_SOURCES or set ir_docs_url in the Harvester Sources UI.",
                company.ticker
            )

        detail["source_used"] = used_source
        all_details.append(detail)

    logger.info("[HARVEST] %d total candidates before dedup", len(all_candidates))
    summary = await dispatch_candidates(all_candidates)
    logger.info(
        "[HARVEST] Complete — new: %d, skipped: %d, failed: %d",
        summary["new"], summary["skipped"], summary["failed"],
    )
    summary["details"] = all_details
    return summary
