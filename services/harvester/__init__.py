"""
Document Harvesting Agent

Priority order per company:
  1. SEC EDGAR  — structured filings, most reliable, no scraping
  2. IR Scraper — fallback for companies not on EDGAR, using
                  ir_docs_url stored in harvester_sources table

The ir_docs_url is set via the Harvester Sources admin panel
in the Data Hub — analysts paste the IR documents page URL
for any company that needs it.
"""

import logging

from sqlalchemy import select

from apps.api.database import AsyncSessionLocal
from apps.api.models import Company, HarvesterSource
from services.harvester.sources.sec_edgar import fetch_sec_edgar, EDGAR_SOURCES
from services.harvester.sources.ir_scraper import scrape_ir_page
from services.harvester.sources.ir_llm_scraper import scrape_ir_with_llm
from services.harvester.dispatcher import dispatch_candidates

logger = logging.getLogger(__name__)


async def run_harvest(tickers: list[str] | None = None) -> dict:
    """
    Run a full harvest cycle.

    For each active company:
      1. Try EDGAR if configured in EDGAR_SOURCES
      2. If no EDGAR config, try IR scraper if ir_docs_url is set in DB
      3. Skip with a log warning if neither source is available

    Args:
        tickers: Restrict to specific tickers. None = all active companies.

    Returns:
        {"new": int, "skipped": int, "failed": int, "tickers": list}
    """
    async with AsyncSessionLocal() as db:
        query = select(Company).where(Company.coverage_status == "active")
        if tickers:
            query = query.where(Company.ticker.in_(tickers))
        result = await db.execute(query)
        companies = result.scalars().all()

        src_result = await db.execute(select(HarvesterSource))
        sources = {s.company_id: s for s in src_result.scalars().all()}

    logger.info("[HARVEST] Starting harvest for %d companies", len(companies))

    all_candidates = []

    for company in companies:
        src = sources.get(company.id)
        used_source = None

        # ── Priority 1: SEC EDGAR ─────────────────────────────────
        if company.ticker in EDGAR_SOURCES:
            try:
                candidates = await fetch_sec_edgar(company.ticker)
                if candidates:
                    all_candidates.extend(candidates)
                    used_source = "edgar"
                    logger.info(
                        "[HARVEST] %s → EDGAR (%d candidates)",
                        company.ticker, len(candidates)
                    )
            except Exception as exc:
                logger.error(
                    "[HARVEST] EDGAR error for %s: %s",
                    company.ticker, exc, exc_info=True
                )

        # ── Priority 2: IR page scraper (regex) ──────────────────
        if used_source is None:
            ir_docs_url = src.ir_docs_url if src else None

            if ir_docs_url:
                try:
                    candidates = await scrape_ir_page(
                        ticker=company.ticker,
                        ir_docs_url=ir_docs_url,
                    )
                    if candidates:
                        all_candidates.extend(candidates)
                        used_source = "ir_scrape"
                        logger.info(
                            "[HARVEST] %s → IR scraper (%d candidates)",
                            company.ticker, len(candidates)
                        )
                except Exception as exc:
                    logger.error(
                        "[HARVEST] IR scraper error for %s: %s",
                        company.ticker, exc, exc_info=True
                    )

        # ── Priority 3: LLM-powered scraper (fallback) ──────────
        if used_source is None:
            ir_docs_url = src.ir_docs_url if src else None

            if ir_docs_url:
                try:
                    candidates = await scrape_ir_with_llm(
                        ticker=company.ticker,
                        company_name=company.name,
                        ir_docs_url=ir_docs_url,
                    )
                    if candidates:
                        all_candidates.extend(candidates)
                        used_source = "ir_llm"
                        logger.info(
                            "[HARVEST] %s → LLM scraper (%d candidates)",
                            company.ticker, len(candidates)
                        )
                except Exception as exc:
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

    logger.info("[HARVEST] %d total candidates before dedup", len(all_candidates))
    summary = await dispatch_candidates(all_candidates)
    logger.info(
        "[HARVEST] Complete — new: %d, skipped: %d, failed: %d",
        summary["new"], summary["skipped"], summary["failed"],
    )
    return summary
