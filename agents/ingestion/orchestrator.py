"""
Ingestion Orchestrator — decides WHEN to scan, WHICH companies, WITH WHAT sources.

Not an LLM agent. This is a thin coordination layer that wraps the existing
harvester plumbing. Its purpose is to centralise the "what to scan next"
decision so future agents (Source Quality, Event Scanner) have one hook
point rather than mutating the harvester entry function directly.

v1 behaviour is almost a pass-through:
  - tier="portfolio" → companies with coverage_status='active'
  - tier="watchlist" → companies with coverage_status='watchlist'
  - tier="all"       → portfolio + watchlist

LLM triage runs inside dispatcher.py on each candidate (see _run_triage).
Source selection is not yet tier-differentiated — Sprint 3 (Source Quality)
will read the scorecard and prune / promote sources per company.
"""

import logging
from typing import Iterable

from sqlalchemy import select

from apps.api.database import AsyncSessionLocal
from apps.api.models import Company

logger = logging.getLogger(__name__)


_TIER_TO_COVERAGE_STATUS: dict[str, list[str]] = {
    "portfolio": ["active"],
    "watchlist": ["watchlist"],
    "all":       ["active", "watchlist"],
}


class IngestionOrchestrator:
    """Coordinates scheduled scans across tiers of companies."""

    async def _tickers_for_tier(self, tier: str) -> list[str]:
        statuses = _TIER_TO_COVERAGE_STATUS.get(tier)
        if not statuses:
            raise ValueError(f"Unknown tier: {tier!r}. Expected one of {list(_TIER_TO_COVERAGE_STATUS)}.")

        async with AsyncSessionLocal() as db:
            r = await db.execute(
                select(Company.ticker).where(Company.coverage_status.in_(statuses))
            )
            return [row[0] for row in r.all() if row[0]]

    async def run_scheduled_scan(
        self,
        tier: str = "portfolio",
        *,
        skip_llm: bool = True,
        tickers: Iterable[str] | None = None,
    ) -> dict:
        """
        Run a scheduled harvest over the given tier.

        Args:
            tier: "portfolio" | "watchlist" | "all"
            skip_llm: defaults to True (matches weekly cron behaviour — the
                      LLM scraper is expensive and mostly overkill when regex
                      + EDGAR cover the main cases).
            tickers: explicit ticker override (e.g. from manual "Rerun" in UI).
                     When set, `tier` is ignored.

        Returns the same shape as services.harvester.run_harvest():
            {"new": int, "skipped": int, "failed": int, "tickers": list, "details": [...]}
        """
        from services.harvester import run_harvest

        if tickers:
            ticker_list = [t for t in tickers if t]
            logger.info("[ORCH] Explicit ticker list (%d) — tier ignored", len(ticker_list))
        else:
            ticker_list = await self._tickers_for_tier(tier)
            logger.info("[ORCH] Scheduled scan tier=%s → %d companies", tier, len(ticker_list))

        if not ticker_list:
            logger.info("[ORCH] No companies matched tier=%s — nothing to scan", tier)
            return {"new": 0, "skipped": 0, "failed": 0, "tickers": [], "details": []}

        summary = await run_harvest(tickers=ticker_list, skip_llm=skip_llm)
        summary["tier"] = tier if not tickers else "explicit"
        return summary

    async def run_targeted_scan(
        self,
        tickers: list[str],
        *,
        skip_llm: bool = False,
    ) -> dict:
        """
        Event-driven scan for a small set of tickers (used by the future
        Event Scanner agent — e.g. "EDGAR filing detected, scan IR page now").

        Defaults skip_llm=False because targeted scans are low-volume and
        we want the best source coverage.
        """
        return await self.run_scheduled_scan(
            tier="all",
            skip_llm=skip_llm,
            tickers=tickers,
        )
