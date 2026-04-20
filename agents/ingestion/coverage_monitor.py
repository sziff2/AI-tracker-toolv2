"""
Coverage Monitor — detects overdue documents and triggers targeted rescans.

Not an LLM agent. Coordination layer over `services/harvester/coverage_advanced.py`
(gap detection) and `agents.ingestion.orchestrator.IngestionOrchestrator`
(rescan execution).

Responsibilities:
  - Run `find_overdue_gaps()` to get the current per-company gap list.
  - For each gap eligible for auto-rescan, check the rescan log to avoid
    hammering (≤1 auto-rescan per 24h per gap, ≤3 total attempts).
  - Trigger a targeted scan with skip_llm=False (we want best-effort
    recovery of missing docs, not cost containment).
  - Record every attempt in `coverage_rescan_log`.

Runs daily on Celery Beat at 14:00 UTC (between European close and US
open). Can also be invoked manually from the UI Rescan button.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from apps.api.models import CoverageRescanLog
from services.harvester.coverage_advanced import (
    CoverageGap, find_overdue_gaps, gap_to_dict,
)

logger = logging.getLogger(__name__)


# Auto-rescan policy — read from settings (env-overridable).
# Module-level constants kept for tests / callers that import them.
from configs.settings import settings as _settings
_MIN_AUTO_RESCAN_INTERVAL_HOURS = _settings.coverage_min_rescan_interval_hours
_MAX_AUTO_RESCAN_ATTEMPTS_PER_GAP = _settings.coverage_max_rescan_attempts
# Severities eligible for auto-rescan (source_broken stays manual — no
# point hammering a clearly-dead source; analyst should investigate)
_AUTO_RESCAN_SEVERITIES = {"overdue", "critical"}


@dataclass
class CoverageMonitorResult:
    """Summary of one Coverage Monitor run."""
    gaps_found: int = 0
    rescans_triggered: int = 0
    rescans_skipped_recent: int = 0
    rescans_skipped_exhausted: int = 0
    rescan_successes: int = 0
    rescan_no_new: int = 0
    rescan_errors: int = 0
    gap_details: list[dict] = field(default_factory=list)
    triggered_tickers: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Rescan gating
# ─────────────────────────────────────────────────────────────────

async def _recent_rescan_count(
    db: AsyncSession,
    company_id,
    doc_type: str,
    expected_period: str,
    *,
    since_hours: int,
) -> int:
    """Count how many rescans have been triggered for this exact gap in
    the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    q = await db.execute(
        select(CoverageRescanLog.id).where(and_(
            CoverageRescanLog.company_id == company_id,
            CoverageRescanLog.doc_type == doc_type,
            CoverageRescanLog.expected_period == expected_period,
            CoverageRescanLog.triggered_at >= cutoff,
        ))
    )
    return len(q.all())


async def _total_auto_rescans_for_gap(
    db: AsyncSession,
    company_id,
    doc_type: str,
    expected_period: str,
) -> int:
    """How many times has CoverageMonitor auto-rescanned this exact gap
    in total? Caps the back-off at _MAX_AUTO_RESCAN_ATTEMPTS_PER_GAP."""
    q = await db.execute(
        select(CoverageRescanLog.id).where(and_(
            CoverageRescanLog.company_id == company_id,
            CoverageRescanLog.doc_type == doc_type,
            CoverageRescanLog.expected_period == expected_period,
            CoverageRescanLog.triggered_by == "auto",
        ))
    )
    return len(q.all())


async def _record_rescan(
    db: AsyncSession,
    gap: CoverageGap,
    triggered_by: str,
    summary: dict | None,
    error_message: str | None,
) -> None:
    """Persist the rescan attempt."""
    if summary and "new" in summary:
        candidates = int(summary.get("new", 0))
        if candidates > 0:
            result = "success"
        elif error_message:
            result = "error"
        else:
            result = "no_new_candidates"
    else:
        result = "error" if error_message else "no_new_candidates"
        candidates = 0

    db.add(CoverageRescanLog(
        id=uuid.uuid4(),
        company_id=uuid.UUID(gap.company_id),
        ticker=gap.ticker,
        doc_type=gap.doc_type,
        expected_period=gap.expected_period,
        triggered_by=triggered_by,
        triggered_at=datetime.now(timezone.utc),
        sources_tried=gap.sources_to_retry,
        candidates_found=candidates,
        result=result,
        error_message=(error_message or "")[:1000] if error_message else None,
    ))
    await db.commit()


# ─────────────────────────────────────────────────────────────────
# Core entry point
# ─────────────────────────────────────────────────────────────────

class CoverageMonitor:
    """Daily-run coordination layer. Not an LLM agent."""

    async def run_daily_check(
        self,
        *,
        auto_trigger: bool = True,
    ) -> CoverageMonitorResult:
        """Find gaps and optionally trigger auto-rescans.

        Args:
            auto_trigger: if False, just report gaps without triggering
                          any scans (useful for the UI panel load path).
        """
        from agents.ingestion.orchestrator import IngestionOrchestrator
        orch = IngestionOrchestrator()

        result = CoverageMonitorResult()

        async with AsyncSessionLocal() as db:
            gaps = await find_overdue_gaps(db)
            result.gaps_found = len(gaps)
            result.gap_details = [gap_to_dict(g) for g in gaps]

            if not auto_trigger:
                return result

            # Group by ticker so we don't scan the same company twice when
            # it has multiple missing doc types this cycle.
            tickers_to_scan: dict[str, list[CoverageGap]] = {}

            for gap in gaps:
                if gap.severity not in _AUTO_RESCAN_SEVERITIES:
                    continue

                recent = await _recent_rescan_count(
                    db, uuid.UUID(gap.company_id), gap.doc_type,
                    gap.expected_period,
                    since_hours=_MIN_AUTO_RESCAN_INTERVAL_HOURS,
                )
                if recent > 0:
                    result.rescans_skipped_recent += 1
                    logger.info(
                        "[COVERAGE] Skipping auto-rescan %s/%s/%s — rescanned %d×"
                        " in last %dh",
                        gap.ticker, gap.doc_type, gap.expected_period,
                        recent, _MIN_AUTO_RESCAN_INTERVAL_HOURS,
                    )
                    continue

                total = await _total_auto_rescans_for_gap(
                    db, uuid.UUID(gap.company_id), gap.doc_type,
                    gap.expected_period,
                )
                if total >= _MAX_AUTO_RESCAN_ATTEMPTS_PER_GAP:
                    result.rescans_skipped_exhausted += 1
                    logger.info(
                        "[COVERAGE] Skipping auto-rescan %s/%s/%s — %d auto "
                        "attempts already made (cap %d)",
                        gap.ticker, gap.doc_type, gap.expected_period,
                        total, _MAX_AUTO_RESCAN_ATTEMPTS_PER_GAP,
                    )
                    continue

                tickers_to_scan.setdefault(gap.ticker, []).append(gap)

        # Batched scan (one call per ticker so we don't re-harvest the
        # whole portfolio for a handful of gaps).
        for ticker, ticker_gaps in tickers_to_scan.items():
            try:
                summary = await orch.run_targeted_scan(
                    tickers=[ticker], skip_llm=False,
                )
                async with AsyncSessionLocal() as db:
                    for gap in ticker_gaps:
                        await _record_rescan(db, gap, "auto", summary, None)

                if summary.get("new", 0) > 0:
                    result.rescan_successes += len(ticker_gaps)
                else:
                    result.rescan_no_new += len(ticker_gaps)
                result.rescans_triggered += len(ticker_gaps)
                result.triggered_tickers.append(ticker)
                logger.info(
                    "[COVERAGE] Auto-rescanned %s for %d gap(s) → new=%d",
                    ticker, len(ticker_gaps), summary.get("new", 0),
                )

            except Exception as exc:
                result.rescan_errors += len(ticker_gaps)
                logger.error(
                    "[COVERAGE] Auto-rescan failed for %s: %s",
                    ticker, exc, exc_info=True,
                )
                async with AsyncSessionLocal() as db:
                    for gap in ticker_gaps:
                        await _record_rescan(db, gap, "auto", None, str(exc))

        logger.info(
            "[COVERAGE] Daily run — %d gaps, %d rescans triggered "
            "(%d skipped recent, %d skipped exhausted, %d success, %d no-new, %d err)",
            result.gaps_found, result.rescans_triggered,
            result.rescans_skipped_recent, result.rescans_skipped_exhausted,
            result.rescan_successes, result.rescan_no_new, result.rescan_errors,
        )
        return result

    async def rescan_one_gap(
        self,
        ticker: str,
        *,
        doc_type: Optional[str] = None,
        expected_period: Optional[str] = None,
    ) -> dict:
        """Manually trigger a rescan for a specific gap (called from the
        UI Rescan button). Always goes through with skip_llm=False — the
        analyst explicitly asked for it so cost concerns don't apply."""
        from agents.ingestion.orchestrator import IngestionOrchestrator
        orch = IngestionOrchestrator()

        summary = await orch.run_targeted_scan(tickers=[ticker], skip_llm=False)

        # Record the attempt (best-effort — if the gap doesn't resolve
        # exactly to one row, we still log generally).
        async with AsyncSessionLocal() as db:
            from apps.api.models import Company
            cq = await db.execute(
                select(Company).where(Company.ticker == ticker)
            )
            company = cq.scalar_one_or_none()
            if company:
                db.add(CoverageRescanLog(
                    id=uuid.uuid4(),
                    company_id=company.id,
                    ticker=ticker,
                    doc_type=doc_type,
                    expected_period=expected_period,
                    triggered_by="manual",
                    triggered_at=datetime.now(timezone.utc),
                    sources_tried=["ir_scrape", "ir_llm"],
                    candidates_found=int(summary.get("new", 0)),
                    result="success" if summary.get("new", 0) > 0 else "no_new_candidates",
                ))
                await db.commit()

        return summary
