"""
Advanced coverage — per-company expected-date prediction.

The baseline `coverage.py` uses a static 75-day lag after quarter-end to
decide whether a company is up to date. That's correct for US 10-Q
filers but misses half-yearly reporters (most UK/EU), Japanese fiscal
years ending March, and anyone with unusual cadence.

This module learns each company's actual reporting pattern from its
historical document dates and flags gaps against that learned pattern
instead of a universal rule.

Outputs a single list of `CoverageGap` records that the Coverage Monitor
agent consumes. No LLM — pure procedural date math.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Company, Document

logger = logging.getLogger(__name__)


# Doc types that signal a primary period result (i.e. earnings cycle)
_EARNINGS_DOC_TYPES = {
    "earnings_release", "10-Q", "10-K", "20-F", "40-F", "6-K",
    "annual_report", "interim_report", "half_year_report",
}
# Doc types that should trail an earnings release within a few days
_TRANSCRIPT_DOC_TYPES = {"transcript"}
_PRESENTATION_DOC_TYPES = {"presentation", "investor_presentation"}


# ─────────────────────────────────────────────────────────────────
# Cadence inference
# ─────────────────────────────────────────────────────────────────

@dataclass
class CompanyCadence:
    """What we know about a company's reporting rhythm from history."""
    company_id: str
    ticker: str
    frequency: str                # "quarterly" | "half_yearly" | "annual_only" | "unknown"
    report_lag_days: int          # median days from period-end to earnings release
    transcript_lag_days: Optional[int]  # days from earnings release to transcript
    doc_types_observed: set[str] = field(default_factory=set)
    last_earnings_release: Optional[date] = None
    last_earnings_period: Optional[str] = None
    sample_size: int = 0


def _period_end(period_label: str) -> Optional[date]:
    """Re-export of services.period_utils.period_end_date.
    Kept as a local symbol for backwards compatibility with imports
    elsewhere in the coverage pipeline."""
    from services.period_utils import period_end_date
    return period_end_date(period_label)


def _infer_frequency(periods: list[str]) -> str:
    """Look at the period labels that this company actually files.
    If we see Q1/Q2/Q3/Q4 → quarterly. Only Q2/Q4 (or FY+H1) → half_yearly.
    Only FY → annual_only. Too few samples → unknown."""
    if len(periods) < 2:
        return "unknown"
    quarters = set()
    has_fy = False
    for p in periods:
        if "_Q" in p:
            try:
                quarters.add(int(p.split("_Q")[1][:1]))
            except (ValueError, IndexError):
                pass
        elif p.endswith("_FY"):
            has_fy = True
    if {1, 3} & quarters and {2, 4} & quarters:
        return "quarterly"
    if quarters and len(quarters) <= 2:
        # e.g. only Q2 and Q4 — half-yearly filer reporting interim + full year
        return "half_yearly"
    if has_fy and not quarters:
        return "annual_only"
    if len(quarters) >= 3:
        return "quarterly"
    return "unknown"


async def analyze_company_cadence(
    db: AsyncSession, company_id, ticker: str, *, lookback_docs: int = 16
) -> CompanyCadence:
    """Query the last N documents for a company and infer its reporting cadence."""
    q = await db.execute(
        select(
            Document.document_type,
            Document.period_label,
            Document.published_at,
        )
        .where(Document.company_id == company_id)
        .where(Document.published_at.isnot(None))
        .order_by(Document.published_at.desc())
        .limit(lookback_docs)
    )
    rows = q.all()

    doc_types_observed: set[str] = set()
    period_labels: list[str] = []
    earnings_points: list[tuple[str, date, date]] = []  # (period, period_end, published)
    transcript_lags: list[int] = []

    # Pair earnings_release → transcript by period_label
    earnings_by_period: dict[str, date] = {}
    transcripts_by_period: dict[str, date] = {}

    for r in rows:
        if r.document_type:
            doc_types_observed.add(r.document_type)
        if r.period_label:
            period_labels.append(r.period_label)
        published = r.published_at.date() if isinstance(r.published_at, datetime) else r.published_at
        if not published:
            continue
        if r.document_type in _EARNINGS_DOC_TYPES and r.period_label:
            pend = _period_end(r.period_label)
            if pend:
                earnings_points.append((r.period_label, pend, published))
                earnings_by_period.setdefault(r.period_label, published)
        if r.document_type in _TRANSCRIPT_DOC_TYPES and r.period_label:
            transcripts_by_period.setdefault(r.period_label, published)

    for period, earn_date in earnings_by_period.items():
        trans_date = transcripts_by_period.get(period)
        if trans_date and earn_date and trans_date >= earn_date:
            transcript_lags.append((trans_date - earn_date).days)

    # Median earnings lag (period_end → publish)
    lag_days_list = [(pub - pend).days for _, pend, pub in earnings_points if pub >= pend]
    report_lag_days = int(statistics.median(lag_days_list)) if lag_days_list else 75

    # Most recent earnings release
    last_earnings_release: Optional[date] = None
    last_earnings_period: Optional[str] = None
    if earnings_points:
        earnings_points_sorted = sorted(earnings_points, key=lambda t: t[2], reverse=True)
        last_earnings_period, _, last_earnings_release = earnings_points_sorted[0]

    return CompanyCadence(
        company_id=str(company_id),
        ticker=ticker,
        frequency=_infer_frequency(period_labels),
        report_lag_days=max(report_lag_days, 30),  # floor at 30d (sanity)
        transcript_lag_days=int(statistics.median(transcript_lags)) if transcript_lags else None,
        doc_types_observed=doc_types_observed,
        last_earnings_release=last_earnings_release,
        last_earnings_period=last_earnings_period,
        sample_size=len(rows),
    )


# ─────────────────────────────────────────────────────────────────
# Expected-date prediction
# ─────────────────────────────────────────────────────────────────

def _current_reporting_period(cadence: CompanyCadence, today: date) -> Optional[tuple[str, date]]:
    """What period SHOULD this company be reporting right now? Returns
    (period_label, period_end_date) or None if no clean answer."""
    if cadence.frequency == "quarterly":
        candidates = []
        for y in (today.year, today.year - 1):
            for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
                candidates.append((f"{y}_Q{q}", date(y, m, d)))
        candidates.sort(key=lambda t: t[1], reverse=True)
        for period, pend in candidates:
            if today >= pend + timedelta(days=cadence.report_lag_days):
                return (period, pend)
        return None

    if cadence.frequency == "half_yearly":
        candidates = []
        for y in (today.year, today.year - 1):
            candidates.append((f"{y}_Q2", date(y, 6, 30)))
            candidates.append((f"{y}_Q4", date(y, 12, 31)))
        candidates.sort(key=lambda t: t[1], reverse=True)
        for period, pend in candidates:
            if today >= pend + timedelta(days=cadence.report_lag_days):
                return (period, pend)
        return None

    if cadence.frequency == "annual_only":
        for y in (today.year, today.year - 1):
            pend = date(y, 12, 31)
            if today >= pend + timedelta(days=cadence.report_lag_days):
                return (f"{y}_FY", pend)
        return None

    return None


@dataclass
class CoverageGap:
    """One missing document for one company."""
    company_id: str
    ticker: str
    name: str
    doc_type: str                # "earnings_release" | "transcript" | "presentation"
    expected_period: str         # e.g. "2026_Q1"
    expected_by: date            # date by which this doc should exist
    days_overdue: int
    severity: str                # "warning" | "overdue" | "critical" | "source_broken"
    reason: str                  # one-line human-readable reason
    cadence_frequency: str
    cadence_sample_size: int
    sources_to_retry: list[str]  # suggested sources (e.g. ["ir_scrape", "ir_llm"])


def _gap_severity(days_overdue: int, sample_size: int, stale_company: bool) -> str:
    """Classify how urgent a gap is."""
    if stale_company:
        return "source_broken"
    if days_overdue < 0:
        return "warning"            # approaching, not yet overdue
    if sample_size < 4:
        # thin history — don't scream, just warn
        return "overdue" if days_overdue <= 14 else "warning"
    if days_overdue > 7:
        return "critical"
    return "overdue"


def _suggested_sources(cadence: CompanyCadence, doc_type: str) -> list[str]:
    """Which sources to retry for a gap. Prefer LLM scraper because regex
    already failed — this is an auto-recovery path."""
    sources: list[str] = []
    # EDGAR / Investegate membership is known at import time but we don't
    # have the ticker->source map here; add both scrape paths which cover
    # the residual cases when EDGAR / Investegate already ran in the
    # weekly harvest and didn't find anything.
    if doc_type in _TRANSCRIPT_DOC_TYPES or doc_type == "earnings_release":
        sources.extend(["ir_scrape", "ir_llm"])
    elif doc_type in _PRESENTATION_DOC_TYPES:
        sources.extend(["ir_scrape", "ir_llm"])
    else:
        sources.extend(["ir_scrape"])
    return sources


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

async def find_overdue_gaps(
    db: AsyncSession,
    *,
    today: Optional[date] = None,
    warning_lead_days: int = 3,
    stale_company_days: int = 120,
) -> list[CoverageGap]:
    """Inspect every active company. For each, infer cadence, predict the
    period that should currently be reporting, and flag any missing
    doc types (earnings, transcript, presentation) that the company
    has historically produced.

    A gap is emitted only for doc types the company has produced before —
    no point demanding a transcript for a company that never hosts
    earnings calls.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    q = await db.execute(
        select(Company.id, Company.ticker, Company.name)
        .where(Company.coverage_status == "active")
        .order_by(Company.ticker)
    )
    companies = q.all()

    gaps: list[CoverageGap] = []

    for co_id, ticker, name in companies:
        cadence = await analyze_company_cadence(db, co_id, ticker)

        # If the company hasn't published anything in a long time, flag
        # the whole thing as source_broken rather than per-doc gaps.
        days_since_last = (
            (today - cadence.last_earnings_release).days
            if cadence.last_earnings_release else 9999
        )
        stale_company = days_since_last > stale_company_days

        current = _current_reporting_period(cadence, today)
        if not current:
            # Can't determine expected period (too little history, or
            # nothing is due yet). Skip.
            continue

        expected_period, period_end = current
        earnings_expected_by = period_end + timedelta(days=cadence.report_lag_days)

        # Check each doc type the company has produced before
        check_types: list[tuple[str, date]] = []
        if "earnings_release" in cadence.doc_types_observed \
                or "10-Q" in cadence.doc_types_observed \
                or "10-K" in cadence.doc_types_observed \
                or "20-F" in cadence.doc_types_observed \
                or "interim_report" in cadence.doc_types_observed \
                or "6-K" in cadence.doc_types_observed \
                or "half_year_report" in cadence.doc_types_observed:
            check_types.append(("earnings_release", earnings_expected_by))
        if "transcript" in cadence.doc_types_observed:
            lag = cadence.transcript_lag_days or 3
            check_types.append(("transcript", earnings_expected_by + timedelta(days=lag)))
        if any(t in cadence.doc_types_observed for t in _PRESENTATION_DOC_TYPES):
            check_types.append(("presentation", earnings_expected_by + timedelta(days=1)))

        if not check_types:
            continue

        # Pull documents that EXIST for the expected period
        exist_q = await db.execute(
            select(Document.document_type).where(
                Document.company_id == co_id,
                Document.period_label == expected_period,
            )
        )
        existing_types = {row[0] for row in exist_q.all()}

        # A document "exists" for a check_type if any equivalent type is present
        earnings_equivalents = {
            "earnings_release", "10-Q", "10-K", "20-F", "40-F", "6-K",
            "annual_report", "interim_report", "half_year_report",
        }
        presentation_equivalents = {"presentation", "investor_presentation"}
        transcript_equivalents = {"transcript"}

        for doc_type, expected_by in check_types:
            if doc_type == "earnings_release":
                covered = bool(existing_types & earnings_equivalents)
            elif doc_type == "transcript":
                covered = bool(existing_types & transcript_equivalents)
            else:  # presentation
                covered = bool(existing_types & presentation_equivalents)

            if covered:
                continue
            days_overdue = (today - expected_by).days
            if days_overdue < -warning_lead_days and not stale_company:
                # Not yet approaching — skip entirely.
                continue

            severity = _gap_severity(days_overdue, cadence.sample_size, stale_company)
            if stale_company:
                reason = (
                    f"No documents from {ticker} in {days_since_last} days — "
                    f"source may be broken"
                )
            elif days_overdue >= 0:
                reason = (
                    f"{doc_type} for {expected_period} expected by "
                    f"{expected_by.isoformat()} — {days_overdue}d overdue "
                    f"({cadence.frequency}, median lag "
                    f"{cadence.report_lag_days}d, n={cadence.sample_size})"
                )
            else:
                reason = (
                    f"{doc_type} for {expected_period} expected by "
                    f"{expected_by.isoformat()} — {-days_overdue}d away"
                )

            gaps.append(CoverageGap(
                company_id=str(co_id),
                ticker=ticker,
                name=name,
                doc_type=doc_type,
                expected_period=expected_period,
                expected_by=expected_by,
                days_overdue=max(days_overdue, 0),
                severity=severity,
                reason=reason,
                cadence_frequency=cadence.frequency,
                cadence_sample_size=cadence.sample_size,
                sources_to_retry=_suggested_sources(cadence, doc_type),
            ))

    return gaps


def gap_to_dict(g: CoverageGap) -> dict:
    """Serialisation helper for API / UI."""
    return {
        "company_id":     g.company_id,
        "ticker":         g.ticker,
        "name":           g.name,
        "doc_type":       g.doc_type,
        "expected_period": g.expected_period,
        "expected_by":    g.expected_by.isoformat(),
        "days_overdue":   g.days_overdue,
        "severity":       g.severity,
        "reason":         g.reason,
        "cadence_frequency":   g.cadence_frequency,
        "cadence_sample_size": g.cadence_sample_size,
        "sources_to_retry":    g.sources_to_retry,
    }
