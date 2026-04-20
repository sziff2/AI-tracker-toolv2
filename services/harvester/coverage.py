"""
Coverage Monitor — detect missing expected documents per company.

Answers: "Which companies are missing results for the current reporting period?"
This is different from harvester health (stale/failed) — it checks content completeness.
"""

from datetime import date, timedelta
from typing import Optional

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def expected_period(today: Optional[date] = None) -> str:
    """Latest period whose results should be published by ``today``.

    Uses a 75-day lag after quarter-end (covers 10-K 60-day deadline
    plus a 15-day buffer for processing / publication delays).
    """
    if today is None:
        today = date.today()
    year = today.year
    # Walk backwards through quarter-ends until we find one whose
    # reporting window has closed.
    candidates = [
        (year, 4, date(year, 12, 31)),
        (year, 3, date(year, 9, 30)),
        (year, 2, date(year, 6, 30)),
        (year, 1, date(year, 3, 31)),
        (year - 1, 4, date(year - 1, 12, 31)),
        (year - 1, 3, date(year - 1, 9, 30)),
    ]
    for y, q, qend in candidates:
        if today >= qend + timedelta(days=75):
            return f"{y}_Q{q}"
    return f"{year - 1}_Q3"


def _period_to_tuple(period: str) -> tuple[int, int]:
    """Re-export of services.period_utils.period_to_tuple.
    Kept as a local symbol for backwards compatibility."""
    from services.period_utils import period_to_tuple
    return period_to_tuple(period)


def period_behind(latest: str, expected: str) -> int:
    """Number of quarters *latest* trails *expected*.  0 = up to date."""
    ly, lq = _period_to_tuple(latest)
    ey, eq = _period_to_tuple(expected)
    return max(0, (ey - ly) * 4 + (eq - lq))


async def check_coverage(db: AsyncSession) -> list[dict]:
    """Check document coverage for all active companies.

    Returns one dict per company::

        {
            "ticker", "name",
            "expected_period",     # e.g. "2025_Q4"
            "latest_period",       # latest period_label in documents table
            "quarters_behind",     # 0 = up to date
            "has_earnings",        # earnings doc exists for expected period
            "has_annual",          # annual report exists for expected period
            "gap":  "ok" | "behind" | "missing" | "no_docs",
        }
    """
    exp = expected_period()

    result = await db.execute(text("""
        SELECT
            c.id,
            c.ticker,
            c.name,
            MAX(d.period_label) AS latest_period,
            COUNT(*) FILTER (
                WHERE d.period_label = :expected
                AND d.document_type IN (
                    'earnings_release', '10-Q', '10-K', '20-F', '6-K'
                )
            ) AS earnings_for_period,
            COUNT(*) FILTER (
                WHERE d.period_label = :expected
                AND d.document_type IN (
                    'annual_report', '10-K', '20-F', '40-F'
                )
            ) AS annual_for_period
        FROM companies c
        LEFT JOIN documents d ON d.company_id = c.id
        WHERE c.coverage_status = 'active'
        GROUP BY c.id, c.ticker, c.name
        ORDER BY c.ticker
    """), {"expected": exp})

    rows = result.all()
    coverage = []

    for row in rows:
        latest = row.latest_period
        quarters = period_behind(latest, exp) if latest else 99

        if latest is None:
            gap = "no_docs"
        elif quarters == 0:
            gap = "ok"
        elif quarters <= 1:
            gap = "behind"
        else:
            gap = "missing"

        coverage.append({
            "ticker": row.ticker,
            "name": row.name,
            "expected_period": exp,
            "latest_period": latest,
            "quarters_behind": min(quarters, 99),
            "has_earnings": row.earnings_for_period > 0,
            "has_annual": row.annual_for_period > 0,
            "gap": gap,
        })

    return coverage


def format_coverage_summary(coverage: list[dict]) -> str:
    """One-line summary for logs / Teams reports."""
    ok = sum(1 for c in coverage if c["gap"] == "ok")
    behind = [c for c in coverage if c["gap"] == "behind"]
    missing = [c for c in coverage if c["gap"] == "missing"]
    no_docs = [c for c in coverage if c["gap"] == "no_docs"]
    total = len(coverage)

    parts = [f"{ok}/{total} companies up to date"]
    if behind:
        parts.append(f"{len(behind)} 1Q behind")
    if missing:
        parts.append(f"{len(missing)} missing")
    if no_docs:
        parts.append(f"{len(no_docs)} no docs")
    return " | ".join(parts)


def format_coverage_for_teams(coverage: list[dict]) -> str:
    """Markdown section for the Teams weekly report."""
    exp = coverage[0]["expected_period"] if coverage else "?"
    gaps = [c for c in coverage if c["gap"] in ("behind", "missing", "no_docs")]

    if not gaps:
        return f"**Coverage ({exp}):** All companies up to date."

    lines = [f"**Coverage gaps (expected: {exp}):**"]
    for c in sorted(gaps, key=lambda x: (-x["quarters_behind"], x["ticker"])):
        if c["gap"] == "no_docs":
            lines.append(f"- {c['ticker']}: no documents ingested")
        elif c["gap"] == "missing":
            latest = c["latest_period"] or "none"
            lines.append(
                f"- {c['ticker']}: latest is {latest} "
                f"({c['quarters_behind']}Q behind)"
            )
        else:
            lines.append(
                f"- {c['ticker']}: latest is {c['latest_period']} (1Q behind)"
            )
    return "\n".join(lines)
