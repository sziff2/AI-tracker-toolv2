"""
Coverage parity check — compares the old static 75-day-lag coverage
check against the new learned-cadence Coverage Monitor, per company.

Purpose: validation window before we retire the old `coverage.py` logic
(see CLAUDE.md / Dev plans §8). The new monitor handles half-yearly
filers, Japanese fiscal years, and per-doc-type gaps — but we want 2-3
weeks of production data showing the two systems broadly agree before
cutting over. Systematic disagreements are the signal we need to catch.

No LLM. Just runs both engines and aggregates per-company.

Agreement rules:
  - Both clean → agree
  - Both flagging gaps → agree
  - Old clean, new flagging (non-warning) → disagree: new_flagged_old_clean
  - Old flagging, new clean of overdue/critical → disagree: old_flagged_new_clean

`warning` severity gaps from the new system are deliberately IGNORED
for parity purposes — they mean "approaching, not yet overdue" and the
old system has no equivalent concept. Treating them as active gaps
would produce false disagreements every single week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Severities that should COUNT as "new system is flagging this company"
# — warnings are pre-overdue and have no old-system equivalent.
_ACTIVE_SEVERITIES = {"overdue", "critical", "source_broken"}

_OLD_GAP_CLEAN = "ok"                                              # all other old gaps count as "flagging"


@dataclass
class CompanyParity:
    """One row of the parity table."""
    ticker: str
    name: str
    agreement: str                    # "agree" | "disagree"
    disagree_type: Optional[str] = None   # None | "old_flagged_new_clean" | "new_flagged_old_clean"
    reason: str = ""
    # Old-system verdict
    old_gap: Optional[str] = None
    old_latest_period: Optional[str] = None
    old_expected_period: Optional[str] = None
    old_quarters_behind: Optional[int] = None
    # New-system verdict (aggregated)
    new_active_gap_count: int = 0
    new_gap_severities: list[str] = field(default_factory=list)
    new_gap_doc_types: list[str] = field(default_factory=list)


@dataclass
class ParityReport:
    generated_at: str
    total: int = 0
    agree: int = 0
    disagree: int = 0
    old_flagged_new_clean: int = 0
    new_flagged_old_clean: int = 0
    companies: list[CompanyParity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total": self.total,
                "agree": self.agree,
                "disagree": self.disagree,
                "old_flagged_new_clean": self.old_flagged_new_clean,
                "new_flagged_old_clean": self.new_flagged_old_clean,
            },
            "companies": [asdict(c) for c in self.companies],
        }


def _classify(old_row: dict, new_gaps: list[dict]) -> CompanyParity:
    """Pure function so it's trivially testable. Takes one old coverage
    row (as dict) and the list of new gaps for the same ticker (already
    filtered), returns a CompanyParity."""
    ticker = old_row.get("ticker", "")
    name = old_row.get("name", "")
    old_gap = old_row.get("gap")

    active_gaps = [g for g in new_gaps if g.get("severity") in _ACTIVE_SEVERITIES]
    old_flagging = old_gap != _OLD_GAP_CLEAN
    new_flagging = len(active_gaps) > 0

    row = CompanyParity(
        ticker=ticker,
        name=name,
        agreement="agree",
        old_gap=old_gap,
        old_latest_period=old_row.get("latest_period"),
        old_expected_period=old_row.get("expected_period"),
        old_quarters_behind=old_row.get("quarters_behind"),
        new_active_gap_count=len(active_gaps),
        new_gap_severities=[g["severity"] for g in active_gaps],
        new_gap_doc_types=[g["doc_type"] for g in active_gaps],
    )

    if old_flagging == new_flagging:
        # Both clean OR both flagging → they agree (even if for different reasons)
        row.agreement = "agree"
        return row

    row.agreement = "disagree"
    if new_flagging and not old_flagging:
        row.disagree_type = "new_flagged_old_clean"
        severities = ", ".join(row.new_gap_severities)
        doc_types = ", ".join(row.new_gap_doc_types)
        row.reason = (
            f"Old says up to date (latest={old_row.get('latest_period')}). "
            f"New flags {len(active_gaps)} gap(s): {doc_types} [{severities}]."
        )
    else:  # old_flagging and not new_flagging
        row.disagree_type = "old_flagged_new_clean"
        row.reason = (
            f"Old says {old_gap} ({row.old_quarters_behind}Q behind, "
            f"latest={row.old_latest_period}). "
            f"New finds no overdue/critical gaps."
        )

    return row


async def compare_coverage(db: AsyncSession) -> ParityReport:
    """Run both coverage engines and produce a per-company parity table."""
    from services.harvester.coverage import check_coverage
    from services.harvester.coverage_advanced import find_overdue_gaps, gap_to_dict

    old_coverage = await check_coverage(db)
    new_gaps = await find_overdue_gaps(db)

    # Bucket new gaps by ticker
    gaps_by_ticker: dict[str, list[dict]] = {}
    for g in new_gaps:
        gaps_by_ticker.setdefault(g.ticker, []).append(gap_to_dict(g))

    report = ParityReport(generated_at=datetime.now(timezone.utc).isoformat())
    for old_row in old_coverage:
        ticker = old_row.get("ticker", "")
        row = _classify(old_row, gaps_by_ticker.get(ticker, []))
        report.companies.append(row)
        report.total += 1
        if row.agreement == "agree":
            report.agree += 1
        else:
            report.disagree += 1
            if row.disagree_type == "old_flagged_new_clean":
                report.old_flagged_new_clean += 1
            elif row.disagree_type == "new_flagged_old_clean":
                report.new_flagged_old_clean += 1

    logger.info(
        "[PARITY] %d companies, %d agree, %d disagree "
        "(%d old-flagged-new-clean, %d new-flagged-old-clean)",
        report.total, report.agree, report.disagree,
        report.old_flagged_new_clean, report.new_flagged_old_clean,
    )
    return report
