"""
Pure date ↔ period-label utilities.

Consolidates three previously duplicated helpers:
  - services/harvester/dispatcher.py::_fallback_period
  - services/harvester/coverage.py::_period_to_tuple (ordering)
  - services/harvester/coverage_advanced.py::_period_end

Period label conventions (used throughout the codebase):
  - Quarterly: "2025_Q1", "2025_Q2", "2025_Q3", "2025_Q4"
  - Full year: "2025_FY"

No LLM. No DB. Pure functions — trivially testable.

Business rules (e.g. the 75-day reporting-lag rule used by
`services.harvester.coverage.expected_period`) intentionally stay in
their domain modules — only pure date arithmetic lives here.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional


_QUARTER_ENDS = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def quarter_from_date(dt: Optional[datetime] = None) -> str:
    """Return the calendar-quarter period label for a datetime.

    Defaults to 'now' in UTC when no datetime is passed.

    >>> quarter_from_date(datetime(2025, 4, 15, tzinfo=timezone.utc))
    '2025_Q2'
    >>> quarter_from_date(datetime(2025, 12, 31, tzinfo=timezone.utc))
    '2025_Q4'
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}_Q{q}"


def period_end_date(period_label: str) -> Optional[date]:
    """Convert a period label into the last calendar date of that period.

    Returns None for malformed or unsupported labels.

      '2025_Q1' → date(2025, 3, 31)
      '2025_FY' → date(2025, 12, 31)
      'garbage' → None
    """
    if not period_label:
        return None
    try:
        year_s, rest = period_label.split("_", 1)
        year = int(year_s)
    except (ValueError, IndexError):
        return None
    if rest == "FY":
        return date(year, 12, 31)
    if rest.startswith("Q"):
        try:
            q = int(rest[1:])
        except ValueError:
            return None
        end = _QUARTER_ENDS.get(q)
        if end is None:
            return None
        return date(year, end[0], end[1])
    return None


def period_to_tuple(period_label: str) -> tuple[int, int]:
    """Convert a period label to (year, quarter) for ordering.

    FY labels collapse to Q4. Returns (0, 0) for malformed labels so
    comparisons against real periods sort them to the start.

      '2025_Q3' → (2025, 3)
      '2025_FY' → (2025, 4)
      'garbage' → (0, 0)
    """
    if not period_label:
        return (0, 0)
    try:
        year_s, rest = period_label.split("_", 1)
        year = int(year_s)
    except (ValueError, IndexError):
        return (0, 0)
    if rest == "FY":
        return (year, 4)
    if rest.startswith("Q"):
        try:
            return (year, int(rest[1:]))
        except ValueError:
            return (0, 0)
    return (0, 0)


def shift_period(period_label: str, *, quarters: int) -> Optional[str]:
    """Shift a YYYY_QN label by N quarters (positive = forward, negative
    = backward). Returns None for unsupported formats (FY, H1/H2,
    malformed). FY labels are out-of-scope because shifting a full-year
    label by a single quarter has no clean answer.

      shift_period('2026_Q1', quarters=-1)  →  '2025_Q4'
      shift_period('2025_Q4', quarters=+1)  →  '2026_Q1'
      shift_period('2026_Q1', quarters=-4)  →  '2025_Q1'  (year-ago)
    """
    if not period_label or "_Q" not in period_label:
        return None
    try:
        year_s, q_s = period_label.split("_Q", 1)
        year, q = int(year_s), int(q_s)
    except (ValueError, IndexError):
        return None
    abs_q = year * 4 + (q - 1) + quarters
    new_year = abs_q // 4
    new_q = (abs_q % 4) + 1
    return f"{new_year}_Q{new_q}"
