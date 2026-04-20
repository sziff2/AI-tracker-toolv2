"""Tests for services/period_utils.py — pure date ↔ period-label utilities.
No fixtures, no DB, no LLM. Runs in <10ms."""

from datetime import date, datetime, timezone

import pytest

from services.period_utils import (
    period_end_date,
    period_to_tuple,
    quarter_from_date,
    shift_period,
)


class TestQuarterFromDate:
    def test_all_four_quarters(self):
        assert quarter_from_date(datetime(2025, 1, 15, tzinfo=timezone.utc)) == "2025_Q1"
        assert quarter_from_date(datetime(2025, 4, 15, tzinfo=timezone.utc)) == "2025_Q2"
        assert quarter_from_date(datetime(2025, 7, 15, tzinfo=timezone.utc)) == "2025_Q3"
        assert quarter_from_date(datetime(2025, 10, 15, tzinfo=timezone.utc)) == "2025_Q4"

    def test_month_boundaries(self):
        # March = Q1, April = Q2
        assert quarter_from_date(datetime(2025, 3, 31, tzinfo=timezone.utc)) == "2025_Q1"
        assert quarter_from_date(datetime(2025, 4, 1, tzinfo=timezone.utc)) == "2025_Q2"
        # December = Q4
        assert quarter_from_date(datetime(2025, 12, 31, tzinfo=timezone.utc)) == "2025_Q4"

    def test_defaults_to_now(self):
        # Just call it and verify it returns something sane
        result = quarter_from_date(None)
        assert result.startswith("20")
        assert "_Q" in result


class TestPeriodEndDate:
    def test_quarters(self):
        assert period_end_date("2025_Q1") == date(2025, 3, 31)
        assert period_end_date("2025_Q2") == date(2025, 6, 30)
        assert period_end_date("2025_Q3") == date(2025, 9, 30)
        assert period_end_date("2025_Q4") == date(2025, 12, 31)

    def test_full_year(self):
        assert period_end_date("2025_FY") == date(2025, 12, 31)

    def test_malformed(self):
        assert period_end_date("") is None
        assert period_end_date("garbage") is None
        assert period_end_date("2025_H1") is None
        assert period_end_date("2025_Q5") is None
        assert period_end_date("abc_Q1") is None


class TestPeriodToTuple:
    def test_quarters(self):
        assert period_to_tuple("2025_Q1") == (2025, 1)
        assert period_to_tuple("2025_Q4") == (2025, 4)

    def test_fy_collapses_to_q4(self):
        assert period_to_tuple("2025_FY") == (2025, 4)

    def test_malformed_sorts_low(self):
        assert period_to_tuple("") == (0, 0)
        assert period_to_tuple("garbage") == (0, 0)
        assert period_to_tuple("2025_H1") == (0, 0)

    def test_orderable(self):
        periods = ["2025_Q3", "2024_Q1", "2025_FY", "2025_Q1"]
        sorted_periods = sorted(periods, key=period_to_tuple)
        assert sorted_periods == ["2024_Q1", "2025_Q1", "2025_Q3", "2025_FY"]


class TestShiftPeriod:
    def test_backwards_one_quarter(self):
        assert shift_period("2026_Q1", quarters=-1) == "2025_Q4"
        assert shift_period("2025_Q4", quarters=-1) == "2025_Q3"

    def test_forwards_one_quarter(self):
        assert shift_period("2025_Q4", quarters=1) == "2026_Q1"

    def test_year_ago(self):
        assert shift_period("2026_Q1", quarters=-4) == "2025_Q1"

    def test_unsupported_formats(self):
        assert shift_period("2025_FY", quarters=-1) is None
        assert shift_period("2025_H1", quarters=-1) is None
        assert shift_period("garbage", quarters=-1) is None
        assert shift_period("", quarters=1) is None


class TestWrapperBackwardsCompat:
    """The legacy symbols in dispatcher.py, coverage.py,
    coverage_advanced.py, completeness_gate.py should all still work
    and delegate to period_utils."""

    def test_dispatcher_fallback_period(self):
        from services.harvester.dispatcher import _fallback_period
        assert _fallback_period(datetime(2025, 7, 15, tzinfo=timezone.utc)) == "2025_Q3"
        assert _fallback_period(None).startswith("20")

    def test_coverage_period_to_tuple(self):
        from services.harvester.coverage import _period_to_tuple
        assert _period_to_tuple("2025_Q3") == (2025, 3)
        assert _period_to_tuple("garbage") == (0, 0)

    def test_coverage_advanced_period_end(self):
        from services.harvester.coverage_advanced import _period_end
        assert _period_end("2025_Q2") == date(2025, 6, 30)
        assert _period_end("bad") is None

    def test_completeness_gate_shift_period(self):
        from services.completeness_gate import _shift_period
        assert _shift_period("2026_Q1", quarters=-1) == "2025_Q4"
