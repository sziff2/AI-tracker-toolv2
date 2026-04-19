"""Tests for services/harvester/coverage_advanced.py — pure procedural
date math. No LLM, no DB — all fixtures inline."""

from datetime import date, datetime
from uuid import uuid4

import pytest

from services.harvester.coverage_advanced import (
    CompanyCadence,
    _current_reporting_period,
    _gap_severity,
    _infer_frequency,
    _period_end,
    _suggested_sources,
    gap_to_dict,
    CoverageGap,
)


class TestPeriodEnd:
    def test_quarterly(self):
        assert _period_end("2025_Q1") == date(2025, 3, 31)
        assert _period_end("2025_Q2") == date(2025, 6, 30)
        assert _period_end("2025_Q3") == date(2025, 9, 30)
        assert _period_end("2025_Q4") == date(2025, 12, 31)

    def test_full_year(self):
        assert _period_end("2025_FY") == date(2025, 12, 31)

    def test_invalid(self):
        assert _period_end("garbage") is None
        assert _period_end("2025_H1") is None
        assert _period_end("") is None


class TestInferFrequency:
    def test_quarterly_full_cycle(self):
        assert _infer_frequency(["2025_Q1", "2025_Q2", "2025_Q3", "2025_Q4"]) == "quarterly"

    def test_quarterly_from_three_unique(self):
        # Three distinct quarters is enough to call quarterly
        assert _infer_frequency(["2025_Q1", "2025_Q2", "2025_Q3"]) == "quarterly"

    def test_half_yearly(self):
        # Only interim + full year each year → half-yearly filer
        assert _infer_frequency(["2025_Q2", "2025_Q4", "2024_Q2", "2024_Q4"]) == "half_yearly"

    def test_annual_only(self):
        assert _infer_frequency(["2025_FY", "2024_FY", "2023_FY"]) == "annual_only"

    def test_single_period_is_unknown(self):
        assert _infer_frequency(["2025_Q1"]) == "unknown"

    def test_empty_is_unknown(self):
        assert _infer_frequency([]) == "unknown"


class TestCurrentReportingPeriod:
    def _cadence(self, frequency: str, report_lag: int = 45) -> CompanyCadence:
        return CompanyCadence(
            company_id=str(uuid4()),
            ticker="TEST",
            frequency=frequency,
            report_lag_days=report_lag,
            transcript_lag_days=None,
        )

    def test_quarterly_picks_most_recent_closed_quarter(self):
        # Q1 ends 2026-03-31. With 45d lag, earliest expected report is 2026-05-15.
        # Today = 2026-05-20 → Q1 2026 is the current reporting period.
        cadence = self._cadence("quarterly", report_lag=45)
        result = _current_reporting_period(cadence, date(2026, 5, 20))
        assert result == ("2026_Q1", date(2026, 3, 31))

    def test_quarterly_rolls_back_to_prior_year(self):
        # Early January — only Q3 of prior year has closed with 45d lag buffer.
        cadence = self._cadence("quarterly", report_lag=45)
        # 2026-01-10 — Q4 2025 ended Dec 31, +45d = Feb 14. Not yet due.
        # Q3 2025 ended Sep 30, +45d = Nov 14. Due.
        result = _current_reporting_period(cadence, date(2026, 1, 10))
        assert result == ("2025_Q3", date(2025, 9, 30))

    def test_half_yearly(self):
        cadence = self._cadence("half_yearly", report_lag=60)
        # H1 2026 ends Jun 30, +60d = Aug 29. Today = Sep 10 → H1 2026 due.
        result = _current_reporting_period(cadence, date(2026, 9, 10))
        assert result == ("2026_Q2", date(2026, 6, 30))

    def test_annual_only(self):
        cadence = self._cadence("annual_only", report_lag=90)
        # FY 2025 ends Dec 31, +90d = Mar 31 2026. Today = Apr 15 2026 → FY 2025 due.
        result = _current_reporting_period(cadence, date(2026, 4, 15))
        assert result == ("2025_FY", date(2025, 12, 31))

    def test_unknown_frequency_returns_none(self):
        cadence = self._cadence("unknown", report_lag=45)
        assert _current_reporting_period(cadence, date(2026, 5, 20)) is None


class TestGapSeverity:
    def test_source_broken_overrides_everything(self):
        assert _gap_severity(days_overdue=0, sample_size=20, stale_company=True) == "source_broken"
        assert _gap_severity(days_overdue=-5, sample_size=20, stale_company=True) == "source_broken"

    def test_warning_for_not_yet_overdue(self):
        assert _gap_severity(days_overdue=-2, sample_size=10, stale_company=False) == "warning"

    def test_critical_when_very_overdue_with_thick_history(self):
        assert _gap_severity(days_overdue=10, sample_size=10, stale_company=False) == "critical"

    def test_overdue_when_mildly_late(self):
        assert _gap_severity(days_overdue=3, sample_size=10, stale_company=False) == "overdue"

    def test_thin_history_caps_at_overdue(self):
        # sample_size < 4 means we don't trust the cadence enough to scream "critical"
        assert _gap_severity(days_overdue=20, sample_size=2, stale_company=False) == "warning"
        assert _gap_severity(days_overdue=5, sample_size=2, stale_company=False) == "overdue"


class TestSuggestedSources:
    def _cadence(self) -> CompanyCadence:
        return CompanyCadence(
            company_id=str(uuid4()),
            ticker="TEST",
            frequency="quarterly",
            report_lag_days=45,
            transcript_lag_days=2,
        )

    def test_transcript_gets_both_scrapers(self):
        # Transcript / earnings releases warrant the LLM fallback since
        # regex presumably already failed during the weekly harvest.
        assert _suggested_sources(self._cadence(), "transcript") == ["ir_scrape", "ir_llm"]
        assert _suggested_sources(self._cadence(), "earnings_release") == ["ir_scrape", "ir_llm"]

    def test_presentation_gets_both(self):
        assert _suggested_sources(self._cadence(), "presentation") == ["ir_scrape", "ir_llm"]

    def test_other_types_regex_only(self):
        assert _suggested_sources(self._cadence(), "proxy") == ["ir_scrape"]


class TestGapToDict:
    def test_serialises_all_fields(self):
        gap = CoverageGap(
            company_id="abc",
            ticker="ASML",
            name="ASML",
            doc_type="transcript",
            expected_period="2026_Q1",
            expected_by=date(2026, 4, 19),
            days_overdue=3,
            severity="overdue",
            reason="because",
            cadence_frequency="quarterly",
            cadence_sample_size=8,
            sources_to_retry=["ir_scrape", "ir_llm"],
        )
        d = gap_to_dict(gap)
        assert d["ticker"] == "ASML"
        assert d["expected_by"] == "2026-04-19"  # ISO string, not date object
        assert d["severity"] == "overdue"
        assert d["sources_to_retry"] == ["ir_scrape", "ir_llm"]
