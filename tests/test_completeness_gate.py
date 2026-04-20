"""Tests for services/completeness_gate.py — deterministic pre-flight
checks. No LLM, no DB — externals mocked via AsyncMock / MagicMock."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.completeness_gate import (
    HALT_INCOMPLETE, PROCEED, PROCEED_WITH_CAVEATS,
    compute_completeness, compute_source_coverage,
    _metric_matches_any, _shift_period,
)


# ─────────────────────────────────────────────────────────────────
# Pure helpers (no DB)
# ─────────────────────────────────────────────────────────────────

class TestMetricMatchesAny:
    def test_substring_case_insensitive(self):
        assert _metric_matches_any("EPS Diluted", {"eps"}) is True
        assert _metric_matches_any("Net Interest Margin", {"nim", "margin"}) is True
        assert _metric_matches_any("Revenue", {"eps"}) is False
        assert _metric_matches_any("", {"eps"}) is False
        assert _metric_matches_any(None, {"eps"}) is False


class TestShiftPeriod:
    def test_quarter_backwards(self):
        assert _shift_period("2026_Q1", quarters=-1) == "2025_Q4"
        assert _shift_period("2026_Q3", quarters=-1) == "2026_Q2"
        assert _shift_period("2026_Q1", quarters=-4) == "2025_Q1"   # YoY

    def test_quarter_forwards(self):
        assert _shift_period("2025_Q4", quarters=1) == "2026_Q1"

    def test_non_quarter_formats_return_none(self):
        assert _shift_period("2025_FY", quarters=-1) is None
        assert _shift_period("2025_H1", quarters=-1) is None
        assert _shift_period("garbage", quarters=-1) is None
        assert _shift_period("", quarters=-1) is None


# ─────────────────────────────────────────────────────────────────
# Test helpers — mock DB session builders
# ─────────────────────────────────────────────────────────────────

def _mock_metric(metric_name: str, *, segment: str | None = None,
                metric_value: float | None = None) -> MagicMock:
    """Build a mock ExtractedMetric row."""
    m = MagicMock()
    m.metric_name = metric_name
    m.segment = segment
    m.metric_value = metric_value
    return m


class _FakeResult:
    """Approximates SQLAlchemy Result for the subset we use."""
    def __init__(self, items: list, scalar_value=None):
        self._items = items
        self._scalar = scalar_value

    def scalars(self):
        inner = MagicMock()
        inner.all = MagicMock(return_value=self._items)
        return inner

    def all(self):
        return self._items

    def scalar(self):
        return self._scalar


def _make_db(execute_results: list) -> AsyncMock:
    """Build an AsyncMock session where each call to db.execute returns
    the next FakeResult from `execute_results` in order. Lets tests
    script the sequence of query outcomes."""
    db = AsyncMock()
    iterator = iter(execute_results)

    async def _execute(*args, **kwargs):
        return next(iterator)

    db.execute = _execute
    return db


# ─────────────────────────────────────────────────────────────────
# compute_completeness
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestComputeCompleteness:
    async def test_all_required_plus_rich_recommended_proceeds(self):
        # Current period metrics: EPS + margin + revenue + segments + BS + CF
        current = [
            _mock_metric("EPS Diluted", metric_value=1.04),
            _mock_metric("Net Interest Margin", metric_value=3.45),
            _mock_metric("Revenue", metric_value=2065),
            _mock_metric("Revenue", segment="retail_auto", metric_value=1200),
            _mock_metric("Revenue", segment="corporate_finance", metric_value=865),
            _mock_metric("Total Assets", metric_value=185000),
            _mock_metric("Operating Cash Flow", metric_value=450),
            _mock_metric("GUIDANCE: FY NIM",  segment="guidance",
                         metric_value=3.45),
        ]
        # Confidence profile with 12 signals → mgmt_language passes
        profile_row = ({"signals": [f"item_{i}" for i in range(12)]},)

        # Execute order: current metrics → prior_comparator → mgmt_language
        # → dual_comparators → industry_kpis (matches compute_completeness)
        db = _make_db([
            _FakeResult(current),                    # current period metrics
            _FakeResult([], scalar_value=5),         # prior_comparator count > 0
            _FakeResult([profile_row]),              # mgmt language
            _FakeResult([("EPS Diluted", "2025_Q4"), ("EPS Diluted", "2025_Q1")]),  # dual comparators
            _FakeResult([("NIM",), ("Revenue",)]),   # industry KPIs check
        ])

        report = await compute_completeness(db, "co-uuid", "2026_Q1")

        # All four required checks pass
        assert report.status == PROCEED
        assert report.missing_required == []
        assert report.required["current_eps"] is True
        assert report.required["margin_metric"] is True
        assert report.required["forward_guidance"] is True
        assert report.required["prior_period_comparator"] is True
        assert report.recommended_score >= 0.7

    async def test_missing_eps_halts(self):
        current = [
            _mock_metric("Net Interest Margin"),
            _mock_metric("GUIDANCE: FY", segment="guidance"),
        ]
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),  # prior_comparator
            _FakeResult([]),                  # dual comparators empty
            _FakeResult([]),                  # mgmt language empty
            _FakeResult([]),                  # industry kpis empty
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        assert report.status == HALT_INCOMPLETE
        assert "current_eps" in report.missing_required

    async def test_missing_guidance_halts(self):
        current = [
            _mock_metric("EPS Diluted"),
            _mock_metric("Operating Margin"),
        ]
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),  # prior_comparator
            _FakeResult([]),
            _FakeResult([]),
            _FakeResult([]),
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        assert report.status == HALT_INCOMPLETE
        assert "forward_guidance" in report.missing_required

    async def test_all_required_but_thin_recommended_caveats(self):
        # Required pass, but only 1 of 7 recommended (revenue) → ~14% < 50% → HALT
        current = [
            _mock_metric("EPS Diluted"),
            _mock_metric("Operating Margin"),
            _mock_metric("Revenue"),
            _mock_metric("GUIDANCE: Rev", segment="guidance"),
        ]
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),
            _FakeResult([]),
            _FakeResult([]),
            _FakeResult([]),
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        # With only 1 of 7 recommended populated the stricter <50% threshold fires
        assert report.status == HALT_INCOMPLETE
        assert "recommended" in report.reason.lower()

    async def test_all_required_moderate_recommended_proceeds_with_caveats(self):
        # 4 of 7 recommended = ~57% — between 50% and 70% → PROCEED_WITH_CAVEATS
        current = [
            _mock_metric("EPS Diluted"),
            _mock_metric("Operating Margin"),
            _mock_metric("Revenue"),
            _mock_metric("Revenue", segment="retail"),
            _mock_metric("Revenue", segment="corporate"),
            _mock_metric("Total Assets"),
            _mock_metric("GUIDANCE: Rev", segment="guidance"),
        ]
        profile_row = ({"signals": [1] * 12},)
        # Order: current → prior_comparator → mgmt_language → dual_comparators → industry_kpis
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),
            _FakeResult([profile_row]),     # mgmt language OK
            _FakeResult([]),                # no dual comparators
            _FakeResult([]),                # no industry kpis
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        assert report.status == PROCEED_WITH_CAVEATS
        assert 0.5 <= report.recommended_score < 0.7

    async def test_guidance_via_segment_alone_counts(self):
        # Guidance recognised via segment="guidance" even if name has no prefix
        current = [
            _mock_metric("EPS Diluted"),
            _mock_metric("Gross Margin"),
            _mock_metric("Organic Growth", segment="guidance"),
        ]
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),
            _FakeResult([]),
            _FakeResult([]),
            _FakeResult([]),
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        # Forward guidance should be detected
        assert report.required["forward_guidance"] is True


# ─────────────────────────────────────────────────────────────────
# compute_source_coverage
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestComputeSourceCoverage:
    async def test_all_present_proceeds(self):
        db = _make_db([
            _FakeResult([("earnings_release",), ("transcript",), ("presentation",)]),
        ])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.status == PROCEED
        assert report.has_results_doc is True
        assert report.has_transcript is True
        assert report.has_presentation is True

    async def test_missing_transcript_caveats(self):
        db = _make_db([
            _FakeResult([("earnings_release",), ("presentation",)]),
        ])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.status == PROCEED_WITH_CAVEATS
        assert "transcript" in report.missing_recommended
        assert "transcript" in report.reason.lower()

    async def test_missing_results_doc_halts(self):
        db = _make_db([
            _FakeResult([("transcript",), ("presentation",)]),
        ])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.status == HALT_INCOMPLETE
        assert report.has_results_doc is False

    async def test_ten_q_satisfies_results_doc(self):
        # 10-Q alone is a valid results doc
        db = _make_db([
            _FakeResult([("10-Q",), ("transcript",), ("presentation",)]),
        ])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.status == PROCEED
        assert report.has_results_doc is True

    async def test_investor_presentation_synonym(self):
        # Some extractors use "investor_presentation" not "presentation"
        db = _make_db([
            _FakeResult([
                ("10-Q",), ("transcript",), ("investor_presentation",),
            ]),
        ])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.has_presentation is True
        assert report.status == PROCEED

    async def test_no_docs_at_all_halts(self):
        db = _make_db([_FakeResult([])])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        assert report.status == HALT_INCOMPLETE
        assert report.has_results_doc is False
        assert report.has_transcript is False


# ─────────────────────────────────────────────────────────────────
# Report dict serialisation
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestReportSerialisation:
    async def test_completeness_report_round_trips_to_dict(self):
        current = [
            _mock_metric("EPS"),
            _mock_metric("Margin"),
            _mock_metric("GUIDANCE: X", segment="guidance"),
        ]
        db = _make_db([
            _FakeResult(current),
            _FakeResult([], scalar_value=1),
            _FakeResult([]),
            _FakeResult([]),
            _FakeResult([]),
        ])
        report = await compute_completeness(db, "co-uuid", "2026_Q1")
        d = report.to_dict()
        assert d["status"] in (PROCEED, PROCEED_WITH_CAVEATS, HALT_INCOMPLETE)
        assert "required" in d
        assert "recommended" in d
        assert "missing_required" in d
        assert "reason" in d
        assert "checked_at" in d

    async def test_source_coverage_report_to_dict(self):
        db = _make_db([_FakeResult([("earnings_release",), ("transcript",)])])
        report = await compute_source_coverage(db, "co-uuid", "2026_Q1")
        d = report.to_dict()
        assert d["has_results_doc"] is True
        assert d["has_transcript"] is True
        assert isinstance(d["missing_recommended"], list)
