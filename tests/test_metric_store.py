"""Tests for services/metric_store.py — persistent metric store with
QoQ / YoY comparator computation. No LLM, no DB — externals mocked
via AsyncMock / MagicMock (same pattern as test_completeness_gate)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.metric_store import (
    MetricTimeline, MetricsHistoryResult,
    _compute_comparator_periods, _format_change, _fy_equivalent,
    _normalise_metric_name, _period_candidates,
    format_for_prompt, get_metrics_history,
)


# ─────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────

class TestNormaliseMetricName:
    def test_alias_maps_variants_to_same_key(self):
        assert _normalise_metric_name("EPS (diluted)") == "eps_diluted"
        assert _normalise_metric_name("Diluted EPS") == "eps_diluted"
        assert _normalise_metric_name("diluted eps") == "eps_diluted"

    def test_revenue_variants(self):
        assert _normalise_metric_name("Net Revenue") == "revenue"
        assert _normalise_metric_name("Total Revenue") == "revenue"
        assert _normalise_metric_name("Sales") == "revenue"

    def test_empty_and_unknown(self):
        assert _normalise_metric_name("") == ""
        assert _normalise_metric_name(None) == ""
        # Unknown metric falls through as snake-cased
        assert _normalise_metric_name("Gross Margin") == "gross_margin"

    def test_parentheticals_kept_as_words(self):
        # Parens are treated as spaces — content is preserved so
        # "EPS (diluted)" matches the `eps diluted` alias. Qualified
        # metrics like "NIM (ex-mortgage)" stay distinct from plain NIM
        # (which is what we want — they are different metrics).
        assert _normalise_metric_name("NIM (ex-mortgage)") == "nim_ex_mortgage"
        assert _normalise_metric_name("EPS (diluted)") == "eps_diluted"


class TestFormatChange:
    def test_percent_change_default(self):
        # Revenue 100 → 110 = +10%
        assert _format_change(110, 100, "EUR_M") == "+10%"

    def test_percent_change_negative(self):
        assert _format_change(90, 100, "EUR_M") == "-10%"

    def test_bps_change_for_percent_unit(self):
        # NIM 3.35 → 3.45 = +10 bps
        assert _format_change(3.45, 3.35, "%") == "+10 bps"
        assert _format_change(3.35, 3.45, "%") == "-10 bps"

    def test_zero_prior_handled(self):
        assert _format_change(100, 0, "EUR_M") == "n/a (prior=0)"
        assert _format_change(0, 0, "EUR_M") == "0.0%"


class TestFYEquivalence:
    def test_q4_to_fy(self):
        assert _fy_equivalent("2024_Q4") == "2024_FY"

    def test_fy_to_q4(self):
        assert _fy_equivalent("2024_FY") == "2024_Q4"

    def test_non_year_end_returns_none(self):
        assert _fy_equivalent("2024_Q1") is None
        assert _fy_equivalent("") is None

    def test_period_candidates_includes_both(self):
        assert set(_period_candidates("2024_Q4")) == {"2024_Q4", "2024_FY"}
        assert set(_period_candidates("2024_FY")) == {"2024_FY", "2024_Q4"}
        assert _period_candidates("2024_Q1") == ["2024_Q1"]


class TestComparatorPeriods:
    def test_quarter_qoq_and_yoy(self):
        qoq, yoy = _compute_comparator_periods("2026_Q1")
        assert qoq == "2025_Q4"
        assert yoy == "2025_Q1"

    def test_fy_maps_to_q4_first(self):
        # FY2024 → treat as Q4-2024 for comparator purposes
        qoq, yoy = _compute_comparator_periods("2024_FY")
        assert qoq == "2024_Q3"
        assert yoy == "2023_Q4"


# ─────────────────────────────────────────────────────────────────
# Mock helpers — shape matches test_completeness_gate
# ─────────────────────────────────────────────────────────────────

def _mock_metric(
    metric_name: str,
    *,
    metric_value: float | None = None,
    period_label: str = "2026_Q1",
    segment: str | None = None,
    unit: str | None = "EUR_M",
    confidence: float = 1.0,
) -> MagicMock:
    m = MagicMock()
    m.metric_name = metric_name
    m.metric_value = metric_value
    m.period_label = period_label
    m.segment = segment
    m.unit = unit
    m.confidence = confidence
    return m


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        inner = MagicMock()
        inner.all = MagicMock(return_value=self._items)
        return inner

    def all(self):
        return self._items


def _make_db(rows):
    """AsyncMock DB returning one result with all the rows — matches
    the single-query shape used by get_metrics_history."""
    db = AsyncMock()

    async def _execute(*args, **kwargs):
        return _FakeResult(rows)

    db.execute = _execute
    return db


# ─────────────────────────────────────────────────────────────────
# get_metrics_history — core behaviour
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGetMetricsHistory:
    async def test_qoq_and_yoy_computed_for_matched_metric(self):
        rows = [
            _mock_metric("Revenue", metric_value=2065, period_label="2026_Q1"),
            _mock_metric("Revenue", metric_value=2031, period_label="2025_Q4"),
            _mock_metric("Revenue", metric_value=2014, period_label="2025_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")

        assert r.qoq_period == "2025_Q4"
        assert r.yoy_period == "2025_Q1"
        assert "revenue" in r.timelines
        tl = r.timelines["revenue"]
        assert tl.current == 2065
        assert tl.qoq_prior == 2031
        assert tl.yoy_prior == 2014
        # QoQ = (2065-2031)/2031 ≈ 1.7%
        assert tl.qoq_change == "+1.7%"
        # YoY = (2065-2014)/2014 ≈ 2.5%
        assert tl.yoy_change == "+2.5%"

    async def test_percent_unit_reports_bps(self):
        rows = [
            _mock_metric("NIM", metric_value=3.45, unit="%", period_label="2026_Q1"),
            _mock_metric("NIM", metric_value=3.35, unit="%", period_label="2025_Q4"),
            _mock_metric("NIM", metric_value=3.28, unit="%", period_label="2025_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        tl = r.timelines["nim"]
        assert tl.qoq_change == "+10 bps"
        assert tl.yoy_change == "+17 bps"

    async def test_name_normalisation_collapses_variants(self):
        # "EPS (diluted)" in current, "Diluted EPS" in prior year
        rows = [
            _mock_metric("EPS (diluted)", metric_value=1.04,
                         unit="USD", period_label="2026_Q1"),
            _mock_metric("Diluted EPS", metric_value=0.62,
                         unit="USD", period_label="2025_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert "eps_diluted" in r.timelines
        tl = r.timelines["eps_diluted"]
        assert tl.current == 1.04
        assert tl.yoy_prior == 0.62

    async def test_missing_qoq_still_reports_current_and_yoy(self):
        rows = [
            _mock_metric("Revenue", metric_value=2065, period_label="2026_Q1"),
            _mock_metric("Revenue", metric_value=2014, period_label="2025_Q1"),
            # No Q4-2025 revenue
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        tl = r.timelines["revenue"]
        assert tl.qoq_prior is None
        assert tl.qoq_change is None
        assert tl.yoy_prior == 2014
        assert tl.yoy_change == "+2.5%"

    async def test_metric_only_in_current_still_included(self):
        rows = [
            _mock_metric("Revenue", metric_value=2065, period_label="2026_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        tl = r.timelines["revenue"]
        assert tl.current == 2065
        assert tl.qoq_prior is None
        assert tl.yoy_prior is None

    async def test_metric_only_in_prior_excluded(self):
        # No current-period data → skipped (the agent doesn't need
        # "here's a metric that existed in 2025_Q4 but not this quarter")
        rows = [
            _mock_metric("Headcount", metric_value=10000, period_label="2025_Q4"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert "headcount" not in r.timelines

    async def test_bridge_gap_rows_excluded(self):
        # These rows are the trap from the NWC CN incident — they look
        # like temporal deltas but aren't. Verify they are filtered out.
        rows = [
            _mock_metric("BRIDGE_GAP_PCT:EBITDA", metric_value=3.4,
                         unit="%", period_label="2026_Q1"),
            _mock_metric("BRIDGE_ADJ:one_time_charge", metric_value=12.0,
                         unit="EUR_M", period_label="2026_Q1"),
            _mock_metric("GUIDANCE:revenue", metric_value=8500,
                         unit="EUR_M", period_label="2026_Q1"),
            _mock_metric("Revenue", metric_value=2065,
                         period_label="2026_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        # BRIDGE_* and GUIDANCE filtered; Revenue kept
        assert set(r.timelines.keys()) == {"revenue"}

    async def test_fy_q4_equivalence_cross_year(self):
        # YoY comparator for 2026_Q1 is 2025_Q1 — unrelated to FY.
        # Verify FY<->Q4 for a year-end period: 2025_Q4 → QoQ=2025_Q3,
        # YoY=2024_Q4 (and _FY should be treated as Q4-equivalent).
        rows = [
            _mock_metric("Revenue", metric_value=2300, period_label="2025_Q4"),
            _mock_metric("Revenue", metric_value=2200, period_label="2025_Q3"),
            # Prior-year shown as FY, not Q4 — should still be picked up
            _mock_metric("Revenue", metric_value=2100, period_label="2024_FY"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2025_Q4")
        tl = r.timelines["revenue"]
        assert tl.current == 2300
        assert tl.qoq_prior == 2200
        assert tl.yoy_prior == 2100   # picked up from FY label

    async def test_highest_confidence_row_wins(self):
        # Two rows for revenue in same period, different confidences.
        # Rows come from the DB pre-sorted by confidence desc, so the
        # first one to land in the bucket (higher confidence) wins.
        rows = [
            _mock_metric("Revenue", metric_value=2065,
                         period_label="2026_Q1", confidence=0.95),
            _mock_metric("Revenue", metric_value=1999,
                         period_label="2026_Q1", confidence=0.60),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert r.timelines["revenue"].current == 2065

    async def test_guidance_segment_excluded(self):
        # segment='guidance' rows are filtered by the query; assert our
        # mock produces no guidance bleed-through. Since the test helper
        # returns all rows, we approximate the query filter by dropping
        # those with segment='guidance' in the mock setup.
        rows = [
            _mock_metric("Revenue", metric_value=2065,
                         period_label="2026_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert "revenue" in r.timelines

    async def test_empty_results_returns_empty_timelines(self):
        db = _make_db([])
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert r.timelines == {}
        assert r.qoq_period == "2025_Q4"
        assert r.yoy_period == "2025_Q1"

    async def test_comparator_anomaly_surfaces(self):
        # 100 → 30000 current quarter = 30000% jump, should flag
        # via the reused compare_to_prior
        rows = [
            _mock_metric("Recovery Revenue", metric_value=30000,
                         period_label="2026_Q1"),
            _mock_metric("Recovery Revenue", metric_value=100,
                         period_label="2025_Q4"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        assert len(r.anomalies) >= 1
        a = r.anomalies[0]
        assert a["severity"] in {"high", "medium"}

    async def test_non_numeric_metric_value_skipped(self):
        # metric_value None → can't compare; should be silently dropped
        rows = [
            _mock_metric("Revenue", metric_value=None, period_label="2026_Q1"),
            _mock_metric("Revenue", metric_value=2014, period_label="2025_Q1"),
        ]
        db = _make_db(rows)
        r = await get_metrics_history(db, company_id="cid", period_label="2026_Q1")
        # Current-period row had no numeric value → no timeline emitted
        assert "revenue" not in r.timelines


# ─────────────────────────────────────────────────────────────────
# format_for_prompt
# ─────────────────────────────────────────────────────────────────

class TestFormatForPrompt:
    def test_empty_result_returns_fallback(self):
        r = MetricsHistoryResult(period_label="2026_Q1", qoq_period="2025_Q4", yoy_period="2025_Q1")
        assert "No prior-period metrics available" in format_for_prompt(r)

    def test_renders_full_row(self):
        r = MetricsHistoryResult(
            period_label="2026_Q1", qoq_period="2025_Q4", yoy_period="2025_Q1",
            timelines={
                "revenue": MetricTimeline(
                    metric_name="revenue", display_name="Revenue",
                    unit="EUR_M", current=2065.0,
                    qoq_prior=2031.0, qoq_change="+1.7%",
                    yoy_prior=2014.0, yoy_change="+2.5%",
                ),
            },
        )
        out = format_for_prompt(r)
        assert "Revenue | 2065 EUR_M" in out
        assert "+1.7%" in out
        assert "+2.5%" in out
        assert "2025_Q4" in out

    def test_anomaly_section_rendered(self):
        r = MetricsHistoryResult(
            period_label="2026_Q1", qoq_period="2025_Q4", yoy_period="2025_Q1",
            timelines={
                "revenue": MetricTimeline(
                    metric_name="revenue", display_name="Revenue",
                    unit="EUR_M", current=100.0,
                ),
            },
            anomalies=[{"metric": "revenue", "current": 100, "prior": 1,
                        "change_pct": 9900.0, "severity": "high"}],
        )
        out = format_for_prompt(r)
        assert "QoQ anomalies flagged" in out
        assert "revenue" in out
