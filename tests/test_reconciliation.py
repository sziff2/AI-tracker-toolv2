"""Tests for services/reconciliation.py — the third pre-flight gate.

Same mock pattern as test_completeness_gate: AsyncMock DB returning
scripted FakeResults. No real DB, no LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.reconciliation import (
    HALT_INCOMPLETE, PROCEED, PROCEED_WITH_CAVEATS,
    ReconciliationReport, compute_reconciliation,
    _CROSS_SOURCE_TOLERANCE,
)


# ─────────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────────

def _mock_metric(
    name: str,
    *,
    value: float | None = None,
    period: str = "2026_Q1",
    segment: str | None = None,
    unit: str | None = "EUR_M",
    document_id: str = "doc1",
    qualifier_json: dict | None = None,
) -> MagicMock:
    m = MagicMock()
    m.metric_name = name
    m.metric_value = value
    m.period_label = period
    m.segment = segment
    m.unit = unit
    m.document_id = document_id
    m.qualifier_json = qualifier_json
    return m


def _mock_company(sector: str = "", industry: str = "") -> MagicMock:
    c = MagicMock()
    c.sector = sector
    c.industry = industry
    return c


class _FakeResult:
    """Scalar / scalars shim — same shape as test_completeness_gate."""
    def __init__(self, items, scalar_value=None):
        self._items = items
        self._scalar = scalar_value

    def scalars(self):
        inner = MagicMock()
        inner.all = MagicMock(return_value=self._items)
        return inner

    def scalar_one_or_none(self):
        return self._scalar

    def all(self):
        return self._items


def _make_db(execute_results):
    """Script a sequence of DB responses — each execute() pops the next.

    compute_reconciliation issues roughly 5 queries:
      1. ExtractedMetric for validation
      2. Company (sector/industry)
      3. ExtractionProfile.reconciliation for structural checks
      4. ExtractedMetric for current period (anomaly)
      5. ExtractedMetric for prior period (anomaly)
      6. ExtractedMetric for cross-source checks
    Tests provide results in that order.
    """
    db = AsyncMock()
    iterator = iter(execute_results)

    async def _execute(*args, **kwargs):
        return next(iterator)

    db.execute = _execute
    return db


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestComputeReconciliation:
    async def test_clean_run_proceeds(self):
        # Validation: one normal metric, no issues
        # Company: generic sector (no sector rules fire)
        # Structural: nothing persisted
        # Anomaly: same value QoQ, no anomaly
        # Cross-source: single document
        db = _make_db([
            _FakeResult([_mock_metric("Revenue", value=2000)]),       # validation
            _FakeResult([], scalar_value=_mock_company()),            # company
            _FakeResult([], scalar_value=None),                       # structural
            _FakeResult([_mock_metric("Revenue", value=2000)]),       # anomaly current
            _FakeResult([_mock_metric("Revenue", value=1990,
                                      period="2025_Q4")]),            # anomaly prior
            _FakeResult([_mock_metric("Revenue", value=2000)]),       # cross-source
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.status == PROCEED
        assert r.critical_count == 0

    async def test_out_of_range_flags_critical_then_halts(self):
        # Operating Margin = 500% → critical out-of-range
        db = _make_db([
            _FakeResult([_mock_metric("Operating Margin",
                                      value=500.0, unit="%")]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("Operating Margin", value=500.0, unit="%")]),
            _FakeResult([_mock_metric("Operating Margin",
                                      value=15.0, unit="%",
                                      period="2025_Q4")]),
            _FakeResult([_mock_metric("Operating Margin",
                                      value=500.0, unit="%")]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.status == HALT_INCOMPLETE
        assert r.critical_count >= 1

    async def test_sector_specific_flag(self):
        # NIM 12% is impossible for a bank — sector rule catches it
        db = _make_db([
            _FakeResult([_mock_metric("NIM", value=12.0, unit="%")]),
            _FakeResult([], scalar_value=_mock_company(
                sector="Financials", industry="Banks")),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("NIM", value=12.0, unit="%")]),
            _FakeResult([_mock_metric("NIM", value=3.3, unit="%",
                                      period="2025_Q4")]),
            _FakeResult([_mock_metric("NIM", value=12.0, unit="%")]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.status == HALT_INCOMPLETE
        # At least one critical
        assert r.critical_count >= 1
        # Validation issues should name NIM
        assert any("NIM" in vi["metric_name"] for vi in r.validation_issues)

    async def test_bridge_gap_rows_ignored(self):
        # BRIDGE_GAP rows bypass validation entirely — they are not
        # real metric values
        db = _make_db([
            _FakeResult([
                _mock_metric("BRIDGE_GAP_PCT:EBITDA", value=999, unit="%"),
                _mock_metric("Revenue", value=1000),
            ]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("Revenue", value=1000)]),
            _FakeResult([_mock_metric("Revenue", value=950, period="2025_Q4")]),
            _FakeResult([_mock_metric("Revenue", value=1000)]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        # 999% would absolutely trigger a universal range rule if not filtered
        # → confirm zero critical issues
        assert r.critical_count == 0

    async def test_cross_source_disagreement_warns(self):
        # Same metric (Revenue) extracted from two documents with
        # values 1000 vs 1100 → 10% spread, above 5% → warning
        db = _make_db([
            _FakeResult([_mock_metric("Revenue", value=1000, document_id="d1")]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("Revenue", value=1000, document_id="d1")]),
            _FakeResult([_mock_metric("Revenue", value=1000,
                                      period="2025_Q4", document_id="d1")]),
            # Cross-source query sees both docs
            _FakeResult([
                _mock_metric("Revenue", value=1000, document_id="d1"),
                _mock_metric("Revenue", value=1100, document_id="d2"),
            ]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.cross_source_checks.get("disagreements")
        assert r.status == PROCEED_WITH_CAVEATS
        assert r.warning_count >= 1

    async def test_cross_source_within_tolerance_passes(self):
        # 3.45 vs 3.4 = ~1.45% spread — below 2% tolerance
        db = _make_db([
            _FakeResult([_mock_metric("NIM", value=3.45, unit="%", document_id="d1")]),
            _FakeResult([], scalar_value=_mock_company(
                sector="Financials", industry="Banks")),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("NIM", value=3.45, unit="%", document_id="d1")]),
            _FakeResult([_mock_metric("NIM", value=3.30, unit="%",
                                      period="2025_Q4", document_id="d1")]),
            _FakeResult([
                _mock_metric("NIM", value=3.45, unit="%", document_id="d1"),
                _mock_metric("NIM", value=3.40, unit="%", document_id="d2"),
            ]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.cross_source_checks.get("disagreements") == []

    async def test_anomaly_high_severity_warns(self):
        # 1000 → 30000 in one quarter = 2900% jump → high severity
        db = _make_db([
            _FakeResult([_mock_metric("Recovery Revenue", value=30000)]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=None),
            _FakeResult([_mock_metric("Recovery Revenue", value=30000)]),
            _FakeResult([_mock_metric("Recovery Revenue",
                                      value=1000, period="2025_Q4")]),
            _FakeResult([_mock_metric("Recovery Revenue", value=30000)]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        anomalies = r.anomaly_checks.get("anomalies", [])
        assert len(anomalies) >= 1
        assert r.status == PROCEED_WITH_CAVEATS

    async def test_structural_reconciliation_loaded_and_flagged(self):
        # ExtractionProfile.reconciliation has a critical structural issue
        structural = {
            "passed": False,
            "issues": [
                {"check": "q_sum_vs_fy", "severity": "critical",
                 "message": "Q1+Q2+Q3+Q4 ≠ FY Revenue (spread 4.2%)"},
            ],
        }
        db = _make_db([
            _FakeResult([_mock_metric("Revenue", value=2000)]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=structural),
            _FakeResult([_mock_metric("Revenue", value=2000)]),
            _FakeResult([_mock_metric("Revenue", value=1990, period="2025_Q4")]),
            _FakeResult([_mock_metric("Revenue", value=2000)]),
        ])
        r = await compute_reconciliation(db, "cid", "2026_Q1")
        assert r.status == HALT_INCOMPLETE
        assert r.critical_count >= 1
        # Issue should be in the flat list
        assert any(i["source"] == "structural" for i in r.issues)

    async def test_fy_period_has_no_qoq_but_still_runs(self):
        # Period is FY — no QoQ comparator, anomaly_checks returns note
        db = _make_db([
            _FakeResult([_mock_metric("Revenue", value=8000, period="2025_FY")]),
            _FakeResult([], scalar_value=_mock_company()),
            _FakeResult([], scalar_value=None),
            # (anomaly_checks won't run query if qoq_period is None)
            _FakeResult([_mock_metric("Revenue", value=8000, period="2025_FY")]),
        ])
        r = await compute_reconciliation(db, "cid", "2025_FY")
        # Should complete without error
        assert r.status in {PROCEED, PROCEED_WITH_CAVEATS, HALT_INCOMPLETE}
        assert "No QoQ prior period" in r.anomaly_checks.get("note", "")


# ─────────────────────────────────────────────────────────────────
# Report shape
# ─────────────────────────────────────────────────────────────────

class TestReconciliationReport:
    def test_to_dict_round_trips(self):
        r = ReconciliationReport(
            status=PROCEED, critical_count=0, warning_count=0,
            info_count=0, reason="ok", checked_at="2026-04-18T00:00:00",
        )
        d = r.to_dict()
        assert d["status"] == PROCEED
        assert d["critical_count"] == 0
        assert d["issues"] == []
