"""Tests for services/methodology_tracker — deterministic bridge-diff +
restatement detection. Fake AsyncSession returns hand-crafted rows so
we don't need Postgres."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from services.methodology_tracker import (
    MethodologyFlag,
    compute_methodology_report,
    detect_metric_restatements,
    _canonical_label,
    _diff_labels,
    _extract_labels_and_gap,
    _gap_drift_flag,
)


# ─────────────────────────────────────────────────────────────────
# Unit tests — pure helpers
# ─────────────────────────────────────────────────────────────────

def test_canonical_label_strips_filler():
    assert _canonical_label("Restructuring Charges") == _canonical_label("Restructuring costs")
    assert _canonical_label("Restructuring Charges (Non-recurring)") == _canonical_label("Restructuring")


def test_canonical_label_empty_inputs():
    assert _canonical_label("") == ""
    assert _canonical_label("costs") == "costs"  # all-filler stays as single word


def test_diff_labels_detects_new_and_removed():
    current = ["Restructuring charges", "Impairment", "Strategic repositioning charges"]
    prior   = ["Restructuring costs", "Impairment of goodwill", "M&A transaction fees"]

    new, removed = _diff_labels(current, prior)
    new_canon = [_canonical_label(l) for l in new]
    removed_canon = [_canonical_label(l) for l in removed]
    # Strategic repositioning is new (no prior match)
    assert any("strategic" in c or "repositioning" in c for c in new_canon)
    # M&A transaction fees gone (no current match)
    assert any("transaction" in c or "fees" in c for c in removed_canon)


def test_diff_labels_fuzzy_match_treats_as_same():
    current = ["Restructuring charges"]
    prior   = ["Restructuring costs"]
    new, removed = _diff_labels(current, prior)
    assert new == []
    assert removed == []


def test_gap_drift_ignores_tiny_absolute_moves():
    """0.3pp movement below the absolute threshold should not flag."""
    flag = _gap_drift_flag(3.0, 3.3, "2026_Q1", "2025_Q4")
    assert flag is None


def test_gap_drift_ignores_small_relative_moves():
    """10% rel change is below the 25% rel threshold."""
    flag = _gap_drift_flag(11.0, 10.0, "2026_Q1", "2025_Q4")
    assert flag is None


def test_gap_drift_flags_material_widening():
    flag = _gap_drift_flag(8.0, 3.0, "2026_Q1", "2025_Q4")
    assert flag is not None
    assert flag.kind == "gap_drift"
    assert flag.severity == "warning"
    assert "widened" in flag.message


def test_gap_drift_narrowing_is_info_not_warning():
    flag = _gap_drift_flag(1.5, 5.0, "2026_Q1", "2025_Q4")
    assert flag is not None
    assert flag.severity == "info"
    assert "narrowed" in flag.message


def test_gap_drift_none_when_either_missing():
    assert _gap_drift_flag(None, 5.0, "a", "b") is None
    assert _gap_drift_flag(5.0, None, "a", "b") is None


# ─────────────────────────────────────────────────────────────────
# Fake DB session
# ─────────────────────────────────────────────────────────────────

@dataclass
class _FakeMetric:
    metric_name: str
    metric_value: Any
    period_label: str
    segment: str = "non_gaap_bridge"
    document_id: str = "doc-0"


@dataclass
class _JoinRow:
    metric_name: str
    period_label: str
    metric_value: float
    document_id: str
    source_period: str
    published_at: datetime


class _FakeScalars:
    def __init__(self, items): self._items = items
    def all(self): return self._items


class _FakeResult:
    def __init__(self, items, is_join=False):
        self._items = items
        self._is_join = is_join
    def scalars(self): return _FakeScalars(self._items)
    def all(self): return self._items


class _FakeSession:
    """Queue of results (one per execute() call, in order)."""
    def __init__(self, results: list):
        self._results = list(results)

    async def execute(self, *_args, **_kwargs):
        if not self._results:
            return _FakeResult([])
        return self._results.pop(0)


def test_extract_labels_picks_largest_gap_pct():
    rows = [
        _FakeMetric("BRIDGE_GAP_PCT:EBITDA", 3.4, "2026_Q1"),
        _FakeMetric("BRIDGE_GAP_PCT:EPS", 0.2, "2026_Q1"),
        _FakeMetric("BRIDGE_ADJ:Restructuring", 12.0, "2026_Q1"),
        _FakeMetric("BRIDGE_ADJ:Impairment", 5.0, "2026_Q1"),
    ]
    labels, gap = _extract_labels_and_gap(rows)
    assert gap == 3.4
    assert set(labels) == {"Restructuring", "Impairment"}


def test_compute_methodology_report_flags_new_adjustment():
    current = [
        _FakeMetric("BRIDGE_GAP_PCT:EBITDA", 4.0, "2026_Q1"),
        _FakeMetric("BRIDGE_ADJ:Restructuring charges", 10.0, "2026_Q1"),
        _FakeMetric("BRIDGE_ADJ:Strategic repositioning programme", 20.0, "2026_Q1"),
    ]
    prior = [
        _FakeMetric("BRIDGE_GAP_PCT:EBITDA", 3.8, "2025_Q4"),
        _FakeMetric("BRIDGE_ADJ:Restructuring costs", 9.5, "2025_Q4"),
    ]

    session = _FakeSession([
        _FakeResult(current),  # current bridge rows
        _FakeResult(prior),    # prior bridge rows
        _FakeResult([], is_join=True),  # restatement join — empty
    ])

    report = asyncio.run(compute_methodology_report(session, "X", "2026_Q1"))
    kinds = [f["kind"] for f in report.flags]
    assert "new_adjustment" in kinds
    # Gap moved 3.8 → 4.0, only 0.2pp — should NOT flag drift
    assert "gap_drift" not in kinds


def test_compute_methodology_report_flags_gap_widening():
    current = [_FakeMetric("BRIDGE_GAP_PCT:EBITDA", 12.0, "2026_Q1")]
    prior   = [_FakeMetric("BRIDGE_GAP_PCT:EBITDA", 3.0, "2025_Q4")]

    session = _FakeSession([
        _FakeResult(current),
        _FakeResult(prior),
        _FakeResult([], is_join=True),
    ])

    report = asyncio.run(compute_methodology_report(session, "X", "2026_Q1"))
    kinds = [f["kind"] for f in report.flags]
    assert "gap_drift" in kinds
    drift = next(f for f in report.flags if f["kind"] == "gap_drift")
    assert drift["severity"] == "warning"


def test_detect_restatement_flags_ally_style_case():
    """Core ROTCE reported as 15.3% in 2025_Q2 filing, then re-reported
    as 12.3% in the 2025_Q3 filing (comparator column). Restatement."""
    pub_q2 = datetime(2025, 8, 1, tzinfo=timezone.utc)
    pub_q3 = datetime(2025, 11, 1, tzinfo=timezone.utc)
    join_rows = [
        # First filing (2025_Q2) reports Q2 value
        _JoinRow("Core ROTCE", "2025_Q2", 15.3, "d1", "2025_Q2", pub_q2),
        # Second filing (2025_Q3) restates Q2 value
        _JoinRow("Core ROTCE", "2025_Q2", 12.3, "d2", "2025_Q3", pub_q3),
    ]
    session = _FakeSession([_FakeResult(join_rows, is_join=True)])
    flags = asyncio.run(detect_metric_restatements(session, "X", "2025_Q3"))
    assert len(flags) == 1
    f = flags[0]
    assert f.kind == "restatement"
    assert f.severity == "warning"
    assert f.prior_period == "2025_Q2"
    assert f.prior_value == 15.3
    assert f.current_value == 12.3
    assert "restated" in f.message.lower()


def test_detect_restatement_tolerates_rounding():
    pub_q2 = datetime(2025, 8, 1, tzinfo=timezone.utc)
    pub_q3 = datetime(2025, 11, 1, tzinfo=timezone.utc)
    # 4.12 vs 4.13 — within 1% tolerance
    join_rows = [
        _JoinRow("NIM", "2025_Q2", 4.12, "d1", "2025_Q2", pub_q2),
        _JoinRow("NIM", "2025_Q2", 4.13, "d2", "2025_Q3", pub_q3),
    ]
    session = _FakeSession([_FakeResult(join_rows, is_join=True)])
    flags = asyncio.run(detect_metric_restatements(session, "X", "2025_Q3"))
    assert flags == []


def test_detect_restatement_ignores_current_period_own_value():
    """A metric for 2025_Q3 extracted from the 2025_Q3 filing — that's
    not a restatement, it's just the current measurement."""
    pub_q3 = datetime(2025, 11, 1, tzinfo=timezone.utc)
    join_rows = [
        _JoinRow("EPS", "2025_Q3", 1.04, "d1", "2025_Q3", pub_q3),
    ]
    session = _FakeSession([_FakeResult(join_rows, is_join=True)])
    flags = asyncio.run(detect_metric_restatements(session, "X", "2025_Q3"))
    assert flags == []


def test_detect_restatement_ignores_bridge_rows():
    pub_q2 = datetime(2025, 8, 1, tzinfo=timezone.utc)
    pub_q3 = datetime(2025, 11, 1, tzinfo=timezone.utc)
    join_rows = [
        _JoinRow("BRIDGE_GAP_PCT:EBITDA", "2025_Q2", 3.4, "d1", "2025_Q2", pub_q2),
        _JoinRow("BRIDGE_GAP_PCT:EBITDA", "2025_Q2", 5.1, "d2", "2025_Q3", pub_q3),
    ]
    session = _FakeSession([_FakeResult(join_rows, is_join=True)])
    flags = asyncio.run(detect_metric_restatements(session, "X", "2025_Q3"))
    # Bridge rows are out-of-scope for restatement detection (diffed elsewhere)
    assert flags == []
