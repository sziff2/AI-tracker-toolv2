"""Tests for the pipeline-run diff helpers (Tier 3.2).

The endpoint integration is exercised by hand — these focus on the
pure-function diff logic so the serialisation / metadata rules are
locked down.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.api.routes.pipeline import (
    _json_diff,
    _metadata_diff,
    _ao_summary,
)


# ─────────────────────────────────────────────────────────────────
# _json_diff
# ─────────────────────────────────────────────────────────────────

def test_json_diff_identical_returns_empty_sets():
    d = _json_diff({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert d["changed"] == set()
    assert d["added"] == set()
    assert d["removed"] == set()


def test_json_diff_changed_value_flagged():
    d = _json_diff({"a": 1}, {"a": 2})
    assert d["changed"] == {"a"}
    assert d["added"] == set()
    assert d["removed"] == set()


def test_json_diff_added_and_removed_keys():
    d = _json_diff({"a": 1, "removed": True}, {"a": 1, "added": True})
    assert d["changed"] == set()
    assert d["added"] == {"added"}
    assert d["removed"] == {"removed"}


def test_json_diff_nested_structure_change_shows_as_top_level():
    a = {"scenario": {"bear": 0.3, "base": 0.4, "bull": 0.3}}
    b = {"scenario": {"bear": 0.4, "base": 0.4, "bull": 0.2}}
    d = _json_diff(a, b)
    assert d["changed"] == {"scenario"}


def test_json_diff_list_key_order_sensitivity():
    """List order matters — different order = different output."""
    d = _json_diff({"risks": ["a", "b"]}, {"risks": ["b", "a"]})
    assert "risks" in d["changed"]


def test_json_diff_none_vs_dict_is_root_change():
    d = _json_diff(None, {"a": 1})
    assert d["changed"] == {"_root"}


# ─────────────────────────────────────────────────────────────────
# _metadata_diff
# ─────────────────────────────────────────────────────────────────

def _ao(**kw) -> SimpleNamespace:
    """Helper to build an AgentOutput-like namespace."""
    defaults = dict(
        id="00000000-0000-0000-0000-000000000001",
        status="completed",
        confidence=0.8,
        qc_score=0.9,
        duration_ms=1000,
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.01,
        prompt_variant_id=None,
        output_json={"k": "v"},
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_metadata_diff_identical_is_empty():
    a = _ao()
    b = _ao()
    assert _metadata_diff(a, b) == {}


def test_metadata_diff_flags_variant_change():
    a = _ao(prompt_variant_id="v1")
    b = _ao(prompt_variant_id="v2")
    d = _metadata_diff(a, b)
    assert d["prompt_variant_id"] == {"base": "v1", "compare": "v2"}


def test_metadata_diff_flags_big_cost_jump():
    """Cost change >20% should flag."""
    a = _ao(cost_usd=0.01)
    b = _ao(cost_usd=0.10)
    d = _metadata_diff(a, b)
    assert "cost_usd" in d
    assert d["cost_usd"]["delta"] == 0.09


def test_metadata_diff_ignores_tiny_cost_drift():
    """Cost 0.010 → 0.011 is noise, shouldn't flag."""
    a = _ao(cost_usd=0.010)
    b = _ao(cost_usd=0.011)
    d = _metadata_diff(a, b)
    assert "cost_usd" not in d


def test_metadata_diff_flags_status_change():
    a = _ao(status="completed")
    b = _ao(status="failed")
    d = _metadata_diff(a, b)
    assert d["status"] == {"base": "completed", "compare": "failed"}


# ─────────────────────────────────────────────────────────────────
# _ao_summary
# ─────────────────────────────────────────────────────────────────

def test_ao_summary_roundtrip():
    a = _ao(prompt_variant_id="v1", output_json={"foo": [1, 2, 3]})
    s = _ao_summary(a)
    assert s["id"] == a.id
    assert s["status"] == "completed"
    assert s["confidence"] == 0.8
    assert s["prompt_variant_id"] == "v1"
    assert s["output_json"] == {"foo": [1, 2, 3]}
