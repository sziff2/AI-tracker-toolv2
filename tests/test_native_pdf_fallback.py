"""Tests for services/native_pdf_fallback — pure-function helpers +
shape conversion. The Anthropic call itself is not exercised here (no
mocked SDK); that's covered by the Sprint C-prep A/B harness."""

from __future__ import annotations

import pytest

from services.native_pdf_fallback import (
    _cell_str,
    _safe_parse_json,
    tables_to_baseline_shape,
    DEFAULT_FALLBACK_DOC_TYPES,
)


# ─────────────────────────────────────────────────────────────────
# _cell_str
# ─────────────────────────────────────────────────────────────────

def test_cell_str_none_becomes_empty():
    assert _cell_str(None) == ""


def test_cell_str_collapses_whitespace():
    assert _cell_str("  hello   world  \n\t") == "hello world"


def test_cell_str_coerces_numbers():
    assert _cell_str(1234) == "1234"
    assert _cell_str(1.5) == "1.5"


# ─────────────────────────────────────────────────────────────────
# _safe_parse_json
# ─────────────────────────────────────────────────────────────────

def test_safe_parse_json_bare():
    assert _safe_parse_json('{"tables": []}') == {"tables": []}


def test_safe_parse_json_strips_code_fences():
    raw = '```json\n{"tables": [1, 2, 3]}\n```'
    assert _safe_parse_json(raw) == {"tables": [1, 2, 3]}


def test_safe_parse_json_tolerates_prose_wrap():
    """Claude occasionally opens with 'Here is the JSON:' — we should
    still recover the object between the first { and last }."""
    raw = 'Here is the JSON:\n{"tables": [{"page": 1}]}\nDone.'
    assert _safe_parse_json(raw) == {"tables": [{"page": 1}]}


def test_safe_parse_json_returns_none_on_garbage():
    assert _safe_parse_json("not json at all") is None
    assert _safe_parse_json("") is None
    assert _safe_parse_json("{incomplete") is None


# ─────────────────────────────────────────────────────────────────
# tables_to_baseline_shape
# ─────────────────────────────────────────────────────────────────

def test_baseline_shape_groups_by_page():
    fallback = [
        {"page": 1, "caption": "Balance Sheet", "rows": [["Cash", "100"], ["AR", "50"]]},
        {"page": 1, "caption": "Income Stmt",   "rows": [["Rev", "1000"]]},
        {"page": 3, "caption": "Cash Flow",     "rows": [["OCF", "75"]]},
    ]
    out = tables_to_baseline_shape(fallback)
    pages = [r["page"] for r in out]
    assert pages == [1, 3]
    # Page 1 got two tables
    assert len(out[0]["tables"]) == 2
    # Caption prepended as a single-cell header row
    assert out[0]["tables"][0][0] == ["Balance Sheet"]
    assert out[0]["tables"][0][1] == ["Cash", "100"]


def test_baseline_shape_handles_missing_caption():
    fallback = [{"page": 1, "caption": "", "rows": [["a", "b"]]}]
    out = tables_to_baseline_shape(fallback)
    # No caption row prepended
    assert out[0]["tables"][0] == [["a", "b"]]


def test_baseline_shape_empty_input():
    assert tables_to_baseline_shape([]) == []


# ─────────────────────────────────────────────────────────────────
# Default fallback doc-type set
# ─────────────────────────────────────────────────────────────────

def test_default_fallback_covers_known_failure_classes():
    # NWC is "financial_statements"; US 10-Q tuples are "10-Q"; RNS half-years
    for key in ("financial_statements", "condensed_financials", "10-Q", "10-K",
                "annual_report", "rns", "earnings_release"):
        assert key in DEFAULT_FALLBACK_DOC_TYPES


def test_default_fallback_excludes_transcripts_and_decks():
    # 0 tables on a transcript / deck is expected, not a failure
    for key in ("transcript", "presentation", "deck"):
        assert key not in DEFAULT_FALLBACK_DOC_TYPES
