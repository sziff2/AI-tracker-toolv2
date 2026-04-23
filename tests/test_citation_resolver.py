"""Tests for services/citation_resolver — Tier 4.4.

Uses fake AsyncSession so we don't need Postgres."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from services.citation_resolver import (
    resolve_citations,
    VALID_KINDS,
    _resolve_extracted_metric,
    _resolve_document_section,
    CitationResult,
)


# ─────────────────────────────────────────────────────────────────
# Pure helpers — no DB
# ─────────────────────────────────────────────────────────────────

def test_extracted_metric_exact_match():
    names = {("combined ratio (p&c insurance)", "p&c insurance")}
    r = _resolve_extracted_metric(
        {"kind": "extracted_metric",
         "metric_name": "Combined Ratio (P&C Insurance)",
         "segment": "P&C Insurance"},
        names,
    )
    assert r.resolved is True
    assert "exact" in r.reason


def test_extracted_metric_segment_mismatch_still_resolves():
    """Metric exists under a different segment — soft-match allowed."""
    names = {("net premiums written", "north america personal p&c")}
    r = _resolve_extracted_metric(
        {"kind": "extracted_metric",
         "metric_name": "Net Premiums Written",
         "segment": "North America Commercial"},  # wrong segment
        names,
    )
    assert r.resolved is True
    assert "segment differs" in r.reason


def test_extracted_metric_no_match_with_partial_hit():
    """Partial-string match reported as NOT resolved + flagged as likely mis-cite."""
    names = {("combined ratio (p&c insurance)", "p&c insurance")}
    r = _resolve_extracted_metric(
        {"kind": "extracted_metric",
         "metric_name": "combined ratio"},
        names,
    )
    assert r.resolved is False
    assert "partial matches" in r.reason


def test_extracted_metric_truly_missing():
    names = {("revenue", "group")}
    r = _resolve_extracted_metric(
        {"kind": "extracted_metric",
         "metric_name": "Loss Ratio"},
        names,
    )
    assert r.resolved is False
    assert "no metric named" in r.reason


def test_extracted_metric_missing_name_is_unresolved():
    r = _resolve_extracted_metric(
        {"kind": "extracted_metric", "segment": "x"},
        set(),
    )
    assert r.resolved is False
    assert "missing metric_name" in r.reason


# ─────────────────────────────────────────────────────────────────
# document_section
# ─────────────────────────────────────────────────────────────────

def test_document_section_doc_id_resolves():
    doc_ids = {"abc-123", "def-456"}
    r = _resolve_document_section(
        {"kind": "document_section", "doc_id": "abc-123", "page": 5},
        doc_ids,
    )
    assert r.resolved is True
    assert r.doc_id == "abc-123"


def test_document_section_bad_doc_id_unresolved():
    r = _resolve_document_section(
        {"kind": "document_section", "doc_id": "not-a-real-doc"},
        {"abc-123"},
    )
    assert r.resolved is False
    assert "not in this period" in r.reason


def test_document_section_snippet_only_resolves_shape_only():
    r = _resolve_document_section(
        {"kind": "document_section", "snippet": "Combined ratio worsened to 84.0% from 82.1%"},
        set(),
    )
    assert r.resolved is True
    assert "shape-only" in r.reason


def test_document_section_no_doc_id_no_snippet_unresolved():
    r = _resolve_document_section(
        {"kind": "document_section"},
        set(),
    )
    assert r.resolved is False
    assert "needs either doc_id or snippet" in r.reason


# ─────────────────────────────────────────────────────────────────
# Fake AsyncSession for the full path
# ─────────────────────────────────────────────────────────────────

@dataclass
class _Row:
    metric_name: str | None = None
    segment: str | None = None
    doc_id: str | None = None


class _FakeResult:
    def __init__(self, rows: list):
        self._rows = rows

    def all(self):
        return [(r.metric_name, r.segment) if r.doc_id is None
                else (r.doc_id,) for r in self._rows]


class _FakeSession:
    """Returns (metric_name, segment) rows on first execute, doc_id rows on second.

    resolve_citations calls _load_metric_name_set first, then _load_doc_id_set."""
    def __init__(self, metrics: list[_Row], docs: list[_Row]):
        self._queue = [_FakeResult(metrics), _FakeResult(docs)]

    async def execute(self, *_args, **_kwargs):
        return self._queue.pop(0)


def test_resolve_citations_full_report():
    sources = [
        {"kind": "extracted_metric", "metric_name": "Combined Ratio (P&C Insurance)",
         "segment": "P&C Insurance"},
        {"kind": "extracted_metric", "metric_name": "Loss Ratio"},  # missing
        {"kind": "document_section", "doc_id": "abc-123", "page": 4},
        {"kind": "document_section", "snippet": "Greenberg used the word 'dumb'"},
        {"kind": "management_quote",
         "snippet": "policy rates were expected to go down further in UK and in Norway"},
        {"kind": "methodology_note"},  # missing snippet → unresolved
        {"kind": "unknown_kind"},  # malformed
        "not-a-dict",  # malformed
    ]
    session = _FakeSession(
        metrics=[
            _Row(metric_name="Combined Ratio (P&C Insurance)", segment="P&C Insurance"),
            _Row(metric_name="Net Premiums Written", segment="NA Commercial"),
        ],
        docs=[_Row(doc_id="abc-123"), _Row(doc_id="def-456")],
    )
    report = asyncio.run(resolve_citations(session, "company-x", "2026_Q1", sources))
    assert report.total == 8
    # expected resolved: extracted_metric[0], document_section[0], document_section[1], management_quote[0]
    assert report.resolved == 4
    # expected unresolved: extracted_metric[1] (missing), methodology_note[0] (no snippet)
    assert report.unresolved == 2
    # expected malformed: unknown_kind, non-dict
    assert report.malformed == 2
    assert 0.49 < report.resolve_rate < 0.51  # 4/8 = 0.5


def test_resolve_citations_empty_input_vacuous_pass():
    report = asyncio.run(resolve_citations(_FakeSession([], []), "c", "p", None))
    assert report.total == 0
    assert report.resolve_rate == 1.0

    report2 = asyncio.run(resolve_citations(_FakeSession([], []), "c", "p", []))
    assert report2.total == 0


def test_valid_kinds_covers_expected_set():
    for k in ("extracted_metric", "document_section", "management_quote",
              "methodology_note", "bridge_gap", "historical_drawdown", "thesis"):
        assert k in VALID_KINDS
