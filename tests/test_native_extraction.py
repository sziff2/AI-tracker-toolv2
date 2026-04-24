"""Tests for services/native_extraction — Phase 1 of the Tier 1.3
consolidation (single-call Claude extraction replacing section splitter
+ two-pass + statement extractors).

Doesn't hit the LLM — mocks call_llm_native_async. Verifies:
  - JSON parse is tolerant of markdown fences + trailing prose
  - Legacy-shape mapping is complete (all keys downstream code reads)
  - Guidance items are promoted into raw_items with segment='guidance'
  - Empty/failed results still match the contract so fallback works
  - Input routing (native PDF vs text) picks the right path
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.native_extraction import (
    _empty_result,
    _map_to_legacy_shape,
    _parse_json_safe,
    run_native_extraction,
)


# ─────────────────────────────────────────────────────────────────
# _parse_json_safe — defensive parsing
# ─────────────────────────────────────────────────────────────────

def test_parse_json_safe_strips_markdown_fences():
    raw = '```json\n{"foo": 1}\n```'
    assert _parse_json_safe(raw) == {"foo": 1}


def test_parse_json_safe_strips_bare_fences():
    raw = "```\n{\"foo\": 1}\n```"
    assert _parse_json_safe(raw) == {"foo": 1}


def test_parse_json_safe_recovers_from_trailing_prose():
    """Sonnet occasionally appends an explanation after valid JSON. The
    recovery path slices from first `{` to last `}`."""
    raw = '{"foo": 1, "bar": [2, 3]}\n\nThat\'s the extraction.'
    assert _parse_json_safe(raw) == {"foo": 1, "bar": [2, 3]}


def test_parse_json_safe_returns_none_on_garbage():
    assert _parse_json_safe("not json at all") is None
    assert _parse_json_safe("") is None
    assert _parse_json_safe("[1, 2, 3]") is None  # top-level list, not dict


# ─────────────────────────────────────────────────────────────────
# _map_to_legacy_shape — contract with downstream code
# ─────────────────────────────────────────────────────────────────

def _valid_parsed() -> dict:
    return {
        "metrics": [
            {
                "metric_name":    "revenue",
                "metric_value":   100.5,
                "metric_text":    "$100.5M",
                "unit":           "USD_M",
                "segment":        "consolidated",
                "source_snippet": "Revenue was $100.5M for the quarter",
                "confidence":     0.95,
                "is_one_off":     False,
            },
            {
                "metric_name":    "restructuring_charge",
                "metric_value":   12.0,
                "metric_text":    "$12M",
                "unit":           "USD_M",
                "segment":        None,
                "source_snippet": "A one-time $12M restructuring charge",
                "confidence":     0.9,
                "is_one_off":     True,
            },
        ],
        "segments": [
            {"name": "Wholesale", "revenue": 60, "margin": 4.5, "growth": 3.0, "notes": "flat"},
        ],
        "mda_narrative": "The period saw mixed results.",
        "confidence_profile": {"overall_signal": "mixed", "hedge_rate": 0.2, "one_off_rate": 0.05},
        "non_gaap_bridges": [
            {"from_metric": "reported_ni", "to_metric": "adjusted_ni",
             "items": [{"description": "restructuring", "amount": 12}]},
        ],
        "detected_period": "2025_Q4",
        "guidance": [
            {"metric": "revenue_growth", "value_or_range": "3-5%", "specificity": "range"},
        ],
    }


class TestLegacyShapeMapping:
    def test_all_top_level_keys_present(self):
        """Downstream code reads these keys — if any go missing we'd break
        _extract_with_sections callers silently. Verify every one is
        emitted even when the input is minimal."""
        result = _map_to_legacy_shape(_valid_parsed(), "annual_report", use_native_pdf=False)
        expected = {
            "document_type", "extraction_method", "sections_found", "section_types",
            "items_extracted", "raw_items", "mda_narrative", "segment_data",
            "detected_period", "confidence_profile", "disappearance_flags",
            "non_gaap_bridge", "non_gaap_comparison", "reconciliation",
        }
        assert expected.issubset(set(result.keys()))

    def test_raw_items_shape_matches_persistence_contract(self):
        """_persist_earnings_metrics reads these specific keys —
        if the names don't match, metrics silently don't persist."""
        result = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=False)
        item = result["raw_items"][0]
        required = {"metric_name", "metric_value", "metric_text", "unit",
                    "segment", "source_snippet", "confidence", "period",
                    "_is_one_off", "_qualifiers"}
        assert required.issubset(item.keys())
        assert item["metric_name"] == "revenue"
        assert item["metric_value"] == 100.5
        assert item["_is_one_off"] is False

    def test_one_off_flag_preserved(self):
        result = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=False)
        restructuring = next(
            it for it in result["raw_items"] if it["metric_name"] == "restructuring_charge"
        )
        assert restructuring["_is_one_off"] is True

    def test_guidance_promoted_to_raw_items(self):
        """Guidance lives in raw_items with segment='guidance' so it flows
        through the same persistence path as normal metrics. Legacy code
        relies on this convention."""
        result = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=False)
        guidance_rows = [it for it in result["raw_items"] if it["segment"] == "guidance"]
        assert len(guidance_rows) == 1
        assert guidance_rows[0]["metric_name"] == "revenue_growth"
        # metric_text holds the verbatim range ("3-5%")
        assert "3-5%" in guidance_rows[0]["metric_text"]

    def test_skips_metrics_with_empty_name(self):
        """Nameless metrics are garbage — drop them, don't crash."""
        parsed = _valid_parsed()
        parsed["metrics"].append(
            {"metric_name": "", "metric_value": 1, "source_snippet": "junk"}
        )
        parsed["metrics"].append(
            {"metric_name": None, "metric_value": 2, "source_snippet": "more junk"}
        )
        result = _map_to_legacy_shape(parsed, "10-K", use_native_pdf=False)
        # 2 real metrics + 1 guidance (the empty+None metrics dropped)
        assert len(result["raw_items"]) == 3

    def test_extraction_method_reflects_input_mode(self):
        text_mode = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=False)
        pdf_mode = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=True)
        assert text_mode["extraction_method"] == "native_claude_v1_text"
        assert pdf_mode["extraction_method"] == "native_claude_v1_pdf"

    def test_mda_capped_at_20k(self):
        parsed = _valid_parsed()
        parsed["mda_narrative"] = "x" * 25_000
        result = _map_to_legacy_shape(parsed, "10-K", use_native_pdf=False)
        assert len(result["mda_narrative"]) == 20_000

    def test_handles_missing_optional_sections(self):
        """Empty/absent segments/guidance/bridges should still produce a
        valid result with just the metrics."""
        parsed = {
            "metrics": [
                {"metric_name": "revenue", "metric_value": 1, "source_snippet": "s", "confidence": 0.9}
            ],
        }
        result = _map_to_legacy_shape(parsed, "earnings_release", use_native_pdf=False)
        assert result["items_extracted"] == 1
        assert result["segment_data"] is None
        assert result["non_gaap_bridge"] == []
        assert result["mda_narrative"] == ""


# ─────────────────────────────────────────────────────────────────
# _empty_result — fallback contract
# ─────────────────────────────────────────────────────────────────

class TestEmptyResult:
    def test_empty_result_has_same_keys_as_full_result(self):
        """The caller branches on items_extracted == 0 to fall back to
        legacy. If the empty-result shape doesn't match the full shape,
        the caller breaks."""
        full = _map_to_legacy_shape(_valid_parsed(), "10-K", use_native_pdf=False)
        empty = _empty_result("10-K", reason="test")
        assert set(full.keys()) == set(empty.keys())
        assert empty["items_extracted"] == 0
        assert empty["raw_items"] == []

    def test_reason_is_captured_in_method(self):
        empty = _empty_result("10-K", reason="json_parse_failed")
        assert "json_parse_failed" in empty["extraction_method"]


# ─────────────────────────────────────────────────────────────────
# run_native_extraction — end-to-end with mocked LLM
# ─────────────────────────────────────────────────────────────────

def _fake_document(doc_type="annual_report", period="2025_Q4", file_path=None):
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.company_id = uuid.uuid4()
    doc.document_type = doc_type
    doc.period_label = period
    doc.file_path = file_path
    return doc


def _fake_db_company_lookup_empty():
    """AsyncSession stub that returns None for the Company lookup."""
    db = MagicMock()
    res = MagicMock()
    res.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=res)
    return db


def test_run_native_extraction_success_via_text_mode():
    import json as _json
    doc = _fake_document()
    db = _fake_db_company_lookup_empty()

    llm_result = {
        "text": _json.dumps(_valid_parsed()),
        "input_tokens": 10000,
        "output_tokens": 2000,
    }
    with patch(
        "services.llm_client.call_llm_native_async",
        new=AsyncMock(return_value=llm_result),
    ):
        result = asyncio.run(run_native_extraction(
            db, doc, full_text="Some 10-K body text.",
        ))

    assert result["items_extracted"] >= 2
    assert result["extraction_method"] == "native_claude_v1_text"


def test_run_native_extraction_returns_empty_on_llm_exception():
    doc = _fake_document()
    db = _fake_db_company_lookup_empty()

    with patch(
        "services.llm_client.call_llm_native_async",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        result = asyncio.run(run_native_extraction(db, doc, full_text="text"))

    assert result["items_extracted"] == 0
    assert "llm_call_failed" in result["extraction_method"]


def test_run_native_extraction_returns_empty_on_bad_json():
    doc = _fake_document()
    db = _fake_db_company_lookup_empty()

    llm_result = {
        "text": "Sorry, I couldn't parse that document.",
        "input_tokens": 100, "output_tokens": 50,
    }
    with patch(
        "services.llm_client.call_llm_native_async",
        new=AsyncMock(return_value=llm_result),
    ):
        result = asyncio.run(run_native_extraction(db, doc, full_text="text"))

    assert result["items_extracted"] == 0
    assert "json_parse_failed" in result["extraction_method"]


def test_run_native_extraction_truncates_long_text_input():
    """400K-char cap on text input must fire — protect from blowing the
    context window on a 2MB stripped 10-K."""
    import json as _json
    captured = {}

    async def _capture(prompt, **kwargs):
        captured["prompt_len"] = len(prompt)
        captured["pdf_path"] = kwargs.get("pdf_path")
        return {"text": _json.dumps({"metrics": []}), "input_tokens": 1, "output_tokens": 1}

    doc = _fake_document()
    db = _fake_db_company_lookup_empty()
    long_text = "x" * 800_000  # over the 400K cap

    with patch("services.llm_client.call_llm_native_async", new=_capture):
        asyncio.run(run_native_extraction(db, doc, full_text=long_text))

    # Prompt template ~2k chars + truncated text (400k) → should be close to 402k
    assert captured["prompt_len"] < 410_000
    assert captured["prompt_len"] > 390_000  # sanity: truncation actually applied
    assert captured["pdf_path"] is None  # text mode
