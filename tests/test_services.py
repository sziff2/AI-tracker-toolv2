"""
Unit tests for service-level logic (no DB or LLM required).
"""

import json
import pytest

from services.thesis_comparator import _previous_period, _comparable_periods
from services.metric_normaliser import (
    normalise_metric_name,
    normalise_unit,
    normalise_metrics_batch,
    deduplicate_metrics,
    post_process_metrics,
    normalise_period,
    validate_segment_sums,
)
from services.metric_validator import check_plausibility, validate_metrics_batch, filter_by_confidence
from services.metric_extractor import _is_low_value, _smart_chunk


# ─────────────────────────────────────────────────────────────────
# Period Logic
# ─────────────────────────────────────────────────────────────────

class TestPreviousPeriod:
    def test_q2(self):
        assert _previous_period("2026_Q2") == "2026_Q1"

    def test_q1(self):
        assert _previous_period("2026_Q1") == "2025_Q4"

    def test_q4(self):
        assert _previous_period("2025_Q4") == "2025_Q3"

    def test_fy(self):
        assert _previous_period("2025_FY") == "2024_FY"

    def test_hy(self):
        assert _previous_period("2025_HY") == "2024_HY"

    def test_h2(self):
        assert _previous_period("2025_H2") == "2025_H1"

    def test_invalid(self):
        assert _previous_period("invalid") == ""

    def test_empty(self):
        assert _previous_period("") == ""


class TestComparablePeriods:
    def test_q4(self):
        result = _comparable_periods("2025_Q4")
        assert result[0] == "2025_Q3"  # Sequential
        assert "2024_Q4" in result  # YoY

    def test_q1(self):
        result = _comparable_periods("2025_Q1")
        assert result[0] == "2024_Q4"
        assert "2024_Q1" in result

    def test_fy(self):
        result = _comparable_periods("2025_FY")
        assert "2024_FY" in result

    def test_invalid(self):
        assert _comparable_periods("bad") == []


# ─────────────────────────────────────────────────────────────────
# Period Normalisation
# ─────────────────────────────────────────────────────────────────

class TestNormalisePeriod:
    def test_q4_2025(self):
        assert normalise_period("Q4 2025") == "2025_Q4"

    def test_fy_2025(self):
        assert normalise_period("FY 2025") == "2025_FY"

    def test_2025_q4(self):
        assert normalise_period("2025 Q4") == "2025_Q4"

    def test_hy(self):
        assert normalise_period("HY 2025") == "2025_HY"

    def test_empty(self):
        assert normalise_period("") == ""

    def test_single_word(self):
        assert normalise_period("annual") == "annual"


# ─────────────────────────────────────────────────────────────────
# Metric Name Normalisation
# ─────────────────────────────────────────────────────────────────

class TestNormaliseMetricName:
    def test_revenue_variants(self):
        assert normalise_metric_name("Total Revenue") == "Revenue"
        assert normalise_metric_name("net revenue") == "Revenue"
        assert normalise_metric_name("Net Sales") == "Revenue"
        assert normalise_metric_name("total sales") == "Revenue"

    def test_eps_variants(self):
        assert normalise_metric_name("Earnings per share") == "EPS"
        assert normalise_metric_name("diluted eps") == "EPS (Diluted)"

    def test_operating_profit(self):
        assert normalise_metric_name("operating income") == "Operating Profit"

    def test_with_period_prefix(self):
        result = normalise_metric_name("[Q4 2024] total revenue")
        assert "Revenue" in result
        assert "[Q4 2024]" in result

    def test_no_match(self):
        assert normalise_metric_name("Custom Metric XYZ") == "Custom Metric XYZ"

    def test_empty(self):
        assert normalise_metric_name("") == ""

    def test_pattern_match(self):
        assert normalise_metric_name("organic revenue growth rate") == "Organic Revenue Growth"
        assert normalise_metric_name("free cash flow generation") == "Free Cash Flow"


class TestNormaliseUnit:
    def test_usd_variants(self):
        assert normalise_unit("$M") == "USD_M"
        assert normalise_unit("US$M") == "USD_M"
        assert normalise_unit("USD M") == "USD_M"

    def test_percent(self):
        assert normalise_unit("%") == "%"
        assert normalise_unit("PCT") == "%"

    def test_bps(self):
        assert normalise_unit("BPS") == "bps"
        assert normalise_unit("basis points") == "bps"


# ─────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_keeps_highest_confidence(self):
        items = [
            {"metric_name": "Revenue", "period": "Q4 2025", "segment": "Total", "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4 2025", "segment": "Total", "confidence": 0.7},
        ]
        result = deduplicate_metrics(items)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_different_segments_kept(self):
        items = [
            {"metric_name": "Revenue", "period": "Q4 2025", "segment": "Europe", "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4 2025", "segment": "Americas", "confidence": 0.8},
        ]
        result = deduplicate_metrics(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate_metrics([]) == []


# ─────────────────────────────────────────────────────────────────
# Plausibility Checks
# ─────────────────────────────────────────────────────────────────

class TestPlausibility:
    def test_revenue_normal(self):
        result = check_plausibility("Revenue", 5000, "USD_M")
        assert result["plausible"] is True
        assert result["confidence_penalty"] == 0

    def test_revenue_implausible(self):
        result = check_plausibility("Revenue", 999999, "USD_M")
        assert result["plausible"] is False
        assert result["confidence_penalty"] > 0

    def test_margin_normal(self):
        result = check_plausibility("Operating Margin", 15.5, "%")
        assert result["plausible"] is True

    def test_margin_implausible(self):
        result = check_plausibility("Net Margin", 500, "%")
        assert result["plausible"] is False

    def test_none_value(self):
        result = check_plausibility("Revenue", None, "USD_M")
        assert result["plausible"] is True

    def test_unknown_metric(self):
        result = check_plausibility("Custom Metric", 12345, "units")
        assert result["plausible"] is True


# ─────────────────────────────────────────────────────────────────
# Confidence Filtering
# ─────────────────────────────────────────────────────────────────

class TestConfidenceFiltering:
    def test_filters_low_confidence(self):
        items = [
            {"metric_name": "Revenue", "confidence": 0.9},
            {"metric_name": "Bad Metric", "confidence": 0.3},
        ]
        result = filter_by_confidence(items, 0.6)
        assert len(result) == 1
        assert result[0]["metric_name"] == "Revenue"

    def test_keeps_all_above_threshold(self):
        items = [
            {"metric_name": "A", "confidence": 0.8},
            {"metric_name": "B", "confidence": 0.7},
        ]
        assert len(filter_by_confidence(items, 0.6)) == 2


# ─────────────────────────────────────────────────────────────────
# Smart Chunking
# ─────────────────────────────────────────────────────────────────

class TestSmartChunk:
    def test_short_text_single_chunk(self):
        text = "Revenue was $5B. " * 100
        chunks = _smart_chunk(text)
        assert len(chunks) == 1

    def test_low_value_skipped(self):
        assert _is_low_value("short")
        assert _is_low_value("Forward-looking statements disclaimer. This is legal text.")

    def test_financial_content_not_skipped(self):
        text = "Revenue was $5,234 million, up 12.3% from $4,567 million in the prior year."
        assert not _is_low_value(text)

    def test_empty_text_returns_empty(self):
        assert _smart_chunk("") == []


# ─────────────────────────────────────────────────────────────────
# Segment Sum Validation
# ─────────────────────────────────────────────────────────────────

class TestSegmentSumValidation:
    def test_matching_sums(self):
        items = [
            {"metric_name": "Revenue", "period": "Q4", "segment": "Total", "metric_value": 100, "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4", "segment": "Europe", "metric_value": 60, "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4", "segment": "Americas", "metric_value": 40, "confidence": 0.9},
        ]
        result = validate_segment_sums(items)
        # Sums match exactly, no penalty
        for item in result:
            if item["segment"] != "Total":
                assert item["confidence"] == 0.9

    def test_mismatched_sums(self):
        items = [
            {"metric_name": "Revenue", "period": "Q4", "segment": "Total", "metric_value": 100, "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4", "segment": "Europe", "metric_value": 80, "confidence": 0.9},
            {"metric_name": "Revenue", "period": "Q4", "segment": "Americas", "metric_value": 50, "confidence": 0.9},
        ]
        result = validate_segment_sums(items)
        # 80+50=130 vs 100: 30% off, so penalty applied
        for item in result:
            if item["segment"] not in ("Total", ""):
                assert item["confidence"] < 0.9


# ─────────────────────────────────────────────────────────────────
# Post-processing Pipeline
# ─────────────────────────────────────────────────────────────────

class TestPostProcessMetrics:
    def test_full_pipeline(self):
        items = [
            {"metric_name": "total revenue", "period": "Q4 2025", "segment": "Total", "confidence": 0.9, "metric_value": 5000, "unit": "$M"},
            {"metric_name": "Total Revenue", "period": "Q4 2025", "segment": "Total", "confidence": 0.8, "metric_value": 5000, "unit": "USD_M"},
        ]
        result = post_process_metrics(items)
        # Should normalise names and dedup
        assert len(result) == 1
        assert result[0]["metric_name"] == "Revenue"


# ─────────────────────────────────────────────────────────────────
# Settings Validation
# ─────────────────────────────────────────────────────────────────

class TestSettingsValidation:
    def test_cors_origins_list_empty(self):
        from configs.settings import Settings
        s = Settings(cors_origins="")
        assert s.cors_origins_list == []

    def test_cors_origins_list_multiple(self):
        from configs.settings import Settings
        s = Settings(cors_origins="http://localhost:3000,https://app.example.com")
        assert s.cors_origins_list == ["http://localhost:3000", "https://app.example.com"]

    def test_max_upload_bytes(self):
        from configs.settings import Settings
        s = Settings(max_upload_size_mb=10)
        assert s.max_upload_bytes == 10 * 1024 * 1024


# ─────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────

class TestLLMJsonParsing:
    def test_parse_clean_json(self):
        from services.llm_client import _parse_json
        result = _parse_json('[{"metric": "revenue", "value": 100}]')
        assert len(result) == 1
        assert result[0]["metric"] == "revenue"

    def test_parse_with_markdown_fences(self):
        from services.llm_client import _parse_json
        raw = '```json\n[{"a": 1}]\n```'
        result = _parse_json(raw)
        assert result == [{"a": 1}]

    def test_repair_truncated(self):
        from services.llm_client import _repair_truncated_json
        raw = '[{"a": 1}, {"b": 2}'
        # Should find the last } and attempt repair
        result = _repair_truncated_json(raw)
        assert isinstance(result, list)

    def test_usage_tracker(self):
        from services.llm_client import _UsageTracker
        tracker = _UsageTracker()
        tracker.record(100, 50)
        tracker.record(200, 100)
        tracker.record_failure()
        summary = tracker.summary
        assert summary["total_requests"] == 2
        assert summary["failed_requests"] == 1
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150
