"""
Unit tests for service-level logic (no DB or LLM required).
"""

import json
import pytest

from services.metric_normaliser import _previous_period, _comparable_periods
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
from services.metric_extractor import _is_low_value, _smart_chunk, _is_noise_period, _resolve_relative_period
from services.period_derivation import derive_period_metrics


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
        # FY is its own canonical shape — prior is FY of the prior year (YoY).
        assert _previous_period("2025_FY") == "2024_FY"

    def test_hy(self):
        assert _previous_period("2025_HY") == "2024_HY"

    def test_h2(self):
        assert _previous_period("2025_H2") == "2025_H1"

    def test_l3q(self):
        assert _previous_period("2025_L3Q") == "2024_L3Q"

    def test_ltm(self):
        assert _previous_period("2025_LTM") == "2024_LTM"

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
    """Canonical 9-shape taxonomy: Q1, Q2, Q3, Q4, H1, H2, L3Q, FY, LTM.
    No folding — each shape preserves its own period_label."""

    def test_q4_2025(self):
        assert normalise_period("Q4 2025") == "2025_Q4"

    def test_fy_preserved(self):
        # FY is its own canonical shape, not folded into Q4 — the
        # period_frequency column distinguishes 12-month from 3-month figures.
        assert normalise_period("FY 2025") == "2025_FY"
        assert normalise_period("2025_FY") == "2025_FY"
        assert normalise_period("FY25") == "2025_FY"

    def test_2025_q4(self):
        assert normalise_period("2025 Q4") == "2025_Q4"

    def test_h1_preserved(self):
        # H1/HY stays as H1 — distinct from Q2 (3 months vs 6 months).
        assert normalise_period("HY 2025") == "2025_H1"
        assert normalise_period("H1 2025") == "2025_H1"
        assert normalise_period("2025_H1") == "2025_H1"

    def test_h2_preserved(self):
        assert normalise_period("H2 2025") == "2025_H2"
        assert normalise_period("2025_H2") == "2025_H2"

    def test_l3q_nine_months(self):
        # Q3 10-Q YTD figures land as L3Q (9-month accumulation).
        assert normalise_period("9M 2025") == "2025_L3Q"
        assert normalise_period("L3Q 2025") == "2025_L3Q"
        assert normalise_period("Nine Months Ended September 30, 2025") == "2025_L3Q"
        assert normalise_period("2025_L3Q") == "2025_L3Q"

    def test_ltm_trailing(self):
        assert normalise_period("LTM 2025") == "2025_LTM"
        assert normalise_period("TTM 2025") == "2025_LTM"
        assert normalise_period("Trailing Twelve Months ending Sept 2025") == "2025_LTM"

    def test_full_year_variant(self):
        # Common in European filers: "FULL YEAR 2025" instead of "FY 2025".
        assert normalise_period("FULL YEAR 2025") == "2025_FY"
        assert normalise_period("Full-Year 2025") == "2025_FY"

    def test_alt_quarterly_format(self):
        # Inverted format: 1Q-26, 2Q25, 4Q-25, 1Q/26
        assert normalise_period("1Q-26") == "2026_Q1"
        assert normalise_period("2Q25") == "2025_Q2"
        assert normalise_period("4Q-25") == "2025_Q4"
        assert normalise_period("3Q/26") == "2026_Q3"

    def test_period_ended_phrases(self):
        assert normalise_period("Three Months Ended June 30, 2025") == "2025_Q2"
        assert normalise_period("Six Months Ended June 30, 2025") == "2025_H1"
        assert normalise_period("Twelve Months Ended December 31, 2025") == "2025_FY"

    def test_noise_period_filter(self):
        # Bare future-year suffixes from contractual-obligation tables.
        assert _is_noise_period("2027", "2027") is True
        assert _is_noise_period("2030 AND THEREAFTER", "2030 AND THEREAFTER") is True
        assert _is_noise_period("2020 AND PRIOR", "2020 AND PRIOR") is True
        assert _is_noise_period("DUE 3/15/38", "DUE 3/15/38") is True
        # Real periods that canonicalise must NOT be filtered.
        assert _is_noise_period("Q3 2025", "2025_Q3") is False
        assert _is_noise_period("FY 2025", "2025_FY") is False
        assert _is_noise_period("9M 2025", "2025_L3Q") is False
        # Empty input — not noise, just absent.
        assert _is_noise_period("", "") is False

    def test_resolve_relative_period(self):
        # CURRENT inherits the document period.
        assert _resolve_relative_period("CURRENT", "2025_Q3") == "2025_Q3"
        assert _resolve_relative_period("Current Period", "2025_FY") == "2025_FY"
        # PRIOR steps back via _previous_period (Q3 → Q2 same year).
        assert _resolve_relative_period("PRIOR PERIOD", "2025_Q3") == "2025_Q2"
        # PRIOR for FY is YoY (FY 2024).
        assert _resolve_relative_period("PRIOR", "2025_FY") == "2024_FY"
        # Unknown labels pass through unchanged.
        assert _resolve_relative_period("Q4 2025", "2025_Q3") == "Q4 2025"
        assert _resolve_relative_period("", "2025_Q3") == ""

    def test_empty(self):
        assert normalise_period("") == ""

    def test_single_word(self):
        assert normalise_period("annual") == "annual"


# ─────────────────────────────────────────────────────────────────
# Metric Name Normalisation
# ─────────────────────────────────────────────────────────────────

class TestPeriodDerivation:
    """Cross-period additive identities + LTM cross-year derivation."""

    def test_within_year_derivations(self):
        # Q1+Q2 = H1, Q1+Q2+Q3 = L3Q, ... (within-year rules).
        seed = [
            {"metric_name": "sales", "metric_value": 100, "period_label": "2025_Q1", "period_frequency": "Q1"},
            {"metric_name": "sales", "metric_value": 110, "period_label": "2025_Q2", "period_frequency": "Q2"},
            {"metric_name": "sales", "metric_value": 120, "period_label": "2025_Q3", "period_frequency": "Q3"},
            {"metric_name": "sales", "metric_value": 130, "period_label": "2025_Q4", "period_frequency": "Q4"},
        ]
        derived = derive_period_metrics(seed)
        by_freq = {d["period_frequency"]: d["metric_value"] for d in derived}
        assert by_freq.get("H1") == 210         # Q1+Q2
        assert by_freq.get("L3Q") == 330        # Q1+Q2+Q3
        assert by_freq.get("FY") == 460         # Q1+Q2+Q3+Q4
        assert by_freq.get("H2") == 250         # Q3+Q4

    def test_ltm_cross_year(self):
        # LTM ending Q3 2025 = Q4(2024) + Q1(2025) + Q2(2025) + Q3(2025)
        seed = [
            {"metric_name": "sales", "metric_value": 100, "period_label": "2024_Q4", "period_frequency": "Q4"},
            {"metric_name": "sales", "metric_value": 110, "period_label": "2025_Q1", "period_frequency": "Q1"},
            {"metric_name": "sales", "metric_value": 120, "period_label": "2025_Q2", "period_frequency": "Q2"},
            {"metric_name": "sales", "metric_value": 130, "period_label": "2025_Q3", "period_frequency": "Q3"},
        ]
        derived = derive_period_metrics(seed)
        ltm = [d for d in derived if d["period_frequency"] == "LTM"]
        assert len(ltm) == 1
        assert ltm[0]["period_label"] == "2025_LTM"
        assert ltm[0]["metric_value"] == 460
        assert ltm[0]["is_derived"] is True
        assert "LTM as of 2025_Q3" in ltm[0]["source_snippet"]

    def test_ltm_skips_when_real_ltm_exists(self):
        # Don't overwrite a real LTM row with a derived one.
        seed = [
            {"metric_name": "sales", "metric_value": 100, "period_label": "2024_Q4", "period_frequency": "Q4"},
            {"metric_name": "sales", "metric_value": 110, "period_label": "2025_Q1", "period_frequency": "Q1"},
            {"metric_name": "sales", "metric_value": 120, "period_label": "2025_Q2", "period_frequency": "Q2"},
            {"metric_name": "sales", "metric_value": 130, "period_label": "2025_Q3", "period_frequency": "Q3"},
            {"metric_name": "sales", "metric_value": 999, "period_label": "2025_LTM", "period_frequency": "LTM"},
        ]
        derived = derive_period_metrics(seed)
        ltm = [d for d in derived if d["period_frequency"] == "LTM"]
        assert ltm == []  # real LTM exists, no derived one

    def test_ltm_skips_stock_metrics(self):
        # Total assets is a stock — never derive an additive LTM.
        seed = [
            {"metric_name": "total_assets", "metric_value": 5000, "period_label": "2024_Q4", "period_frequency": "Q4"},
            {"metric_name": "total_assets", "metric_value": 5100, "period_label": "2025_Q1", "period_frequency": "Q1"},
            {"metric_name": "total_assets", "metric_value": 5200, "period_label": "2025_Q2", "period_frequency": "Q2"},
            {"metric_name": "total_assets", "metric_value": 5300, "period_label": "2025_Q3", "period_frequency": "Q3"},
        ]
        derived = derive_period_metrics(seed)
        assert derived == []


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
        assert result["confidence_penalty"] <= 0.1  # Small penalty acceptable

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
    def test_settings_loads(self):
        from configs.settings import settings
        assert settings.app_name is not None
        assert settings.max_upload_size_mb > 0

    def test_max_upload_bytes(self):
        from configs.settings import settings
        assert settings.max_upload_bytes == settings.max_upload_size_mb * 1024 * 1024


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
        tracker.record(100, 50, "claude-sonnet-4-6")
        tracker.record(200, 100, "claude-sonnet-4-6")
        tracker.record_failure()
        summary = tracker.summary
        assert summary["total_requests"] == 2
        assert summary["failed_requests"] == 1
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150


# ─────────────────────────────────────────────────────────────────
# Coverage Monitor (pure logic — no DB)
# ─────────────────────────────────────────────────────────────────

from datetime import date
from services.harvester.coverage import expected_period, period_behind, _period_to_tuple


class TestExpectedPeriod:
    def test_mid_april(self):
        # Apr 9: Q4 ends Dec 31 + 75 = Mar 16 → Q4 results expected
        assert expected_period(date(2026, 4, 9)) == "2025_Q4"

    def test_late_june(self):
        # Jun 20: Q1 ends Mar 31 + 75 = Jun 14 → Q1 results expected
        assert expected_period(date(2026, 6, 20)) == "2026_Q1"

    def test_early_june(self):
        # Jun 10: Q1 ends Mar 31 + 75 = Jun 14 → still only Q4 expected
        assert expected_period(date(2026, 6, 10)) == "2025_Q4"

    def test_mid_september(self):
        # Sep 20: Q2 ends Jun 30 + 75 = Sep 13 → Q2 results expected
        assert expected_period(date(2026, 9, 20)) == "2026_Q2"

    def test_late_december(self):
        # Dec 20: Q3 ends Sep 30 + 75 = Dec 14 → Q3 results expected
        assert expected_period(date(2026, 12, 20)) == "2026_Q3"

    def test_january(self):
        # Jan 15: Q3 ends Sep 30 + 75 = Dec 14 → Q3 results expected
        assert expected_period(date(2026, 1, 15)) == "2025_Q3"


class TestPeriodBehind:
    def test_same_period(self):
        assert period_behind("2025_Q4", "2025_Q4") == 0

    def test_one_quarter(self):
        assert period_behind("2025_Q3", "2025_Q4") == 1

    def test_two_quarters(self):
        assert period_behind("2025_Q2", "2025_Q4") == 2

    def test_cross_year(self):
        assert period_behind("2024_Q4", "2025_Q2") == 2

    def test_ahead_is_zero(self):
        assert period_behind("2026_Q1", "2025_Q4") == 0

    def test_bad_format(self):
        # (0,0) vs (2025,4) = 2025*4 + 4 = 8104
        assert period_behind("bad", "2025_Q4") > 0
