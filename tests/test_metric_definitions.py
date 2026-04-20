"""Tests for services/metric_definitions.py — per-metric validation
registry. Pure deterministic checks, no DB, no LLM."""

import pytest

from services.metric_definitions import (
    ValidationIssue, _clean_name, _match_rule,
    validate_metric, validate_metrics_batch,
    _UNIVERSAL_RULES,
)


class TestCleanName:
    def test_basic_lowercase_strip(self):
        assert _clean_name("EPS") == "eps"
        assert _clean_name("  Revenue  ") == "revenue"

    def test_parens_become_spaces(self):
        assert _clean_name("EPS (Diluted)") == "eps diluted"

    def test_slashes_become_spaces(self):
        assert _clean_name("Debt/Equity") == "debt equity"

    def test_empty_is_empty(self):
        assert _clean_name("") == ""
        assert _clean_name(None) == ""


class TestMatchRule:
    def test_exact_match_wins(self):
        m = _match_rule("Operating Margin", _UNIVERSAL_RULES)
        assert m is not None
        key, rule = m
        assert key == "operating margin"

    def test_whole_word_substring_match(self):
        # "Core Operating Margin" contains "operating margin"
        m = _match_rule("Core Operating Margin", _UNIVERSAL_RULES)
        assert m is not None
        key, _ = m
        assert key == "operating margin"

    def test_no_match_returns_none(self):
        assert _match_rule("Widget Throughput", _UNIVERSAL_RULES) is None


class TestValidateMetricUniversal:
    def test_in_range_passes(self):
        # Operating Margin 15% is in the [-50, 60] band
        assert validate_metric("Operating Margin", 15.0, unit="%") is None

    def test_out_of_range_critical(self):
        # Operating Margin of 200% is absurd — >60 upper bound
        issue = validate_metric("Operating Margin", 200.0, unit="%")
        assert issue is not None
        assert issue.severity in {"warning", "critical"}
        # Span=110, margin=(200-60)/110=1.27 > 0.5 → critical
        assert issue.severity == "critical"
        assert "out_of_range" in issue.rule_violated

    def test_just_over_range_warning(self):
        # Operating Margin of 65% — slightly over, margin ≈ 5/110 ≈ 0.045 → warning
        issue = validate_metric("Operating Margin", 65.0, unit="%")
        assert issue is not None
        assert issue.severity == "warning"

    def test_none_value_passes_silently(self):
        assert validate_metric("Operating Margin", None, unit="%") is None

    def test_unknown_metric_passes(self):
        # No rule for this metric at all — not an error, just "no opinion"
        assert validate_metric("Widget Throughput", 42.0, unit="x") is None

    def test_empty_name_passes(self):
        assert validate_metric("", 100.0, unit="%") is None

    def test_non_numeric_value_passes(self):
        assert validate_metric("Operating Margin", "N/A", unit="%") is None


class TestValidateMetricSector:
    def test_bank_nim_in_range(self):
        # NIM of 3.45% is a normal bank NIM
        issue = validate_metric(
            "NIM", 3.45, unit="%", sector="Financials", industry="Banks"
        )
        assert issue is None

    def test_bank_nim_out_of_range_critical(self):
        # NIM of 12% is physically implausible
        issue = validate_metric(
            "NIM", 12.0, unit="%", sector="Financials", industry="Banks"
        )
        assert issue is not None
        assert issue.severity == "critical"
        assert "out_of_range" in issue.rule_violated

    def test_insurance_combined_ratio(self):
        # 95% combined ratio = underwriting profit
        issue = validate_metric(
            "Combined Ratio", 95.0, unit="%",
            sector="Financials", industry="Insurance",
        )
        assert issue is None

    def test_insurance_combined_ratio_far_off(self):
        issue = validate_metric(
            "Combined Ratio", 300.0, unit="%",
            sector="Financials", industry="Insurance",
        )
        assert issue is not None
        assert issue.severity == "critical"

    def test_sector_rule_precedence_over_universal(self):
        # Gross margin 85% would pass universal (bound 95) AND sector
        # (software: 60-90). Pick a value that differentiates — 95
        # passes universal but is out-of-range for software (>90).
        issue = validate_metric(
            "Gross Margin", 95.0, unit="%",
            sector="Information Technology", industry="Software",
        )
        # Sector rules take precedence, so should flag
        assert issue is not None


class TestDenominatorChecks:
    def test_valid_denominator_passes(self):
        issue = validate_metric(
            "Charge_Off_Rate", 0.8, unit="%",
            sector="Financials", industry="Banks",
            denominator="Average Loans",
        )
        # 0.8% is within bank charge-off range (0-5), denominator is valid
        assert issue is None

    def test_invalid_denominator_critical(self):
        # The NWC-CN-class error: using revenue as denominator for
        # charge-off rate is structurally wrong. Must be critical.
        issue = validate_metric(
            "Charge_Off_Rate", 18.0, unit="%",
            sector="Financials", industry="Banks",
            denominator="Revenue",
        )
        # Even if the value was in the charge-off band, the denominator
        # rule should catch it. But 18% is also out of range.
        # Force a value in-range so only denominator fires:
        issue_in_range = validate_metric(
            "Charge_Off_Rate", 2.0, unit="%",
            sector="Financials", industry="Banks",
            denominator="Revenue",
        )
        assert issue_in_range is not None
        assert issue_in_range.severity == "critical"
        assert "invalid_denominator" in issue_in_range.rule_violated

    def test_unknown_denominator_warning(self):
        # Not in the valid list and not in the invalid list — warning
        issue = validate_metric(
            "Charge_Off_Rate", 2.0, unit="%",
            sector="Financials", industry="Banks",
            denominator="Gross Loans",  # not in valid_denominators
        )
        assert issue is not None
        assert issue.severity == "warning"
        assert "unknown_denominator" in issue.rule_violated


class TestValidateMetricsBatch:
    def test_mixed_batch(self):
        metrics = [
            {"metric_name": "Revenue Growth", "metric_value": 5.0, "unit": "%"},
            {"metric_name": "Operating Margin", "metric_value": 500.0, "unit": "%"},
            {"metric_name": "Unknown Metric", "metric_value": 123},
        ]
        issues = validate_metrics_batch(metrics)
        assert len(issues) == 1
        assert issues[0].metric_name == "Operating Margin"

    def test_empty_batch(self):
        assert validate_metrics_batch([]) == []

    def test_sector_applies_to_whole_batch(self):
        metrics = [
            {"metric_name": "NIM", "metric_value": 3.5, "unit": "%"},
            {"metric_name": "NIM", "metric_value": 50.0, "unit": "%"},
        ]
        issues = validate_metrics_batch(
            metrics, sector="Financials", industry="Banks"
        )
        assert len(issues) == 1
        assert issues[0].value == 50.0
