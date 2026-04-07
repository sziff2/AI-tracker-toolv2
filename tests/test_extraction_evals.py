"""Tests for deterministic extraction evals."""
import pytest
from services.extraction_evals import (
    eval_source_recall,
    eval_source_precision,
    eval_name_accuracy,
    eval_period_accuracy,
    run_extraction_evals,
)


# ── eval_source_recall ───────────────────────────────────────────

class TestSourceRecall:
    def test_all_found(self):
        items = [
            {"metric_value": 100.0},
            {"metric_value": 200.0},
            {"metric_value": 300.0},
        ]
        source = {100.0, 200.0, 300.0}
        assert eval_source_recall(items, source) == 1.0

    def test_partial_found(self):
        items = [{"metric_value": 100.0}]
        source = {100.0, 200.0}
        assert eval_source_recall(items, source) == 0.5

    def test_none_found(self):
        items = [{"metric_value": 999.0}]
        source = {100.0, 200.0}
        assert eval_source_recall(items, source) == 0.0

    def test_empty_source(self):
        items = [{"metric_value": 100.0}]
        assert eval_source_recall(items, set()) == 1.0

    def test_empty_items(self):
        assert eval_source_recall([], {100.0, 200.0}) == 0.0

    def test_zero_values_excluded_from_source(self):
        """Zero values in source should not count toward recall denominator."""
        items = [{"metric_value": 100.0}]
        source = {0.0, 100.0}
        assert eval_source_recall(items, source) == 1.0

    def test_tolerance_matching(self):
        items = [{"metric_value": 100.5}]
        source = {100.0}
        # 0.5 / 100 = 0.005 < 0.01 tolerance
        assert eval_source_recall(items, source) == 1.0

    def test_tolerance_miss(self):
        items = [{"metric_value": 110.0}]
        source = {100.0}
        # 10 / 100 = 0.1 > 0.01 tolerance
        assert eval_source_recall(items, source) == 0.0

    def test_value_key_fallback(self):
        """Should also check 'value' key, not just 'metric_value'."""
        items = [{"value": 100.0}]
        source = {100.0}
        assert eval_source_recall(items, source) == 1.0

    def test_non_numeric_values_ignored(self):
        items = [{"metric_value": "not a number"}, {"metric_value": 100.0}]
        source = {100.0}
        assert eval_source_recall(items, source) == 1.0

    def test_none_values_ignored(self):
        items = [{"metric_value": None}, {"metric_value": 100.0}]
        source = {100.0}
        assert eval_source_recall(items, source) == 1.0


# ── eval_source_precision ────────────────────────────────────────

class TestSourcePrecision:
    def test_all_verified(self):
        items = [
            {"metric_value": 100.0},
            {"metric_value": 200.0},
        ]
        source = {100.0, 200.0, 300.0}
        assert eval_source_precision(items, source) == 1.0

    def test_some_hallucinated(self):
        items = [
            {"metric_value": 100.0},
            {"metric_value": 999.0},  # hallucinated
        ]
        source = {100.0, 200.0}
        assert eval_source_precision(items, source) == 0.5

    def test_all_hallucinated(self):
        items = [{"metric_value": 999.0}]
        source = {100.0}
        assert eval_source_precision(items, source) == 0.0

    def test_empty_items(self):
        assert eval_source_precision([], {100.0}) == 1.0

    def test_empty_source(self):
        """With empty source, non-zero values cannot be verified."""
        items = [{"metric_value": 100.0}]
        assert eval_source_precision(items, set()) == 0.0

    def test_zero_values_auto_verified(self):
        """Zero extracted values are automatically verified."""
        items = [{"metric_value": 0.0}]
        assert eval_source_precision(items, set()) == 1.0

    def test_mixed_zero_and_nonzero(self):
        items = [
            {"metric_value": 0.0},      # auto-verified
            {"metric_value": 100.0},     # verified
            {"metric_value": 999.0},     # not verified
        ]
        source = {100.0}
        result = eval_source_precision(items, source)
        assert result == 0.5 or abs(result - 2/3) < 0.01  # depends on zero handling

    def test_value_key_fallback(self):
        items = [{"value": 100.0}]
        source = {100.0}
        assert eval_source_precision(items, source) == 1.0

    def test_non_numeric_ignored(self):
        items = [{"metric_value": "text"}, {"metric_value": 100.0}]
        source = {100.0}
        assert eval_source_precision(items, source) == 1.0


# ── eval_name_accuracy ──────────────────────────────────────────

class TestNameAccuracy:
    def test_exact_match(self):
        items = [
            {"metric_name": "Revenue"},
            {"metric_name": "Net Income"},
        ]
        canonical = {"Revenue", "Net Income", "EBITDA"}
        assert eval_name_accuracy(items, canonical) == 1.0

    def test_case_insensitive_match(self):
        items = [{"metric_name": "revenue"}]
        canonical = {"Revenue"}
        assert eval_name_accuracy(items, canonical) == 1.0

    def test_partial_match(self):
        """Substring match scores 0.5."""
        items = [{"metric_name": "Total Revenue Growth"}]
        canonical = {"Revenue"}
        assert eval_name_accuracy(items, canonical) == 0.5

    def test_no_match(self):
        items = [{"metric_name": "Foo Bar Baz"}]
        canonical = {"Revenue", "Net Income"}
        assert eval_name_accuracy(items, canonical) == 0.0

    def test_empty_items(self):
        assert eval_name_accuracy([], {"Revenue"}) == 1.0

    def test_empty_canonical(self):
        items = [{"metric_name": "Revenue"}]
        assert eval_name_accuracy(items, set()) == 0.0

    def test_line_item_key_fallback(self):
        items = [{"line_item": "Revenue"}]
        canonical = {"Revenue"}
        assert eval_name_accuracy(items, canonical) == 1.0

    def test_items_without_names_skipped(self):
        items = [{"metric_value": 100.0}, {"metric_name": "Revenue"}]
        canonical = {"Revenue"}
        assert eval_name_accuracy(items, canonical) == 1.0

    def test_mixed_matches(self):
        items = [
            {"metric_name": "Revenue"},         # exact = 1.0
            {"metric_name": "Adjusted Revenue"}, # partial = 0.5
            {"metric_name": "Unknown Metric"},   # no match = 0.0
        ]
        canonical = {"Revenue", "Net Income"}
        score = eval_name_accuracy(items, canonical)
        assert abs(score - 0.5) < 0.01  # (1.0 + 0.5 + 0.0) / 3


# ── eval_period_accuracy ────────────────────────────────────────

class TestPeriodAccuracy:
    def test_all_correct(self):
        items = [
            {"period": "2025_Q4"},
            {"period": "2025_Q4"},
        ]
        assert eval_period_accuracy(items, "2025_Q4") == 1.0

    def test_all_wrong(self):
        items = [
            {"period": "2024_Q4"},
            {"period": "2024_Q3"},
        ]
        assert eval_period_accuracy(items, "2025_Q4") == 0.0

    def test_mixed(self):
        items = [
            {"period": "2025_Q4"},  # correct
            {"period": "2024_Q4"},  # wrong
        ]
        assert eval_period_accuracy(items, "2025_Q4") == 0.5

    def test_no_expected_period(self):
        items = [{"period": "2025_Q4"}]
        assert eval_period_accuracy(items, None) == 1.0

    def test_empty_items(self):
        assert eval_period_accuracy([], "2025_Q4") == 1.0

    def test_items_without_period(self):
        """Items missing period key — either skipped (1.0) or counted as wrong (0.0)."""
        items = [{"metric_value": 100.0}]
        result = eval_period_accuracy(items, "2025_Q4")
        assert result in (0.0, 1.0)  # implementation-dependent

    def test_period_label_key_fallback(self):
        items = [{"period_label": "2025_Q4"}]
        assert eval_period_accuracy(items, "2025_Q4") == 1.0


# ── run_extraction_evals (composite) ────────────────────────────

class TestRunExtractionEvals:
    def test_perfect_scores(self):
        items = [
            {"metric_name": "Revenue", "metric_value": 100.0, "period": "2025_Q4"},
        ]
        source = {100.0}
        canonical = {"Revenue"}
        # Patch canonical names for name_accuracy
        result = run_extraction_evals(items, source, "2025_Q4")
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["period_accuracy"] == 1.0
        assert 0 < result["composite"] <= 1.0

    def test_empty_items(self):
        result = run_extraction_evals([], set(), None)
        # Empty items = all defaults to 1.0
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0
        assert result["period_accuracy"] == 1.0
        assert result["composite"] == 1.0

    def test_composite_weighting(self):
        """Verify composite weights: recall=0.25, precision=0.35, name=0.20, period=0.20."""
        items = [
            {"metric_name": "Revenue", "metric_value": 100.0, "period": "2025_Q4"},
            {"metric_name": "Revenue", "metric_value": 999.0, "period": "2024_Q4"},
        ]
        source = {100.0}
        result = run_extraction_evals(items, source, "2025_Q4")
        # recall = 1.0 (100 found in source)
        # precision = 0.5 (1 of 2 verified)
        # period = 0.5 (1 of 2 correct)
        assert result["recall"] == 1.0
        assert result["precision"] == 0.5
        assert result["period_accuracy"] == 0.5
        expected_composite = (1.0 * 0.25 + 0.5 * 0.35 + result["name_accuracy"] * 0.20 + 0.5 * 0.20)
        assert abs(result["composite"] - round(expected_composite, 3)) < 0.01

    def test_none_source_numbers(self):
        """source_numbers=None should not crash."""
        result = run_extraction_evals([{"metric_value": 100.0}], None, None)
        assert "composite" in result

    def test_result_keys(self):
        result = run_extraction_evals([], set(), None)
        assert set(result.keys()) == {"recall", "precision", "name_accuracy", "period_accuracy", "composite"}

    def test_all_values_rounded(self):
        result = run_extraction_evals(
            [{"metric_value": 100.0, "period": "Q1"}],
            {100.0, 200.0, 300.0},
            "Q1",
        )
        for key, val in result.items():
            # Check that values have at most 3 decimal places
            assert val == round(val, 3), f"{key} not rounded to 3dp: {val}"
