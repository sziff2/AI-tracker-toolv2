"""
evals/financial_extraction_evals.py
=====================================
Eval suite for the financial statement extraction pipeline (session 0.8).

Covers four eval categories:
  1. ACCURACY     — extracted values vs ground truth
  2. HALLUCINATION — invented metrics, confidence calibration
  3. RECONCILIATION — cross-checks catch real errors
  4. EFFICIENCY    — token cost and latency regression

Run:
    pytest evals/financial_extraction_evals.py -v
    pytest evals/financial_extraction_evals.py -v -m accuracy
    pytest evals/financial_extraction_evals.py -v -m efficiency

Environment:
    OPENAI_API_KEY or ANTHROPIC_API_KEY must be set.
    TEST_FIXTURES_DIR=evals/fixtures (default)
"""

import json
import os
import time
import pytest
from pathlib import Path
from typing import Any

# ── Internal imports (adjust to your actual module paths) ──────────────────────
from services.financial_statement_segmenter import (
    FinancialStatementSegmenter,
    StatementType,
    parse_financial_value,
    parse_period_label,
    detect_units,
    classify_table,
)
from services.statement_extractors import StatementExtractors
from services.extraction_reconciler import reconcile_extractions
from services.metric_extractor import extract_combined  # legacy path, for regression
from services.document_parser import DocumentParser

# ── Paths ──────────────────────────────────────────────────────────────────────
FIXTURES_DIR = Path(os.getenv("TEST_FIXTURES_DIR", "evals/fixtures"))
GROUND_TRUTH_FILE = FIXTURES_DIR / "ground_truth.json"

# ── Tolerances ─────────────────────────────────────────────────────────────────
VALUE_TOLERANCE_PCT = 0.005   # 0.5% — allow for rounding
CONFIDENCE_BIN_SIZE = 0.1     # for calibration bucketing

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def ground_truth() -> dict:
    """
    Load ground-truth JSON.
    Schema: see evals/fixtures/ground_truth_schema.json
    Example entry:
    {
      "doc_id": "ASML_Q4_2024",
      "file": "ASML_Q4_2024_earnings_release.pdf",
      "metrics": [
        {
          "statement_type": "income_statement",
          "period": "Q4 2024",
          "label": "Revenue",
          "expected_value": 9257.0,
          "currency": "EUR",
          "unit_scale": "millions"
        },
        ...
      ],
      "expected_reconciliation_pass": true,
      "known_issues": []
    }
    """
    if not GROUND_TRUTH_FILE.exists():
        pytest.skip(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
    with open(GROUND_TRUTH_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def segmenter():
    return FinancialStatementSegmenter()


@pytest.fixture(scope="session")
def extractor():
    return StatementExtractors()


@pytest.fixture(scope="session")
def parser():
    return DocumentParser()


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: ACCURACY EVALS
# ══════════════════════════════════════════════════════════════════════════════

class TestValueAccuracy:
    """
    Core accuracy: does the pipeline extract the right numbers?
    Compares extracted values to manually verified ground truth.
    """

    @pytest.mark.accuracy
    def test_income_statement_revenue_accuracy(self, ground_truth, segmenter, extractor, parser):
        """Revenue should match to within 0.5% across all test documents."""
        errors = []
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            for gt_metric in doc["metrics"]:
                if gt_metric["statement_type"] != "income_statement":
                    continue
                if gt_metric["label"].lower() != "revenue":
                    continue

                extracted = _find_extracted_value(
                    results, gt_metric["statement_type"],
                    gt_metric["period"], gt_metric["label"]
                )

                if extracted is None:
                    errors.append(f"{doc['doc_id']} | {gt_metric['period']} Revenue: NOT FOUND")
                    continue

                pct_diff = abs(extracted - gt_metric["expected_value"]) / gt_metric["expected_value"]
                if pct_diff > VALUE_TOLERANCE_PCT:
                    errors.append(
                        f"{doc['doc_id']} | {gt_metric['period']} Revenue: "
                        f"expected {gt_metric['expected_value']}, got {extracted} "
                        f"({pct_diff*100:.2f}% off)"
                    )

        assert not errors, "Revenue extraction errors:\n" + "\n".join(errors)

    @pytest.mark.accuracy
    def test_period_attribution_accuracy(self, ground_truth, segmenter, extractor, parser):
        """
        Metrics must be tagged to the correct period.
        This is the primary error the segmenter is designed to prevent.
        """
        errors = []
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            for gt_metric in doc["metrics"]:
                extracted = _find_extracted_value(
                    results, gt_metric["statement_type"],
                    gt_metric["period"], gt_metric["label"]
                )
                if extracted is None:
                    continue

                # Check that we can also NOT find it under the wrong period
                wrong_period = _get_wrong_period(gt_metric["period"])
                wrong = _find_extracted_value(
                    results, gt_metric["statement_type"],
                    wrong_period, gt_metric["label"]
                )
                if wrong is not None and abs(wrong - gt_metric["expected_value"]) / gt_metric["expected_value"] < 0.01:
                    errors.append(
                        f"{doc['doc_id']} | {gt_metric['label']} appears under WRONG period "
                        f"'{wrong_period}' (correct: '{gt_metric['period']}')"
                    )

        assert not errors, "Period attribution errors:\n" + "\n".join(errors)

    @pytest.mark.accuracy
    def test_segment_vs_consolidated_not_confused(self, ground_truth, segmenter, extractor, parser):
        """
        Segment revenues must not be tagged as consolidated, and vice versa.
        Historical error rate ~10% — target <2%.
        """
        errors = []
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            # Check that segment items are tagged with segment names
            for item in results.get("segments", []):
                if item.get("segment") is None or item["segment"].strip() == "":
                    errors.append(
                        f"{doc['doc_id']} | Segment item '{item.get('label')}' "
                        f"period '{item.get('period')}' has no segment tag"
                    )

            # Check consolidated items are not tagged with segment names
            for item in results.get("income_statement", {}).get("items", []):
                if item.get("segment") and item["segment"].strip() != "":
                    errors.append(
                        f"{doc['doc_id']} | Consolidated P&L item '{item.get('label')}' "
                        f"incorrectly tagged with segment '{item['segment']}'"
                    )

        assert not errors, "Segment/consolidated confusion errors:\n" + "\n".join(errors)

    @pytest.mark.accuracy
    def test_parenthetical_negative_parsing(self):
        """
        Parenthetical negatives like (1,234) must parse to -1234.0.
        This is the #1 sign error in cash flow statements.
        """
        cases = [
            ("(1,234)", -1234.0),
            ("(1,234.5)", -1234.5),
            ("(0.5)", -0.5),
            ("1,234", 1234.0),
            ("—", None),
            ("n/a", None),
            ("nm", None),
            ("(123.4)", -123.4),
            ("€1,234", 1234.0),
            ("$1,234.56", 1234.56),
        ]
        for text, expected in cases:
            result = parse_financial_value(text)
            assert result == expected, f"parse_financial_value({text!r}) = {result}, expected {expected}"

    @pytest.mark.accuracy
    def test_currency_unit_detection(self):
        """Currency and unit scale must be identified correctly."""
        cases = [
            (["€ millions", "EUR m", "Revenue"], "EUR", "millions"),
            (["USD thousands", "$'000"], "USD", "thousands"),
            (["£ billions", "GBP bn"], "GBP", "billions"),
            (["in millions of euros"], "EUR", "millions"),
        ]
        for table_context, exp_currency, exp_unit in cases:
            fake_table = [[cell] for cell in table_context]
            currency, unit = detect_units(fake_table, " ".join(table_context))
            assert currency == exp_currency, f"Currency mismatch for {table_context}: got {currency}"
            assert unit == exp_unit, f"Unit mismatch for {table_context}: got {unit}"

    @pytest.mark.accuracy
    def test_period_label_parsing(self):
        """Period labels in column headers must parse correctly."""
        cases = [
            ("Q4 2025", {"label": "Q4 2025", "type": "quarter"}),
            ("4Q25", {"label": "Q4 2025", "type": "quarter"}),
            ("4Q 2025", {"label": "Q4 2025", "type": "quarter"}),
            ("Three months ended Dec 31, 2025", {"label": "Q4 2025", "type": "quarter"}),
            ("Year ended December 31, 2025", {"label": "FY 2025", "type": "annual"}),
            ("FY2025", {"label": "FY 2025", "type": "annual"}),
            ("H1 2025", {"label": "H1 2025", "type": "half"}),
            ("2025", {"label": "FY 2025", "type": "annual"}),
        ]
        for text, expected in cases:
            result = parse_period_label(text)
            assert result is not None, f"parse_period_label({text!r}) returned None"
            assert result["label"] == expected["label"], (
                f"Label mismatch for {text!r}: got {result['label']}, expected {expected['label']}"
            )
            assert result["type"] == expected["type"], (
                f"Type mismatch for {text!r}: got {result['type']}, expected {expected['type']}"
            )

    @pytest.mark.accuracy
    def test_balance_sheet_vs_income_statement_not_confused(self):
        """
        BS items (assets, liabilities) must be classified as BALANCE_SHEET.
        P&L items must be classified as INCOME_STATEMENT.
        """
        bs_table = [
            ["", "Dec 2025", "Dec 2024"],
            ["Cash and cash equivalents", "4,567", "3,890"],
            ["Total assets", "45,234", "43,678"],
            ["Total liabilities", "28,456", "27,123"],
            ["Shareholders equity", "16,778", "16,555"],
        ]
        pl_table = [
            ["", "Q4 2025", "Q4 2024"],
            ["Revenue", "8,234", "7,891"],
            ["Operating profit", "1,123", "987"],
            ["Net income", "812", "701"],
        ]
        assert classify_table(bs_table, "balance sheet", 3) == StatementType.BALANCE_SHEET
        assert classify_table(pl_table, "income statement", 4) == StatementType.INCOME_STATEMENT


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: HALLUCINATION EVALS
# ══════════════════════════════════════════════════════════════════════════════

class TestHallucination:
    """
    Does the pipeline invent data that isn't in the document?
    These are the most dangerous failures — a fabricated number looks real.
    """

    @pytest.mark.hallucination
    def test_absent_metric_not_fabricated(self, segmenter, extractor, parser):
        """
        If a metric doesn't appear in the document, it must not appear in output.
        Uses a synthetic document that omits certain standard line items.
        """
        # Minimal income statement — no EBITDA, no segment data
        minimal_pdf = FIXTURES_DIR / "synthetic" / "minimal_income_statement.pdf"
        if not minimal_pdf.exists():
            pytest.skip("Synthetic fixture not found")

        tables = parser.extract_tables(str(minimal_pdf))
        structure = segmenter.segment(tables, str(minimal_pdf))
        results = extractor.run_all(structure)

        # EBITDA is not in the doc — must not appear
        ebitda = _find_extracted_value(results, "income_statement", "Q4 2025", "EBITDA")
        assert ebitda is None, f"EBITDA was fabricated: got {ebitda}"

        # No segment data in doc — segment results must be empty
        assert not results.get("segments"), (
            f"Segment data fabricated for a document with no segments: {results['segments']}"
        )

    @pytest.mark.hallucination
    def test_prior_period_not_invented(self, ground_truth, segmenter, extractor, parser):
        """
        When only Q4 data is present, Q1/Q2/Q3 must not be fabricated.
        """
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue
            if doc.get("has_quarterly_breakdown", True):
                continue  # Skip docs that actually have quarterly data

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            for q in ["Q1", "Q2", "Q3"]:
                val = _find_extracted_value(results, "income_statement", f"{q} 2025", "Revenue")
                assert val is None, (
                    f"{doc['doc_id']}: Q period {q} Revenue was fabricated "
                    f"(value: {val}) despite not being in the document"
                )

    @pytest.mark.hallucination
    def test_confidence_score_calibration(self, ground_truth, segmenter, extractor, parser):
        """
        Calibration: metrics marked confidence=0.9 should be correct ~90% of the time.
        Bins confidence in 0.1 increments and checks accuracy within each bin.
        Minimum 10 samples per bin for a valid calibration check.

        Acceptable: calibration error (|expected_acc - actual_acc|) < 0.15 per bin.
        """
        bins = {round(i * 0.1, 1): {"correct": 0, "total": 0} for i in range(1, 11)}

        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            for gt_metric in doc["metrics"]:
                extracted_item = _find_extracted_item(
                    results, gt_metric["statement_type"],
                    gt_metric["period"], gt_metric["label"]
                )
                if extracted_item is None:
                    continue

                confidence = extracted_item.get("confidence", 1.0)
                bin_key = round(round(confidence / CONFIDENCE_BIN_SIZE) * CONFIDENCE_BIN_SIZE, 1)
                bin_key = min(max(bin_key, 0.1), 1.0)

                is_correct = abs(extracted_item["value"] - gt_metric["expected_value"]) \
                             / gt_metric["expected_value"] < VALUE_TOLERANCE_PCT
                bins[bin_key]["total"] += 1
                if is_correct:
                    bins[bin_key]["correct"] += 1

        calibration_errors = []
        for expected_acc, counts in bins.items():
            if counts["total"] < 10:
                continue
            actual_acc = counts["correct"] / counts["total"]
            error = abs(expected_acc - actual_acc)
            if error > 0.15:
                calibration_errors.append(
                    f"Bin {expected_acc:.1f}: expected ~{expected_acc*100:.0f}% correct, "
                    f"got {actual_acc*100:.0f}% ({counts['correct']}/{counts['total']})"
                )

        assert not calibration_errors, (
            "Confidence calibration is off:\n" + "\n".join(calibration_errors)
        )

    @pytest.mark.hallucination
    def test_no_cross_document_contamination(self, segmenter, extractor, parser, ground_truth):
        """
        Running extraction on doc A must not produce values from doc B.
        Tests isolation between parallel calls.
        """
        if len(ground_truth["documents"]) < 2:
            pytest.skip("Need at least 2 documents for contamination test")

        doc_a = ground_truth["documents"][0]
        doc_b = ground_truth["documents"][1]

        pdf_a = FIXTURES_DIR / "pdfs" / doc_a["file"]
        pdf_b = FIXTURES_DIR / "pdfs" / doc_b["file"]
        if not pdf_a.exists() or not pdf_b.exists():
            pytest.skip("Fixture PDFs not found")

        # Extract doc_a
        tables_a = parser.extract_tables(str(pdf_a))
        structure_a = segmenter.segment(tables_a, str(pdf_a))
        results_a = extractor.run_all(structure_a)

        # Collect all unique values from doc_a output
        values_a = set(_collect_all_values(results_a))

        # Collect doc_b's ground truth values that don't appear in doc_a's truth
        doc_a_truth_values = {m["expected_value"] for m in doc_a["metrics"]}
        doc_b_only_values = {
            m["expected_value"] for m in doc_b["metrics"]
            if m["expected_value"] not in doc_a_truth_values
        }

        contaminated = values_a & doc_b_only_values
        assert not contaminated, (
            f"Doc A extraction contains {len(contaminated)} values that are "
            f"unique to Doc B: {list(contaminated)[:5]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: RECONCILIATION EVALS
# ══════════════════════════════════════════════════════════════════════════════

class TestReconciliation:
    """
    Does the reconciler correctly catch internal inconsistencies?
    Two sub-evals: sensitivity (catches real errors) and specificity (no false positives).
    """

    @pytest.mark.reconciliation
    def test_reconciler_catches_quarterly_sum_mismatch(self):
        """
        If Q4 + Q1-Q3 ≠ FY (>2%), reconciler must flag it.
        Inject a known error and verify it's caught.
        """
        good_results = _build_synthetic_results(
            fy_revenue=31456.0,
            quarterly_revenues={"Q1": 7500, "Q2": 7800, "Q3": 7900, "Q4": 8256},
        )
        # Inject error: tamper FY to be wrong
        bad_results = _build_synthetic_results(
            fy_revenue=35000.0,  # wrong — should be ~31456
            quarterly_revenues={"Q1": 7500, "Q2": 7800, "Q3": 7900, "Q4": 8256},
        )
        good_report = reconcile_extractions({}, good_results)
        bad_report = reconcile_extractions({}, bad_results)

        assert good_report["passed"], f"Reconciler falsely flagged good data: {good_report['issues']}"
        assert not bad_report["passed"], "Reconciler failed to catch quarterly vs annual mismatch"
        assert any(i["check"] == "quarterly_sum_vs_annual" for i in bad_report["issues"])

    @pytest.mark.reconciliation
    def test_reconciler_catches_segment_sum_mismatch(self):
        """Segment revenues must sum to consolidated revenue ±2%."""
        bad_results = _build_synthetic_results_with_segments(
            consolidated_revenue=10000.0,
            segment_revenues={"Europe": 3000, "Americas": 3000, "Asia": 2000},
            # Sum = 8000, but consolidated = 10000 → mismatch
        )
        report = reconcile_extractions({}, bad_results)
        assert not report["passed"]
        assert any(i["check"] == "segment_sum_vs_consolidated" for i in report["issues"])

    @pytest.mark.reconciliation
    def test_reconciler_catches_balance_sheet_equation(self):
        """Total Assets must equal Total Liabilities + Equity."""
        bad_results = _build_synthetic_results_bs(
            total_assets=45000.0,
            total_liab_equity=41000.0,  # mismatch > 1%
        )
        report = reconcile_extractions({}, bad_results)
        assert not report["passed"]
        assert any(i["check"] == "balance_sheet_equation" for i in report["issues"])

    @pytest.mark.reconciliation
    def test_reconciler_no_false_positives_on_clean_data(self, ground_truth, segmenter, extractor, parser):
        """
        On documents that are internally consistent, reconciler must not raise false alarms.
        Acceptable false positive rate: <5%.
        """
        false_positives = 0
        total_docs = 0

        for doc in ground_truth["documents"]:
            if not doc.get("expected_reconciliation_pass", True):
                continue  # Skip docs known to have issues
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)
            report = reconcile_extractions(structure, results)

            total_docs += 1
            if not report["passed"]:
                false_positives += 1

        if total_docs == 0:
            pytest.skip("No fixture PDFs available")

        fp_rate = false_positives / total_docs
        assert fp_rate < 0.05, (
            f"Reconciler false positive rate too high: {fp_rate*100:.1f}% "
            f"({false_positives}/{total_docs} clean docs flagged)"
        )

    @pytest.mark.reconciliation
    def test_reconciler_sensitivity_on_known_issues(self, ground_truth, segmenter, extractor, parser):
        """
        On documents with known issues (flagged in ground truth), reconciler must catch them.
        Target: ≥80% sensitivity.
        """
        caught = 0
        total = 0
        for doc in ground_truth["documents"]:
            if doc.get("expected_reconciliation_pass", True):
                continue
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)
            report = reconcile_extractions(structure, results)

            total += 1
            if not report["passed"]:
                caught += 1

        if total == 0:
            pytest.skip("No docs with expected issues in ground truth")

        sensitivity = caught / total
        assert sensitivity >= 0.80, (
            f"Reconciler sensitivity too low: {sensitivity*100:.0f}% "
            f"({caught}/{total} known issues caught)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: EFFICIENCY EVALS
# ══════════════════════════════════════════════════════════════════════════════

class TestEfficiency:
    """
    Token cost and latency must not regress.
    Thresholds based on architecture targets:
      - Token cost: ≤5,500 tokens per typical 30-page earnings release
      - Latency:    ≤13 seconds end-to-end
    """

    @pytest.mark.efficiency
    def test_token_cost_per_document(self, ground_truth, segmenter, extractor, parser):
        """
        New pipeline (pre-segmented) should use ≤5,500 tokens per document.
        Architecture target: ~5K total (down from ~8K).
        """
        overbudget = []
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)

            total_tokens = results.get("_meta", {}).get("total_tokens_used", None)
            if total_tokens is None:
                pytest.skip("Token tracking not implemented in extractor — add _meta.total_tokens_used")

            if total_tokens > 5500:
                overbudget.append(f"{doc['doc_id']}: {total_tokens} tokens (limit: 5,500)")

        assert not overbudget, "Documents exceeded token budget:\n" + "\n".join(overbudget)

    @pytest.mark.efficiency
    def test_latency_per_document(self, ground_truth, segmenter, extractor, parser):
        """
        End-to-end extraction (segmenter + parallel LLM calls + reconciliation)
        must complete in ≤13 seconds for a typical earnings release.
        Architecture target: ~8-12s (down from ~15-20s).
        """
        slow_docs = []
        for doc in ground_truth["documents"]:
            pdf_path = FIXTURES_DIR / "pdfs" / doc["file"]
            if not pdf_path.exists():
                continue

            t0 = time.perf_counter()
            tables = parser.extract_tables(str(pdf_path))
            structure = segmenter.segment(tables, str(pdf_path))
            results = extractor.run_all(structure)
            reconcile_extractions(structure, results)
            elapsed = time.perf_counter() - t0

            if elapsed > 13.0:
                slow_docs.append(f"{doc['doc_id']}: {elapsed:.1f}s (limit: 13s)")

        assert not slow_docs, "Documents exceeded latency budget:\n" + "\n".join(slow_docs)

    @pytest.mark.efficiency
    def test_token_regression_vs_legacy_pipeline(self, ground_truth, segmenter, extractor, parser):
        """
        New pipeline must use fewer tokens than the legacy one-big-prompt approach.
        Uses the first available fixture document for the comparison.
        """
        doc = next(
            (d for d in ground_truth["documents"]
             if (FIXTURES_DIR / "pdfs" / d["file"]).exists()),
            None
        )
        if doc is None:
            pytest.skip("No fixture PDFs available")

        pdf_path = str(FIXTURES_DIR / "pdfs" / doc["file"])

        # New pipeline
        tables = parser.extract_tables(pdf_path)
        structure = segmenter.segment(tables, pdf_path)
        new_results = extractor.run_all(structure)
        new_tokens = new_results.get("_meta", {}).get("total_tokens_used")

        # Legacy pipeline
        legacy_results = extract_combined(pdf_path)
        legacy_tokens = legacy_results.get("_meta", {}).get("total_tokens_used")

        if new_tokens is None or legacy_tokens is None:
            pytest.skip("Token tracking not implemented")

        assert new_tokens < legacy_tokens, (
            f"New pipeline uses MORE tokens than legacy: {new_tokens} vs {legacy_tokens}"
        )
        savings_pct = (legacy_tokens - new_tokens) / legacy_tokens * 100
        print(f"\nToken savings: {savings_pct:.0f}% ({legacy_tokens} → {new_tokens})")

    @pytest.mark.efficiency
    def test_parallel_extraction_faster_than_sequential(self, ground_truth, segmenter, parser):
        """
        Parallel asyncio.gather() extraction must be faster than sequential.
        Documents with ≥3 statement types are used for this test.
        """
        import asyncio

        doc = next(
            (d for d in ground_truth["documents"]
             if (FIXTURES_DIR / "pdfs" / d["file"]).exists()),
            None
        )
        if doc is None:
            pytest.skip("No fixture PDFs available")

        pdf_path = str(FIXTURES_DIR / "pdfs" / doc["file"])
        tables = parser.extract_tables(pdf_path)
        structure = segmenter.segment(tables, pdf_path)

        extractor_parallel = StatementExtractors(mode="parallel")
        extractor_sequential = StatementExtractors(mode="sequential")

        t0 = time.perf_counter()
        asyncio.run(extractor_parallel.run_all_async(structure))
        parallel_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        asyncio.run(extractor_sequential.run_all_async(structure))
        sequential_time = time.perf_counter() - t0

        assert parallel_time < sequential_time, (
            f"Parallel ({parallel_time:.2f}s) was not faster than sequential ({sequential_time:.2f}s)"
        )
        speedup = sequential_time / parallel_time
        print(f"\nParallel speedup: {speedup:.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: REGRESSION EVALS (NEW PIPELINE ≥ LEGACY)
# ══════════════════════════════════════════════════════════════════════════════

class TestRegression:
    """
    Ensure new segmented pipeline is strictly better than the legacy
    one-big-prompt approach across the accuracy dimensions.
    """

    @pytest.mark.regression
    def test_new_pipeline_equal_or_better_accuracy(self, ground_truth, segmenter, extractor, parser):
        """
        Across all ground truth metrics, new pipeline must not be LESS accurate
        than legacy. Measured by count of correctly extracted values (within tolerance).
        """
        new_correct = 0
        legacy_correct = 0
        total = 0

        for doc in ground_truth["documents"]:
            pdf_path = str(FIXTURES_DIR / "pdfs" / doc["file"])
            if not (FIXTURES_DIR / "pdfs" / doc["file"]).exists():
                continue

            # New pipeline
            tables = parser.extract_tables(pdf_path)
            structure = segmenter.segment(tables, pdf_path)
            new_results = extractor.run_all(structure)

            # Legacy pipeline
            legacy_results = extract_combined(pdf_path)

            for gt_metric in doc["metrics"]:
                total += 1
                new_val = _find_extracted_value(
                    new_results, gt_metric["statement_type"],
                    gt_metric["period"], gt_metric["label"]
                )
                legacy_val = _find_extracted_value(
                    legacy_results, gt_metric["statement_type"],
                    gt_metric["period"], gt_metric["label"]
                )
                ev = gt_metric["expected_value"]

                if new_val is not None and abs(new_val - ev) / ev < VALUE_TOLERANCE_PCT:
                    new_correct += 1
                if legacy_val is not None and abs(legacy_val - ev) / ev < VALUE_TOLERANCE_PCT:
                    legacy_correct += 1

        if total == 0:
            pytest.skip("No fixture PDFs available")

        new_acc = new_correct / total
        legacy_acc = legacy_correct / total

        print(f"\nNew pipeline accuracy: {new_acc*100:.1f}% | Legacy: {legacy_acc*100:.1f}%")
        assert new_acc >= legacy_acc, (
            f"New pipeline ({new_acc*100:.1f}%) is less accurate than legacy ({legacy_acc*100:.1f}%)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _find_extracted_value(results: dict, statement_type: str, period: str, label: str) -> float | None:
    item = _find_extracted_item(results, statement_type, period, label)
    return item["value"] if item else None


def _find_extracted_item(results: dict, statement_type: str, period: str, label: str) -> dict | None:
    items = results.get(statement_type, {}).get("items", [])
    label_lower = label.lower()
    for item in items:
        if item.get("period") == period and item.get("label", "").lower() == label_lower:
            return item
    return None


def _get_wrong_period(correct_period: str) -> str:
    """Return a plausible wrong period for the given correct period."""
    if correct_period.startswith("Q4"):
        return correct_period.replace("Q4", "FY")
    if correct_period.startswith("FY"):
        return correct_period.replace("FY", "Q4")
    return correct_period + "_WRONG"


def _collect_all_values(results: dict) -> list[float]:
    values = []
    for key, block in results.items():
        if key.startswith("_"):
            continue
        if isinstance(block, dict):
            for item in block.get("items", []):
                if isinstance(item.get("value"), (int, float)):
                    values.append(item["value"])
    return values


def _build_synthetic_results(fy_revenue: float, quarterly_revenues: dict) -> dict:
    """Build minimal results dict for reconciliation tests."""
    items = [{"label": "Revenue", "value": fy_revenue, "period": "FY 2025"}]
    for q, v in quarterly_revenues.items():
        items.append({"label": "Revenue", "value": v, "period": f"{q} 2025"})
    return {"income_statement": {"items": items}}


def _build_synthetic_results_with_segments(
    consolidated_revenue: float, segment_revenues: dict
) -> dict:
    items = [{"label": "Revenue", "value": consolidated_revenue, "period": "Q4 2025"}]
    segments = []
    for seg, v in segment_revenues.items():
        segments.append({"label": "Revenue", "value": v, "period": "Q4 2025", "segment": seg})
    return {"income_statement": {"items": items}, "segments": segments}


def _build_synthetic_results_bs(total_assets: float, total_liab_equity: float) -> dict:
    items = [
        {"label": "Total Assets", "value": total_assets, "period": "Dec 2025"},
        {"label": "Total Liabilities and Equity", "value": total_liab_equity, "period": "Dec 2025"},
    ]
    return {"balance_sheet": {"items": items}}
