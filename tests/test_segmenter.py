"""
Tests for financial statement segmenter, extractors, and reconciler.
No DB or LLM required — pure unit tests.
"""

import pytest

from services.financial_statement_segmenter import (
    StatementType,
    FinancialTable,
    FinancialDocumentStructure,
    classify_table,
    parse_period_label,
    parse_financial_value,
    detect_units,
    extract_periods_from_headers,
    split_by_period,
    segment_document,
    _extract_month,
    _month_to_quarter,
)
from services.extraction_reconciler import (
    reconcile_extractions,
    get_metric,
    get_all_segment_metrics,
)


# ─────────────────────────────────────────────────────────────────
# classify_table
# ─────────────────────────────────────────────────────────────────

class TestClassifyTable:
    def test_income_statement(self):
        table = [
            ["", "Q4 2025", "Q4 2024"],
            ["Revenue", "8,234", "7,500"],
            ["Cost of Sales", "(5,100)", "(4,800)"],
            ["Gross Profit", "3,134", "2,700"],
            ["Operating Profit", "1,200", "1,050"],
            ["Net Income", "900", "780"],
        ]
        result = classify_table(table, "", 1)
        assert result == StatementType.INCOME_STATEMENT

    def test_balance_sheet(self):
        table = [
            ["", "Dec 2025", "Dec 2024"],
            ["Total Assets", "50,000", "48,000"],
            ["Current Assets", "12,000", "11,000"],
            ["Total Liabilities", "30,000", "29,000"],
            ["Shareholders Equity", "20,000", "19,000"],
        ]
        result = classify_table(table, "", 2)
        assert result == StatementType.BALANCE_SHEET

    def test_cash_flow(self):
        table = [
            ["", "FY 2025", "FY 2024"],
            ["Operating Cash Flow", "2,500", "2,200"],
            ["Capital Expenditure", "(800)", "(700)"],
            ["Free Cash Flow", "1,700", "1,500"],
            ["Investing Activities", "(1,200)", "(1,000)"],
            ["Financing Activities", "(500)", "(400)"],
            ["Dividends Paid", "(300)", "(250)"],
        ]
        result = classify_table(table, "", 3)
        assert result == StatementType.CASH_FLOW

    def test_segment_from_context(self):
        table = [
            ["Region", "Revenue", "EBIT"],
            ["Europe", "3,000", "500"],
            ["Americas", "4,000", "700"],
        ]
        result = classify_table(table, "Revenue by segment", 4)
        assert result == StatementType.SEGMENT_BREAKDOWN

    def test_unknown_for_sparse_table(self):
        table = [
            ["Item", "Value"],
            ["Foo", "123"],
        ]
        result = classify_table(table, "", 1)
        assert result == StatementType.UNKNOWN

    def test_empty_table(self):
        result = classify_table([], "", 1)
        assert result == StatementType.UNKNOWN

    def test_page_text_helps_classification(self):
        """Page text headings should help classify ambiguous tables."""
        table = [
            ["", "2025", "2024"],
            ["Item A", "10,000", "9,000"],
            ["Item B", "5,000", "4,500"],
        ]
        result = classify_table(table, "Consolidated Statement of Cash Flows\nfor the year ended December 2025", 1)
        assert result == StatementType.CASH_FLOW


# ─────────────────────────────────────────────────────────────────
# parse_period_label
# ─────────────────────────────────────────────────────────────────

class TestParsePeriodLabel:
    def test_q4_2025(self):
        r = parse_period_label("Q4 2025")
        assert r == {"label": "Q4 2025", "type": "quarter"}

    def test_4q25(self):
        r = parse_period_label("4Q25")
        assert r == {"label": "Q4 2025", "type": "quarter"}

    def test_4q_2025(self):
        r = parse_period_label("4Q 2025")
        assert r == {"label": "Q4 2025", "type": "quarter"}

    def test_q1_25(self):
        r = parse_period_label("Q1 25")
        assert r == {"label": "Q1 2025", "type": "quarter"}

    def test_three_months_ended(self):
        r = parse_period_label("Three months ended Dec 31, 2025")
        assert r is not None
        assert r["label"] == "Q4 2025"
        assert r["type"] == "quarter"

    def test_six_months_ended(self):
        r = parse_period_label("Six months ended June 30, 2025")
        assert r is not None
        assert r["label"] == "H1 2025"
        assert r["type"] == "half"

    def test_year_ended(self):
        r = parse_period_label("Year ended December 2025")
        assert r == {"label": "FY 2025", "type": "annual"}

    def test_fy2025(self):
        r = parse_period_label("FY2025")
        assert r == {"label": "FY 2025", "type": "annual"}

    def test_fy_2025(self):
        r = parse_period_label("FY 2025")
        assert r == {"label": "FY 2025", "type": "annual"}

    def test_h1_2025(self):
        r = parse_period_label("H1 2025")
        assert r == {"label": "H1 2025", "type": "half"}

    def test_h2_2025(self):
        r = parse_period_label("H2 2025")
        assert r == {"label": "H2 2025", "type": "half"}

    def test_as_at_dec_2025(self):
        r = parse_period_label("As at 31 December 2025")
        assert r is not None
        assert r["label"] == "Q4 2025"
        assert r["type"] == "point_in_time"

    def test_plain_year(self):
        r = parse_period_label("2025")
        assert r == {"label": "FY 2025", "type": "annual"}

    def test_none_input(self):
        assert parse_period_label(None) is None

    def test_empty_string(self):
        assert parse_period_label("") is None

    def test_garbage(self):
        assert parse_period_label("hello world") is None


# ─────────────────────────────────────────────────────────────────
# parse_financial_value
# ─────────────────────────────────────────────────────────────────

class TestParseFinancialValue:
    def test_simple_number(self):
        assert parse_financial_value("1234") == 1234.0

    def test_comma_thousands(self):
        assert parse_financial_value("1,234.5") == 1234.5

    def test_parenthetical_negative(self):
        assert parse_financial_value("(1,234)") == -1234.0

    def test_dash_is_none(self):
        assert parse_financial_value("—") is None
        assert parse_financial_value("-") is None
        assert parse_financial_value("–") is None

    def test_na_is_none(self):
        assert parse_financial_value("n/a") is None
        assert parse_financial_value("N/A") is None

    def test_nm_is_none(self):
        assert parse_financial_value("nm") is None
        assert parse_financial_value("NM") is None

    def test_none_input(self):
        assert parse_financial_value(None) is None

    def test_empty_string(self):
        assert parse_financial_value("") is None

    def test_currency_symbol_stripped(self):
        assert parse_financial_value("$1,234") == 1234.0
        assert parse_financial_value("€500") == 500.0
        assert parse_financial_value("£2,000") == 2000.0

    def test_negative_number(self):
        assert parse_financial_value("-500") == -500.0

    def test_decimal(self):
        assert parse_financial_value("3.14") == 3.14

    def test_large_parenthetical(self):
        assert parse_financial_value("(12,345,678)") == -12345678.0


# ─────────────────────────────────────────────────────────────────
# detect_units
# ─────────────────────────────────────────────────────────────────

class TestDetectUnits:
    def test_eur_millions(self):
        table = [["€ millions", "2025", "2024"]]
        currency, scale = detect_units(table, "")
        assert currency == "EUR"
        assert scale == "millions"

    def test_gbp_thousands(self):
        table = [["£'000", "2025"]]
        currency, scale = detect_units(table, "in thousands of pounds sterling")
        assert currency == "GBP"
        assert scale == "thousands"

    def test_usd_default(self):
        currency, scale = detect_units([[]], "some random text")
        assert currency == "USD"
        assert scale == "millions"

    def test_jpy_billions(self):
        table = [["¥ billions", "2025"]]
        currency, scale = detect_units(table, "")
        assert currency == "JPY"
        assert scale == "billions"

    def test_chf(self):
        table = [["CHF millions", "2025"]]
        currency, scale = detect_units(table, "")
        assert currency == "CHF"
        assert scale == "millions"

    def test_krw(self):
        currency, scale = detect_units([[]], "In KRW millions")
        assert currency == "KRW"
        assert scale == "millions"

    def test_sek(self):
        currency, scale = detect_units([[]], "SEK thousands")
        assert currency == "SEK"
        assert scale == "thousands"

    def test_page_text_override(self):
        """Page text should contribute to detection."""
        table = [["", "2025"]]
        currency, scale = detect_units(table, "All amounts in EUR billions")
        assert currency == "EUR"
        assert scale == "billions"


# ─────────────────────────────────────────────────────────────────
# extract_periods_from_headers + split_by_period
# ─────────────────────────────────────────────────────────────────

class TestExtractPeriodsAndSplit:
    def test_basic_period_extraction(self):
        table = [
            ["", "Q4 2025", "Q4 2024"],
            ["Revenue", "8,234", "7,500"],
        ]
        periods = extract_periods_from_headers(table)
        assert len(periods) == 2
        assert periods[0]["period"] == "Q4 2025"
        assert periods[1]["period"] == "Q4 2024"
        # Q4 2025 should be is_current
        current = [p for p in periods if p["is_current"]]
        assert len(current) == 1
        assert current[0]["period"] == "Q4 2025"

    def test_split_by_period(self):
        table = [
            ["", "Q4 2025", "Q4 2024"],
            ["Revenue", "8,234", "7,500"],
            ["Operating Profit", "1,200", "1,050"],
            ["Net Income", "900", "780"],
        ]
        periods = extract_periods_from_headers(table)
        tables = split_by_period(table, periods, StatementType.INCOME_STATEMENT)
        assert len(tables) == 2

        current = [t for t in tables if t.is_current][0]
        assert current.period == "Q4 2025"
        assert current.statement_type == StatementType.INCOME_STATEMENT
        assert len(current.rows) == 3
        assert current.rows[0]["label"] == "Revenue"
        assert current.rows[0]["value"] == 8234.0

        prior = [t for t in tables if not t.is_current][0]
        assert prior.period == "Q4 2024"
        assert prior.rows[0]["value"] == 7500.0

    def test_fy_periods(self):
        table = [
            ["", "FY 2025", "FY 2024"],
            ["Revenue", "30,000", "28,000"],
        ]
        periods = extract_periods_from_headers(table)
        assert len(periods) == 2
        tables = split_by_period(table, periods, StatementType.INCOME_STATEMENT)
        assert len(tables) == 2
        assert tables[0].period_type == "annual" or tables[1].period_type == "annual"

    def test_no_periods(self):
        table = [
            ["Item", "Amount"],
            ["Something", "100"],
        ]
        periods = extract_periods_from_headers(table)
        assert len(periods) == 0


# ─────────────────────────────────────────────────────────────────
# segment_document (integration-level)
# ─────────────────────────────────────────────────────────────────

class TestSegmentDocument:
    def test_basic_segmentation(self):
        pages = [
            {"page_num": 1, "text": "Consolidated Income Statement\nIn USD millions"},
            {"page_num": 2, "text": "Management discussion and analysis of results..."},
        ]
        tables_by_page = {
            1: [[
                ["", "Q4 2025", "Q4 2024"],
                ["Revenue", "8,234", "7,500"],
                ["Net Income", "900", "780"],
            ]],
        }
        structure = segment_document(pages, tables_by_page)
        assert len(structure.tables) == 2  # one per period
        assert len(structure.narrative_sections) == 1  # page 2
        assert structure.tables[0].statement_type == StatementType.INCOME_STATEMENT

    def test_empty_document(self):
        structure = segment_document([], {})
        assert len(structure.tables) == 0
        assert len(structure.narrative_sections) == 0

    def test_footnote_detection(self):
        pages = [
            {"page_num": 1, "text": "Note 1: Summary of significant accounting policies..."},
        ]
        structure = segment_document(pages, {})
        assert len(structure.footnotes) == 1
        assert len(structure.narrative_sections) == 0


# ─────────────────────────────────────────────────────────────────
# Month helpers
# ─────────────────────────────────────────────────────────────────

class TestMonthHelpers:
    def test_extract_month_names(self):
        assert _extract_month("December 31, 2025") == 12
        assert _extract_month("Jun 30") == 6
        assert _extract_month("March") == 3
        assert _extract_month("sept 2025") == 9

    def test_month_to_quarter(self):
        assert _month_to_quarter(1) == "Q1"
        assert _month_to_quarter(3) == "Q1"
        assert _month_to_quarter(6) == "Q2"
        assert _month_to_quarter(9) == "Q3"
        assert _month_to_quarter(12) == "Q4"


# ─────────────────────────────────────────────────────────────────
# reconcile_extractions
# ─────────────────────────────────────────────────────────────────

class TestReconcileExtractions:
    def _make_results(
        self,
        fy_rev=30000, q1_rev=7000, q2_rev=7500, q3_rev=7500, q4_rev=8000,
        assets=50000, liabilities=30000, equity=20000,
        pl_ni=900, cf_ni=900,
    ):
        """Helper to build a consistent results dict."""
        return {
            "income_statements": [
                {"period": "FY 2025", "data": {"revenue": fy_rev, "net income": pl_ni}, "segment": None},
                {"period": "Q1 2025", "data": {"revenue": q1_rev, "net income": 200}, "segment": None},
                {"period": "Q2 2025", "data": {"revenue": q2_rev, "net income": 220}, "segment": None},
                {"period": "Q3 2025", "data": {"revenue": q3_rev, "net income": 230}, "segment": None},
                {"period": "Q4 2025", "data": {"revenue": q4_rev, "net income": 250}, "segment": None},
            ],
            "balance_sheets": [
                {"period": "Q4 2025", "data": {
                    "total assets": assets,
                    "total liabilities": liabilities,
                    "shareholders equity": equity,
                }},
            ],
            "cash_flows": [
                {"period": "FY 2025", "data": {"net income": cf_ni, "operating cash flow": 2500}},
            ],
            "segments": [],
            "notes": [],
            "narratives": [],
        }

    def test_all_consistent(self):
        results = self._make_results()
        reconciled = reconcile_extractions(results)
        assert reconciled["passed"] is True
        assert reconciled["checks_run"] > 0
        assert reconciled["checks_passed"] == reconciled["checks_run"]
        assert len(reconciled["issues"]) == 0

    def test_quarterly_sum_mismatch(self):
        # Q sum = 7000+7500+7500+8000 = 30000, FY = 25000 → mismatch
        results = self._make_results(fy_rev=25000)
        reconciled = reconcile_extractions(results)
        assert reconciled["passed"] is False
        revenue_issues = [
            i for i in reconciled["issues"]
            if "quarterly_sum" in i["check"] and "revenue" in i["check"]
        ]
        assert len(revenue_issues) >= 1

    def test_balance_sheet_mismatch(self):
        # Assets=50000, Liabilities=30000, Equity=15000 → L+E=45000 ≠ 50000
        results = self._make_results(equity=15000)
        reconciled = reconcile_extractions(results)
        assert reconciled["passed"] is False
        bs_issues = [i for i in reconciled["issues"] if i["check"] == "balance_sheet_equation"]
        assert len(bs_issues) >= 1
        assert bs_issues[0]["severity"] == "critical"

    def test_net_income_pl_vs_cf_mismatch(self):
        # P&L NI=900, CF NI=700 → mismatch
        results = self._make_results(pl_ni=900, cf_ni=700)
        reconciled = reconcile_extractions(results)
        assert reconciled["passed"] is False
        ni_issues = [i for i in reconciled["issues"] if i["check"] == "net_income_pl_vs_cf"]
        assert len(ni_issues) >= 1

    def test_segment_sum_check(self):
        results = self._make_results()
        results["segments"] = [
            {"segment": "Europe", "data": {"revenue": 12000}},
            {"segment": "Americas", "data": {"revenue": 10000}},
            {"segment": "Asia", "data": {"revenue": 5000}},
        ]
        # Segment sum = 27000, consolidated FY = 30000 → 10% off
        reconciled = reconcile_extractions(results)
        seg_issues = [i for i in reconciled["issues"] if "segment" in i["check"]]
        assert len(seg_issues) >= 1

    def test_empty_results(self):
        results = {
            "income_statements": [],
            "balance_sheets": [],
            "cash_flows": [],
            "segments": [],
            "notes": [],
            "narratives": [],
        }
        reconciled = reconcile_extractions(results)
        assert reconciled["passed"] is True
        assert len(reconciled["issues"]) == 0


# ─────────────────────────────────────────────────────────────────
# get_metric / get_all_segment_metrics
# ─────────────────────────────────────────────────────────────────

class TestGetMetric:
    def test_basic_get(self):
        results = {
            "income_statements": [
                {"period": "FY 2025", "data": {"revenue": 30000}},
            ],
        }
        assert get_metric(results, "income_statements", "FY 2025", "revenue") == 30000.0

    def test_alias_matching(self):
        results = {
            "income_statements": [
                {"period": "FY 2025", "data": {"net sales": 30000}},
            ],
        }
        # "net sales" is a revenue alias
        assert get_metric(results, "income_statements", "FY 2025", "revenue") == 30000.0

    def test_not_found(self):
        results = {"income_statements": []}
        assert get_metric(results, "income_statements", "FY 2025", "revenue") is None

    def test_segment_metrics(self):
        results = {
            "segments": [
                {"segment": "Europe", "data": {"revenue": 12000}},
                {"segment": "Americas", "data": {"revenue": 10000}},
            ],
        }
        segs = get_all_segment_metrics(results, "revenue")
        assert segs == {"Europe": 12000.0, "Americas": 10000.0}


# ─────────────────────────────────────────────────────────────────
# _build_reconciler_input — wiring between two-pass output and reconciler
# ─────────────────────────────────────────────────────────────────

class TestBuildReconcilerInput:
    def _mk_table(self, stmt_type, period, segment=None):
        return FinancialTable(
            statement_type=stmt_type,
            period=period,
            period_type="quarter" if period.startswith("Q") else "annual",
            segment=segment,
            currency="USD",
            unit_scale="millions",
            rows=[],
        )

    def test_groups_by_statement_type(self):
        from services.metric_extractor import _build_reconciler_input

        structure = FinancialDocumentStructure(tables=[
            self._mk_table(StatementType.INCOME_STATEMENT, "FY 2025"),
            self._mk_table(StatementType.BALANCE_SHEET, "Dec 2025"),
            self._mk_table(StatementType.CASH_FLOW, "FY 2025"),
        ])
        results = [
            [{"metric": "Revenue", "value": 1000.0, "original_label": "Revenue"},
             {"metric": "Net Income", "value": 100.0, "original_label": "Net Income"}],
            [{"metric": "Total Assets", "value": 5000.0, "original_label": "Total Assets"},
             {"metric": "Total Liabilities", "value": 3000.0, "original_label": "Total Liabilities"},
             {"metric": "Shareholders Equity", "value": 2000.0, "original_label": "Shareholders Equity"}],
            [{"metric": "Net Income", "value": 100.0, "original_label": "Net Income"}],
        ]
        out = _build_reconciler_input(structure, results)
        assert len(out["income_statements"]) == 1
        assert out["income_statements"][0]["period"] == "FY 2025"
        assert out["income_statements"][0]["data"]["Revenue"] == 1000.0
        assert len(out["balance_sheets"]) == 1
        assert out["balance_sheets"][0]["data"]["Total Assets"] == 5000.0
        assert len(out["cash_flows"]) == 1

    def test_balance_sheet_mismatch_flagged(self):
        """End-to-end: build reconciler input from two-pass shape, run reconciler,
        confirm BS equation failure is flagged."""
        from services.metric_extractor import _build_reconciler_input

        structure = FinancialDocumentStructure(tables=[
            self._mk_table(StatementType.BALANCE_SHEET, "Dec 2025"),
        ])
        # 5000 ≠ 3000 + 1500 (off by 500 = 10%)
        results = [
            [
                {"metric": "Total Assets", "value": 5000.0, "original_label": "Total Assets"},
                {"metric": "Total Liabilities", "value": 3000.0, "original_label": "Total Liabilities"},
                {"metric": "Shareholders Equity", "value": 1500.0, "original_label": "Equity"},
            ],
        ]
        recon_input = _build_reconciler_input(structure, results)
        report = reconcile_extractions(recon_input)
        assert report["passed"] is False
        bs_issues = [i for i in report["issues"] if "balance_sheet" in i.get("check", "")]
        assert bs_issues, "expected balance sheet equation failure to be flagged"

    def test_segment_tables_populate_segments_bucket(self):
        from services.metric_extractor import _build_reconciler_input

        structure = FinancialDocumentStructure(tables=[
            self._mk_table(StatementType.SEGMENT_BREAKDOWN, "FY 2025", segment="Europe"),
            self._mk_table(StatementType.SEGMENT_BREAKDOWN, "FY 2025", segment="Americas"),
        ])
        results = [
            [{"metric": "Revenue", "value": 600.0, "original_label": "Revenue"}],
            [{"metric": "Revenue", "value": 400.0, "original_label": "Revenue"}],
        ]
        out = _build_reconciler_input(structure, results)
        assert len(out["segments"]) == 2
        segs = {e["segment"]: e["data"]["Revenue"] for e in out["segments"]}
        assert segs == {"Europe": 600.0, "Americas": 400.0}
