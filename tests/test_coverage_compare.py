"""Tests for services/harvester/coverage_compare.py — pure classifier
tests against fabricated old/new inputs. Verifies agreement rules,
warning-severity filtering, and the summary counts."""

from services.harvester.coverage_compare import (
    CompanyParity,
    _classify,
    _ACTIVE_SEVERITIES,
)


def _old_row(ticker: str, gap: str, **kwargs) -> dict:
    """Build a fake row in the shape check_coverage() returns."""
    return {
        "ticker": ticker,
        "name": ticker,
        "gap": gap,
        "latest_period": kwargs.get("latest_period", "2026_Q1"),
        "expected_period": kwargs.get("expected_period", "2026_Q1"),
        "quarters_behind": kwargs.get("quarters_behind", 0),
        "has_earnings": kwargs.get("has_earnings", True),
        "has_annual": kwargs.get("has_annual", False),
    }


def _new_gap(doc_type: str, severity: str) -> dict:
    """Build a fake gap in the shape gap_to_dict returns."""
    return {
        "doc_type": doc_type,
        "severity": severity,
        "ticker": "X",
        "name": "X",
        "expected_period": "2026_Q1",
        "expected_by": "2026-04-01",
        "days_overdue": 3,
        "reason": "",
    }


class TestAgreement:
    def test_both_clean_agree(self):
        row = _classify(_old_row("LKQ US", "ok"), [])
        assert row.agreement == "agree"
        assert row.disagree_type is None

    def test_both_flagging_agree_even_if_different_reasons(self):
        old = _old_row("ASML NA", "behind", quarters_behind=1, latest_period="2025_Q4")
        new = [_new_gap("transcript", "overdue")]
        row = _classify(old, new)
        assert row.agreement == "agree"

    def test_old_clean_new_flagging_disagrees(self):
        old = _old_row("HEIA NA", "ok")
        new = [_new_gap("transcript", "critical")]
        row = _classify(old, new)
        assert row.agreement == "disagree"
        assert row.disagree_type == "new_flagged_old_clean"
        assert "new flags" in row.reason.lower() or "new flags" in row.reason

    def test_old_flagging_new_clean_disagrees(self):
        old = _old_row("BUNZL LN", "behind", quarters_behind=1, latest_period="2025_Q4")
        new: list[dict] = []  # new system finds no overdue/critical gaps
        row = _classify(old, new)
        assert row.agreement == "disagree"
        assert row.disagree_type == "old_flagged_new_clean"
        assert "1Q behind" in row.reason or "latest=2025_Q4" in row.reason


class TestWarningSeverityIgnored:
    def test_warning_gaps_dont_count_as_new_flagging(self):
        """Warning = approaching expected date but not yet overdue.
        Old system has no equivalent — treat as new-clean for parity."""
        old = _old_row("ASML NA", "ok")
        new = [_new_gap("transcript", "warning")]
        row = _classify(old, new)
        assert row.agreement == "agree"
        assert row.new_active_gap_count == 0

    def test_warning_alongside_overdue_only_overdue_counts(self):
        old = _old_row("ASML NA", "ok")
        new = [
            _new_gap("transcript",   "warning"),   # ignored
            _new_gap("presentation", "overdue"),   # counts
        ]
        row = _classify(old, new)
        assert row.agreement == "disagree"
        assert row.disagree_type == "new_flagged_old_clean"
        assert row.new_active_gap_count == 1
        assert row.new_gap_severities == ["overdue"]

    def test_source_broken_counts_as_active(self):
        """source_broken DOES count — it's a serious gap the old system
        would also flag (as no_docs or missing)."""
        old = _old_row("DEAD LN", "no_docs", latest_period=None)
        new = [_new_gap("earnings_release", "source_broken")]
        row = _classify(old, new)
        assert row.agreement == "agree"
        assert "source_broken" in _ACTIVE_SEVERITIES
        assert row.new_active_gap_count == 1


class TestOldGapVariants:
    def test_missing_old_gap_also_flagging(self):
        """Old system has 4 gap values: ok | behind | missing | no_docs.
        Only 'ok' is clean — the other three should all count as old-flagging."""
        for gap_val in ("behind", "missing", "no_docs"):
            old = _old_row("X", gap_val)
            new = [_new_gap("transcript", "overdue")]
            row = _classify(old, new)
            assert row.agreement == "agree", f"gap={gap_val} should agree with overdue new"

    def test_no_docs_old_clean_new_disagrees(self):
        """Safety check: no_docs + no active new gaps = disagree (old
        flagging, new not). Would happen with a newly-added company that
        has zero history."""
        old = _old_row("NEW CO", "no_docs", latest_period=None)
        new: list[dict] = []
        row = _classify(old, new)
        assert row.agreement == "disagree"
        assert row.disagree_type == "old_flagged_new_clean"
