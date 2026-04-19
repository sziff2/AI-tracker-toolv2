"""Tests for services/harvester/scheduler.py::format_teams_message —
verifies the new Triage + Coverage Monitor sections render correctly
when data is present and are omitted cleanly when absent."""

from services.harvester.scheduler import format_teams_message


def _minimal_harvest_result() -> dict:
    return {
        "new": 3,
        "skipped": 1,
        "failed": 0,
        "details": [
            {"ticker": "ASML NA", "name": "ASML",
             "sources_tried": ["edgar"], "candidates_found": 2,
             "errors": []},
        ],
    }


def _body_text(payload: dict) -> str:
    """Pull the main text block out of the Teams Adaptive Card. The title
    block has no `wrap` property; the body block does."""
    for block in payload["body"]:
        if block.get("wrap") and block.get("text"):
            return block["text"]
    return ""


class TestBaselineReport:
    def test_no_extra_sections_when_stats_absent(self):
        body = _body_text(format_teams_message(_minimal_harvest_result()))
        assert "Triage:" not in body
        assert "Coverage Monitor:" not in body

    def test_summary_always_present(self):
        body = _body_text(format_teams_message(_minimal_harvest_result()))
        assert "3 new documents" in body
        assert "1 skipped" in body


class TestTriageSection:
    def test_renders_when_total_nonzero(self):
        result = _minimal_harvest_result()
        result["triage_stats"] = {
            "total": 12,
            "auto_ingested": 10,
            "needs_review": 1,
            "skipped_by_triage": 1,
        }
        body = _body_text(format_teams_message(result))
        assert "**Triage:** 12 candidates classified" in body
        assert "10 auto-ingested" in body
        assert "1 flagged for review" in body
        assert "1 dropped as junk" in body

    def test_omitted_when_zero_total(self):
        """If no triage decisions were made (e.g. harvest found no candidates),
        don't clutter the report with a 0-count section."""
        result = _minimal_harvest_result()
        result["triage_stats"] = {
            "total": 0,
            "auto_ingested": 0,
            "needs_review": 0,
            "skipped_by_triage": 0,
        }
        body = _body_text(format_teams_message(result))
        assert "Triage:" not in body

    def test_handles_missing_fields_gracefully(self):
        """Old-shape dict shouldn't crash the formatter."""
        result = _minimal_harvest_result()
        result["triage_stats"] = {"total": 5}  # only total, no breakdown
        body = _body_text(format_teams_message(result))
        assert "5 candidates classified" in body
        assert "0 auto-ingested" in body   # defaults to 0 when key absent


class TestCoverageMonitorSection:
    def test_renders_with_gaps_and_rescans(self):
        result = _minimal_harvest_result()
        result["coverage_monitor_stats"] = {
            "total_gaps": 4,
            "by_severity": {"critical": 1, "overdue": 2, "warning": 1, "source_broken": 0},
            "rescans": 2,
            "rescan_successes": 1,
            "rescan_errors": 0,
            "lookback_days": 7,
        }
        body = _body_text(format_teams_message(result))
        assert "**Coverage Monitor:** 4 gaps" in body
        assert "1 critical" in body
        assert "2 overdue" in body
        assert "2 auto-rescans in last 7d" in body
        assert "1 found new docs" in body

    def test_renders_zero_gaps_clean(self):
        """Unlike Triage, Coverage Monitor is always shown even with 0 gaps —
        that's a useful signal (everything up to date)."""
        result = _minimal_harvest_result()
        result["coverage_monitor_stats"] = {
            "total_gaps": 0,
            "by_severity": {"critical": 0, "overdue": 0, "warning": 0, "source_broken": 0},
            "rescans": 0,
            "rescan_successes": 0,
            "rescan_errors": 0,
            "lookback_days": 7,
        }
        body = _body_text(format_teams_message(result))
        assert "**Coverage Monitor:** 0 gaps" in body

    def test_omitted_when_dict_empty(self):
        result = _minimal_harvest_result()
        result["coverage_monitor_stats"] = {}
        body = _body_text(format_teams_message(result))
        assert "Coverage Monitor:" not in body

    def test_omitted_when_key_absent(self):
        result = _minimal_harvest_result()
        # coverage_monitor_stats key not present at all
        body = _body_text(format_teams_message(result))
        assert "Coverage Monitor:" not in body


class TestBothSectionsTogether:
    def test_triage_above_coverage(self):
        """Reading order: harvest summary → new docs → triage → coverage monitor →
        legacy coverage gaps. Triage must appear above the Coverage Monitor line."""
        result = _minimal_harvest_result()
        result["triage_stats"] = {
            "total": 8, "auto_ingested": 7, "needs_review": 0, "skipped_by_triage": 1,
        }
        result["coverage_monitor_stats"] = {
            "total_gaps": 2,
            "by_severity": {"critical": 0, "overdue": 2, "warning": 0, "source_broken": 0},
            "rescans": 1,
            "rescan_successes": 0,
            "rescan_errors": 0,
            "lookback_days": 7,
        }
        body = _body_text(format_teams_message(result))
        triage_idx = body.find("**Triage:**")
        cov_idx = body.find("**Coverage Monitor:**")
        assert triage_idx > 0
        assert cov_idx > 0
        assert triage_idx < cov_idx
