"""Tests for agents/specialist/guidance_tracker.py — a registered
SPECIALIST-tier pipeline agent. Covers should_run gating, validate_output
enforcement, and extract_predictions shape."""

import json

import pytest

from agents.base import AgentTier
from agents.registry import AgentRegistry
from agents.specialist.guidance_tracker import GuidanceTrackerAgent


class TestRegistration:
    def test_has_register_decorator(self):
        """Unlike DocumentTriage, guidance_tracker IS a pipeline agent.
        Register it directly in case the suite's other tests cleared the
        registry (autodiscover is cached by Python's module import system
        and won't re-run @AgentRegistry.register after a clear())."""
        AgentRegistry.clear()
        AgentRegistry.register(GuidanceTrackerAgent)
        agents = AgentRegistry.get_all()
        assert "guidance_tracker" in agents

    def test_tier_is_specialist(self):
        assert GuidanceTrackerAgent.tier == AgentTier.SPECIALIST

    def test_feeds_into_bear_bull_debate(self):
        """Predictions are consumed downstream — declare the edges."""
        feeds = set(GuidanceTrackerAgent.feeds_into)
        assert "bear_case" in feeds
        assert "bull_case" in feeds
        assert "debate_agent" in feeds


class TestShouldRun:
    def setup_method(self):
        self.agent = GuidanceTrackerAgent()

    def test_runs_with_current_guidance(self):
        assert self.agent.should_run({"guidance": "Rev growth 5-7%"}) is True

    def test_runs_with_prior_guidance_only(self):
        assert self.agent.should_run({"prior_guidance": "Rev growth 5%"}) is True

    def test_runs_with_transcript_guidance_only(self):
        assert self.agent.should_run({
            "transcript_deep_dive": {"guidance_statements": [{"metric": "rev", "direction": "up"}]}
        }) is True

    def test_skips_when_all_absent(self):
        assert self.agent.should_run({}) is False

    def test_skips_when_guidance_is_sentinel_only(self):
        """`build_guidance_summary` returns 'No guidance items found.' when
        the DB has nothing. Should treat that as empty."""
        assert self.agent.should_run({
            "guidance": "No guidance items found.",
            "prior_guidance": "No guidance items found.",
            "transcript_deep_dive": {},
        }) is False

    def test_skips_when_transcript_dict_has_empty_guidance_statements(self):
        assert self.agent.should_run({
            "transcript_deep_dive": {"guidance_statements": []},
        }) is False


class TestValidateOutput:
    def setup_method(self):
        self.agent = GuidanceTrackerAgent()

    def _minimal_valid_output(self) -> dict:
        return {
            "prior_guidance_scorecard": [],
            "current_guidance": [
                {"metric": "organic_revenue_growth",
                 "value_or_range": "5-7%",
                 "specificity": "range",
                 "direction_vs_prior": "unchanged"},
            ],
            "methodology_changes": [],
            "notable_walkbacks": [],
            "new_disclosures": [],
            "withdrawn_guidance": [],
            "track_record_signal": "strong",
            "overall_signal": "stable",
            "confidence": 0.85,
        }

    def test_valid_output_parses(self):
        raw = json.dumps(self._minimal_valid_output())
        out = self.agent.validate_output(raw)
        assert out["overall_signal"] == "stable"
        assert len(out["current_guidance"]) == 1

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(self._minimal_valid_output()) + "\n```"
        out = self.agent.validate_output(raw)
        assert out["track_record_signal"] == "strong"

    def test_missing_current_guidance_raises(self):
        data = self._minimal_valid_output()
        del data["current_guidance"]
        with pytest.raises(ValueError, match="missing required fields"):
            self.agent.validate_output(json.dumps(data))

    def test_missing_overall_signal_raises(self):
        data = self._minimal_valid_output()
        del data["overall_signal"]
        with pytest.raises(ValueError, match="missing required fields"):
            self.agent.validate_output(json.dumps(data))

    def test_non_list_current_guidance_raises(self):
        data = self._minimal_valid_output()
        data["current_guidance"] = "should be a list"
        with pytest.raises(ValueError, match="must be a list"):
            self.agent.validate_output(json.dumps(data))

    def test_non_list_walkbacks_raises(self):
        data = self._minimal_valid_output()
        data["notable_walkbacks"] = {"key": "value"}   # dict not list
        with pytest.raises(ValueError, match="must be a list"):
            self.agent.validate_output(json.dumps(data))


class TestExtractPredictions:
    def setup_method(self):
        self.agent = GuidanceTrackerAgent()

    def test_one_prediction_per_current_guidance_entry(self):
        output = {
            "current_guidance": [
                {"metric": "organic_growth", "value_or_range": "5-7%",
                 "specificity": "range", "direction_vs_prior": "tighter"},
                {"metric": "ebitda_margin", "value_or_range": "18-20%",
                 "specificity": "range", "direction_vs_prior": "unchanged"},
            ],
        }
        preds = self.agent.extract_predictions(output)
        assert len(preds) == 2
        assert preds[0]["metric"] == "organic_growth"
        assert preds[0]["horizon_days"] == 180
        assert preds[1]["direction"] == "unchanged"

    def test_entries_without_metric_skipped(self):
        output = {
            "current_guidance": [
                {"metric": "organic_growth", "direction_vs_prior": "tighter"},
                {"value_or_range": "orphan"},   # no metric — should be skipped
            ],
        }
        preds = self.agent.extract_predictions(output)
        assert len(preds) == 1
        assert preds[0]["metric"] == "organic_growth"

    def test_caps_at_ten_entries(self):
        output = {
            "current_guidance": [
                {"metric": f"kpi_{i}", "direction_vs_prior": "unchanged"}
                for i in range(20)
            ],
        }
        preds = self.agent.extract_predictions(output)
        assert len(preds) == 10

    def test_non_dict_entries_skipped(self):
        output = {
            "current_guidance": [
                {"metric": "rev_growth"},
                "some string that isn't a dict",
                None,
            ],
        }
        preds = self.agent.extract_predictions(output)
        assert len(preds) == 1

    def test_non_dict_output_returns_empty(self):
        assert self.agent.extract_predictions(None) == []
        assert self.agent.extract_predictions("oops") == []
        assert self.agent.extract_predictions([]) == []
