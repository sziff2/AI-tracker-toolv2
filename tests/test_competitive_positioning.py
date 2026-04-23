"""Tests for agents/industry/competitive_positioning.py — Tier 5.1.
Covers registration, tier, should_run gating, and validate_output shape."""

import json

import pytest

from agents.base import AgentTier
from agents.registry import AgentRegistry
from agents.industry.competitive_positioning import CompetitivePositioningAgent


class TestRegistration:
    def test_has_register_decorator(self):
        AgentRegistry.clear()
        AgentRegistry.register(CompetitivePositioningAgent)
        assert "competitive_positioning" in AgentRegistry.get_all()

    def test_tier_is_industry(self):
        assert CompetitivePositioningAgent.tier == AgentTier.INDUSTRY

    def test_depends_on_financial_analyst(self):
        assert "financial_analyst" in CompetitivePositioningAgent.depends_on

    def test_feeds_into_bear_bull_debate(self):
        feeds = set(CompetitivePositioningAgent.feeds_into)
        assert "bear_case" in feeds
        assert "bull_case" in feeds
        assert "debate_agent" in feeds


class TestShouldRun:
    def setup_method(self):
        self.agent = CompetitivePositioningAgent()

    def _base_inputs(self, peers=None, peer_metrics=None, subject_metrics=None):
        return {
            "peer_tickers":    peers or [],
            "peer_metrics":    peer_metrics or {},
            "subject_metrics": subject_metrics or [],
        }

    def test_skips_with_no_peers(self):
        assert self.agent.should_run(self._base_inputs()) is False

    def test_skips_when_no_subject_metrics(self):
        i = self._base_inputs(
            peers=["ACE US"],
            peer_metrics={"ACE US": [{"metric": "revenue", "value": 1}]},
            subject_metrics=[],
        )
        assert self.agent.should_run(i) is False

    def test_skips_when_all_peers_empty(self):
        i = self._base_inputs(
            peers=["ACE US", "TRV US"],
            peer_metrics={"ACE US": [], "TRV US": []},
            subject_metrics=[{"metric": "revenue", "value": 1}],
        )
        assert self.agent.should_run(i) is False

    def test_runs_with_one_peer_having_metrics(self):
        i = self._base_inputs(
            peers=["ACE US", "TRV US"],
            peer_metrics={"ACE US": [{"metric": "revenue", "value": 2}], "TRV US": []},
            subject_metrics=[{"metric": "revenue", "value": 1}],
        )
        assert self.agent.should_run(i) is True


class TestValidateOutput:
    def setup_method(self):
        self.agent = CompetitivePositioningAgent()

    def _minimal_valid_output(self) -> dict:
        return {
            "peers_analysed":    ["ACE US"],
            "overlap_metrics":   ["combined_ratio"],
            "leading_metrics":   [],
            "trailing_metrics":  [],
            "trend_vs_prior":    [],
            "thesis_implications": [],
            "data_gaps":         [],
            "overall_signal":    "stable",
            "confidence":        0.6,
            "sources": [
                {"kind": "extracted_metric", "metric_name": "combined_ratio"},
            ],
        }

    def test_valid_output_parses(self):
        raw = json.dumps(self._minimal_valid_output())
        out = self.agent.validate_output(raw)
        assert out["overall_signal"] == "stable"
        assert out["peers_analysed"] == ["ACE US"]

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(self._minimal_valid_output()) + "\n```"
        out = self.agent.validate_output(raw)
        assert out["confidence"] == 0.6

    def test_rejects_missing_required_fields(self):
        bad = {"peers_analysed": []}  # missing overall_signal
        with pytest.raises(ValueError, match="missing required fields"):
            self.agent.validate_output(json.dumps(bad))

    def test_rejects_non_list_leading_metrics(self):
        bad = self._minimal_valid_output()
        bad["leading_metrics"] = "not a list"
        with pytest.raises(ValueError, match="leading_metrics must be a list"):
            self.agent.validate_output(json.dumps(bad))

    def test_normalises_missing_sources_to_empty_list(self):
        """If the model forgets the sources array entirely, we normalise
        rather than hard-fail — mirrors the bear_case pattern from 4.4."""
        out_no_sources = self._minimal_valid_output()
        out_no_sources.pop("sources")
        parsed = self.agent.validate_output(json.dumps(out_no_sources))
        assert parsed["sources"] == []

    def test_normalises_non_list_sources_to_empty_list(self):
        out = self._minimal_valid_output()
        out["sources"] = "not a list"
        parsed = self.agent.validate_output(json.dumps(out))
        assert parsed["sources"] == []

    def test_insufficient_data_is_valid_signal(self):
        out = self._minimal_valid_output()
        out["overall_signal"] = "insufficient_data"
        parsed = self.agent.validate_output(json.dumps(out))
        assert parsed["overall_signal"] == "insufficient_data"
