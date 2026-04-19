"""Tests for agents/ingestion/document_triage.py — DocumentTriageAgent
is deliberately NOT registered with AgentRegistry (it's an ingestion-
time classifier, not a pipeline agent). Tests verify the output-shape
guarantees the dispatcher relies on."""

import json

import pytest

from agents.base import AgentTier
from agents.ingestion.document_triage import DocumentTriageAgent
from agents.registry import AgentRegistry


class TestRegistration:
    def test_not_in_registry(self):
        """DocumentTriageAgent must NOT be auto-registered — the dispatcher
        instantiates it directly. Registering it would pull it into the
        analysis pipeline, which would be wrong."""
        AgentRegistry.autodiscover()
        registered = AgentRegistry.get_all()
        assert "document_triage" not in registered

    def test_tier_is_meta(self):
        assert DocumentTriageAgent.tier == AgentTier.META


class TestShouldRun:
    def test_runs_with_source_url(self):
        agent = DocumentTriageAgent()
        assert agent.should_run({"source_url": "https://sec.gov/cgi-bin/browse-edgar"}) is True

    def test_skips_without_source_url(self):
        agent = DocumentTriageAgent()
        assert agent.should_run({}) is False
        assert agent.should_run({"source_url": ""}) is False
        assert agent.should_run({"source_url": None}) is False


class TestValidateOutput:
    def setup_method(self):
        self.agent = DocumentTriageAgent()

    def test_parses_plain_json(self):
        raw = json.dumps({
            "document_type": "10-Q",
            "period_label": "2026_Q1",
            "priority": "normal",
            "relevance_score": 70,
            "auto_ingest": True,
            "needs_review": False,
            "rationale": "routine filing",
            "confidence": 0.9,
        })
        out = self.agent.validate_output(raw)
        assert out["document_type"] == "10-Q"
        assert out["priority"] == "normal"
        assert out["auto_ingest"] is True

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "document_type": "annual_report",
            "period_label": "2025_FY",
            "priority": "normal",
            "auto_ingest": True,
        }) + "\n```"
        out = self.agent.validate_output(raw)
        assert out["document_type"] == "annual_report"

    def test_missing_required_field_raises(self):
        # auto_ingest is required per output_schema
        raw = json.dumps({
            "document_type": "10-Q",
            "priority": "normal",
            # auto_ingest absent
        })
        with pytest.raises(ValueError, match="missing required fields"):
            self.agent.validate_output(raw)

    def test_none_period_label_coerced_to_empty_string(self):
        raw = json.dumps({
            "document_type": "8-K",
            "period_label": None,
            "priority": "normal",
            "auto_ingest": True,
        })
        out = self.agent.validate_output(raw)
        assert out["period_label"] == ""

    def test_bad_priority_defaults_to_normal(self):
        raw = json.dumps({
            "document_type": "10-Q",
            "period_label": "2026_Q1",
            "priority": "urgent-high-super",  # not in allowed set
            "auto_ingest": True,
        })
        out = self.agent.validate_output(raw)
        assert out["priority"] == "normal"

    def test_allowed_priorities_pass_through(self):
        for p in ("immediate", "normal", "low", "skip"):
            raw = json.dumps({
                "document_type": "10-Q",
                "priority": p,
                "auto_ingest": True,
            })
            out = self.agent.validate_output(raw)
            assert out["priority"] == p

    def test_bad_relevance_score_defaults_to_50(self):
        raw = json.dumps({
            "document_type": "10-Q",
            "priority": "normal",
            "auto_ingest": True,
            "relevance_score": "not a number",
        })
        out = self.agent.validate_output(raw)
        assert out["relevance_score"] == 50

    def test_auto_ingest_coerced_to_bool(self):
        raw = json.dumps({
            "document_type": "10-Q",
            "priority": "normal",
            "auto_ingest": 1,       # truthy non-bool
            "needs_review": 0,      # falsy non-bool
        })
        out = self.agent.validate_output(raw)
        assert out["auto_ingest"] is True
        assert out["needs_review"] is False
