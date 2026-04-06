"""Tests for agent base classes and registry."""
import pytest
from agents.base import BaseAgent, AgentResult, AgentTier
from agents.registry import AgentRegistry


class MockAgent(BaseAgent):
    agent_id = "test_mock"
    tier = AgentTier.EXTRACTION

    def build_prompt(self, inputs):
        return f"Test prompt for {inputs.get('ticker', 'unknown')}"

    def validate_output(self, raw):
        return {"parsed": raw}


class MockAnalysisAgent(BaseAgent):
    agent_id = "test_analysis"
    tier = AgentTier.ANALYSIS
    depends_on = ["test_mock"]

    def build_prompt(self, inputs):
        return "Analysis prompt"

    def validate_output(self, raw):
        return raw


class TestAgentResult:
    def test_default_values(self):
        r = AgentResult(agent_id="test")
        assert r.status == "completed"
        assert r.confidence == 1.0
        assert r.error is None

    def test_failed_result(self):
        r = AgentResult(agent_id="test", status="failed", error="boom")
        assert r.status == "failed"
        assert r.error == "boom"


class TestBaseAgent:
    def test_should_run_default(self):
        agent = MockAgent()
        assert agent.should_run({}) is True

    def test_build_prompt(self):
        agent = MockAgent()
        p = agent.build_prompt({"ticker": "ALLY US"})
        assert "ALLY US" in p

    def test_validate_output(self):
        agent = MockAgent()
        result = agent.validate_output("hello")
        assert result == {"parsed": "hello"}

    def test_get_model_extraction_uses_fast(self):
        agent = MockAgent()  # tier = EXTRACTION
        model = agent.get_model()
        assert "haiku" in model.lower()

    def test_get_model_analysis_uses_default(self):
        agent = MockAnalysisAgent()  # tier = ANALYSIS
        model = agent.get_model()
        assert "sonnet" in model.lower()


class TestAgentRegistry:
    def setup_method(self):
        AgentRegistry.clear()

    def test_register_and_get(self):
        AgentRegistry.register(MockAgent)
        assert AgentRegistry.get("test_mock") is MockAgent

    def test_get_missing(self):
        assert AgentRegistry.get("nonexistent") is None

    def test_get_all(self):
        AgentRegistry.register(MockAgent)
        AgentRegistry.register(MockAnalysisAgent)
        assert len(AgentRegistry.get_all()) == 2

    def test_get_by_tier(self):
        AgentRegistry.register(MockAgent)
        AgentRegistry.register(MockAnalysisAgent)
        extraction = AgentRegistry.get_by_tier(AgentTier.EXTRACTION)
        assert len(extraction) == 1
        assert extraction[0] is MockAgent

    def test_execution_order_respects_deps(self):
        AgentRegistry.register(MockAgent)
        AgentRegistry.register(MockAnalysisAgent)
        order = AgentRegistry.get_execution_order()
        ids = [a.agent_id for a in order]
        assert ids.index("test_mock") < ids.index("test_analysis")

    def test_clear(self):
        AgentRegistry.register(MockAgent)
        AgentRegistry.clear()
        assert len(AgentRegistry.get_all()) == 0
