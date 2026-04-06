"""Tests for agent base classes and registry."""
import pytest
from agents.base import BaseAgent, AgentResult, AgentTier, _TIER_LAYERS, _FAST_TIERS
from agents.registry import AgentRegistry


class MockTaskAgent(BaseAgent):
    agent_id = "test_task"
    agent_name = "Test Task"
    tier = AgentTier.TASK

    def build_prompt(self, inputs):
        return f"Test prompt for {inputs.get('ticker', 'unknown')}"

    def validate_output(self, raw):
        return {"parsed": raw}


class MockDocAgent(BaseAgent):
    agent_id = "test_document"
    agent_name = "Test Document"
    tier = AgentTier.DOCUMENT
    depends_on = ["test_task"]

    def build_prompt(self, inputs):
        return "Document extraction prompt"

    def validate_output(self, raw):
        return raw


class MockSpecialistAgent(BaseAgent):
    agent_id = "test_specialist"
    agent_name = "Test Specialist"
    tier = AgentTier.SPECIALIST
    depends_on = ["test_document"]

    def build_prompt(self, inputs):
        return "Specialist analysis prompt"

    def validate_output(self, raw):
        return raw


class TestAgentTier:
    def test_all_tiers_exist(self):
        assert len(AgentTier) == 8
        assert AgentTier.TASK.value == "task"
        assert AgentTier.DOCUMENT.value == "document"
        assert AgentTier.SPECIALIST.value == "specialist"
        assert AgentTier.INDUSTRY.value == "industry"
        assert AgentTier.SECTOR.value == "sector"
        assert AgentTier.MACRO.value == "macro"
        assert AgentTier.PORTFOLIO.value == "portfolio"
        assert AgentTier.META.value == "meta"

    def test_tier_layers_ordered(self):
        layers = [_TIER_LAYERS[t] for t in AgentTier]
        assert layers == sorted(layers)

    def test_fast_tiers(self):
        assert AgentTier.TASK in _FAST_TIERS
        assert AgentTier.DOCUMENT in _FAST_TIERS
        assert AgentTier.META in _FAST_TIERS
        assert AgentTier.SPECIALIST not in _FAST_TIERS
        assert AgentTier.PORTFOLIO not in _FAST_TIERS


class TestAgentResult:
    def test_default_values(self):
        r = AgentResult(agent_id="test")
        assert r.status == "completed"
        assert r.confidence == 1.0
        assert r.error is None
        assert r.warnings == []
        assert r.predictions == []

    def test_failed_result(self):
        r = AgentResult(agent_id="test", status="failed", error="boom")
        assert r.status == "failed"
        assert r.error == "boom"

    def test_degraded_result(self):
        r = AgentResult(agent_id="test", status="degraded", warnings=["upstream X missing"])
        assert r.status == "degraded"
        assert len(r.warnings) == 1


class TestBaseAgent:
    def test_should_run_default(self):
        agent = MockTaskAgent()
        assert agent.should_run({}) is True

    def test_build_prompt(self):
        agent = MockTaskAgent()
        p = agent.build_prompt({"ticker": "ALLY US"})
        assert "ALLY US" in p

    def test_validate_output(self):
        agent = MockTaskAgent()
        result = agent.validate_output("hello")
        assert result == {"parsed": "hello"}

    def test_get_layer_from_tier(self):
        assert MockTaskAgent().get_layer() == 0
        assert MockDocAgent().get_layer() == 1
        assert MockSpecialistAgent().get_layer() == 2

    def test_get_layer_override(self):
        agent = MockTaskAgent()
        agent.layer = 5
        assert agent.get_layer() == 5

    def test_get_model_task_uses_fast(self):
        agent = MockTaskAgent()
        model = agent.get_model()
        assert "haiku" in model.lower()

    def test_get_model_document_uses_fast(self):
        agent = MockDocAgent()
        model = agent.get_model()
        assert "haiku" in model.lower()

    def test_get_model_specialist_uses_sonnet(self):
        agent = MockSpecialistAgent()
        model = agent.get_model()
        assert "sonnet" in model.lower()

    def test_get_model_override(self):
        agent = MockTaskAgent()
        agent.model_override = "claude-opus-4-20250514"
        assert "opus" in agent.get_model().lower()


class TestAgentRegistry:
    def setup_method(self):
        AgentRegistry.clear()

    def test_register_and_get(self):
        AgentRegistry.register(MockTaskAgent)
        assert AgentRegistry.get("test_task") is MockTaskAgent

    def test_get_missing(self):
        assert AgentRegistry.get("nonexistent") is None

    def test_get_all(self):
        AgentRegistry.register(MockTaskAgent)
        AgentRegistry.register(MockDocAgent)
        AgentRegistry.register(MockSpecialistAgent)
        assert len(AgentRegistry.get_all()) == 3

    def test_get_by_tier(self):
        AgentRegistry.register(MockTaskAgent)
        AgentRegistry.register(MockDocAgent)
        AgentRegistry.register(MockSpecialistAgent)
        docs = AgentRegistry.get_by_tier(AgentTier.DOCUMENT)
        assert len(docs) == 1
        assert docs[0] is MockDocAgent

    def test_execution_order_respects_deps(self):
        AgentRegistry.register(MockSpecialistAgent)
        AgentRegistry.register(MockTaskAgent)
        AgentRegistry.register(MockDocAgent)
        order = AgentRegistry.get_execution_order()
        ids = [a.agent_id for a in order]
        assert ids.index("test_task") < ids.index("test_document")
        assert ids.index("test_document") < ids.index("test_specialist")

    def test_clear(self):
        AgentRegistry.register(MockTaskAgent)
        AgentRegistry.clear()
        assert len(AgentRegistry.get_all()) == 0
