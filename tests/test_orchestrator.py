"""
Tests for AgentOrchestrator.
Uses mock agents — no real LLM calls.
Run: pytest tests/test_orchestrator.py -v
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Any

from agents.base import BaseAgent, AgentTier, AgentResult
from agents.registry import AgentRegistry
from agents.orchestrator import AgentOrchestrator, CRITICAL_AGENTS


class MockFinancialAnalyst(BaseAgent):
    agent_id = "financial_analyst"
    tier = AgentTier.TASK
    depends_on = []
    feeds_into = ["bear_case", "bull_case"]

    def validate_output(self, raw: str) -> Any:
        return {"revenue_growth": 0.05, "thesis_direction": "strengthened"}

    async def run(self, inputs: dict) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id, status="completed",
            output={"revenue_growth": 0.05, "thesis_direction": "strengthened"},
            confidence=0.85, cost_usd=0.012,
            input_tokens=1200, output_tokens=400,
        )


class MockBearCase(BaseAgent):
    agent_id = "bear_case"
    tier = AgentTier.TASK
    depends_on = ["financial_analyst"]

    def validate_output(self, raw: str) -> Any:
        return {"bear_thesis": "margin risk"}

    async def run(self, inputs: dict) -> AgentResult:
        assert "financial_analyst" in inputs, "Bear case must receive FA output"
        return AgentResult(
            agent_id=self.agent_id, status="completed",
            output={"bear_thesis": "margin risk"}, cost_usd=0.008,
        )


class MockFailingAnalyst(BaseAgent):
    agent_id = "financial_analyst"
    tier = AgentTier.TASK
    depends_on = []

    def validate_output(self, raw: str) -> Any:
        return {}

    async def run(self, inputs: dict) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id, status="failed",
            error="LLM returned invalid JSON",
        )


@pytest.fixture(autouse=True)
def clear_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.mark.asyncio
async def test_phase_a_incomplete_blocks_run():
    orch = AgentOrchestrator()
    mock_db = AsyncMock()
    with patch.object(orch, "_is_phase_a_complete", return_value=False):
        result = await orch.run_document_pipeline("company-id", "2025_Q1", db=mock_db)
    assert result.status == "phase_a_incomplete"


@pytest.mark.asyncio
async def test_successful_pipeline():
    AgentRegistry.register(MockFinancialAnalyst)
    AgentRegistry.register(MockBearCase)
    orch = AgentOrchestrator()
    orch.cache_enabled = False
    mock_db = AsyncMock()
    with patch.object(orch, "_is_phase_a_complete", return_value=True), \
         patch("agents.orchestrator.build_agent_context",
               new_callable=AsyncMock,
               return_value={"ticker": "TEST", "period_label": "2025_Q1"}), \
         patch.object(orch, "_create_pipeline_run",
                      new_callable=AsyncMock,
                      return_value=MagicMock(id="test-run-id")), \
         patch.object(orch, "_persist_agent_output", new_callable=AsyncMock), \
         patch.object(orch, "_finalise_pipeline_run", new_callable=AsyncMock):
        result = await orch.run_document_pipeline("company-id", "2025_Q1", db=mock_db)
    assert result.status == "completed"
    assert "financial_analyst" in result.agents_completed
    assert "bear_case" in result.agents_completed


@pytest.mark.asyncio
async def test_critical_agent_failure_aborts():
    AgentRegistry.register(MockFailingAnalyst)
    orch = AgentOrchestrator()
    orch.cache_enabled = False
    mock_db = AsyncMock()
    with patch.object(orch, "_is_phase_a_complete", return_value=True), \
         patch("agents.orchestrator.build_agent_context",
               new_callable=AsyncMock,
               return_value={"ticker": "TEST", "period_label": "2025_Q1"}), \
         patch.object(orch, "_create_pipeline_run",
                      new_callable=AsyncMock,
                      return_value=MagicMock(id="test-run-id")), \
         patch.object(orch, "_persist_agent_output", new_callable=AsyncMock), \
         patch.object(orch, "_finalise_pipeline_run", new_callable=AsyncMock):
        result = await orch.run_document_pipeline("company-id", "2025_Q1", db=mock_db)
    assert result.status == "aborted"
    assert "financial_analyst" in result.agents_failed


@pytest.mark.asyncio
async def test_output_merging():
    AgentRegistry.register(MockFinancialAnalyst)
    AgentRegistry.register(MockBearCase)
    orch = AgentOrchestrator()
    orch.cache_enabled = False
    mock_db = AsyncMock()
    with patch.object(orch, "_is_phase_a_complete", return_value=True), \
         patch("agents.orchestrator.build_agent_context",
               new_callable=AsyncMock,
               return_value={"ticker": "TEST", "period_label": "2025_Q1"}), \
         patch.object(orch, "_create_pipeline_run",
                      new_callable=AsyncMock,
                      return_value=MagicMock(id="test-run-id")), \
         patch.object(orch, "_persist_agent_output", new_callable=AsyncMock), \
         patch.object(orch, "_finalise_pipeline_run", new_callable=AsyncMock):
        result = await orch.run_document_pipeline("company-id", "2025_Q1", db=mock_db)
    assert result.status == "completed"
    assert "financial_analyst" in result.outputs
    assert "bear_case" in result.outputs


@pytest.mark.asyncio
async def test_on_demand_resolves_dependencies():
    """run_agent_on_demand auto-runs financial_analyst before bear_case."""
    AgentRegistry.register(MockFinancialAnalyst)
    AgentRegistry.register(MockBearCase)
    orch = AgentOrchestrator()
    orch.cache_enabled = False
    mock_db = AsyncMock()
    with patch("agents.orchestrator.build_agent_context",
               new_callable=AsyncMock,
               return_value={"ticker": "TEST", "period_label": "2025_Q1"}), \
         patch.object(orch, "_create_pipeline_run",
                      new_callable=AsyncMock,
                      return_value=MagicMock(id="test-run-id")), \
         patch.object(orch, "_persist_agent_output", new_callable=AsyncMock), \
         patch.object(orch, "_finalise_pipeline_run", new_callable=AsyncMock):
        result = await orch.run_agent_on_demand(
            "bear_case", "company-id", "2025_Q1", db=mock_db
        )
    # Both ran — dependency was resolved automatically
    assert "financial_analyst" in result.agents_completed
    assert "bear_case" in result.agents_completed
