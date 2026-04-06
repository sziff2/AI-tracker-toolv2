"""
Base agent class for the investment research agent architecture.

Agents are stateless — they receive inputs, call LLMs, return structured output.
The orchestrator handles DB access, agent ordering, and result persistence.
"""
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentTier(str, Enum):
    """Agent complexity tiers — determines model routing and QC requirements."""
    EXTRACTION = "extraction"      # Parse and extract data from documents
    ANALYSIS = "analysis"          # Compare, score, synthesise extracted data
    SYNTHESIS = "synthesis"        # Generate human-readable outputs
    META = "meta"                  # QC, calibration, orchestration


@dataclass
class AgentResult:
    """Standardised result from any agent run."""
    agent_id: str
    status: str = "completed"      # completed | failed | skipped
    output: Any = None             # The agent's structured output
    confidence: float = 1.0        # 0-1 self-assessed confidence
    qc_score: float | None = None  # Set by QC agent post-hoc
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    predictions: list[dict] = field(default_factory=list)  # Trackable predictions


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Subclasses must implement:
      - agent_id: unique identifier (e.g. "financial_extractor")
      - tier: AgentTier
      - build_prompt(inputs): construct the LLM prompt from inputs dict
      - validate_output(raw): validate and structure the LLM response

    Optional overrides:
      - should_run(inputs): return False to skip this agent
      - extract_predictions(output): extract trackable predictions
      - depends_on: list of agent_ids that must run first
      - feeds_into: list of agent_ids that consume this agent's output
    """

    agent_id: str = "base"
    tier: AgentTier = AgentTier.EXTRACTION
    depends_on: list[str] = []
    feeds_into: list[str] = []
    cache_ttl_hours: int = 24
    tracks_predictions: bool = False

    # Model routing — None means use default for the tier
    model_override: str | None = None

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

    def should_run(self, inputs: dict) -> bool:
        """Return False to skip this agent for the given inputs."""
        return True

    @abstractmethod
    def build_prompt(self, inputs: dict) -> str:
        """Build the LLM prompt from the inputs dict."""
        ...

    @abstractmethod
    def validate_output(self, raw: str) -> Any:
        """Validate and parse the raw LLM response into structured output."""
        ...

    def extract_predictions(self, output: Any) -> list[dict]:
        """Extract trackable predictions from the output. Override if tracks_predictions=True."""
        return []

    def get_model(self) -> str:
        """Determine which model to use for this agent."""
        if self.model_override:
            return self.model_override
        from configs.settings import settings
        if self.tier in (AgentTier.EXTRACTION, AgentTier.META):
            return settings.agent_fast_model  # Haiku for speed/cost
        return settings.agent_default_model   # Sonnet for quality

    async def run(self, inputs: dict) -> AgentResult:
        """Execute the agent. Handles timing, error catching, model routing."""
        if not self.should_run(inputs):
            return AgentResult(agent_id=self.agent_id, status="skipped")

        t0 = time.time()
        try:
            prompt = self.build_prompt(inputs)
            model = self.get_model()

            from services.llm_client import call_llm_async
            raw = await call_llm_async(
                prompt,
                model=model,
                feature=f"agent_{self.agent_id}",
                ticker=inputs.get("ticker"),
                period=inputs.get("period_label"),
            )

            output = self.validate_output(raw)
            predictions = self.extract_predictions(output) if self.tracks_predictions else []
            duration = int((time.time() - t0) * 1000)

            self.logger.info("Completed in %dms (model=%s)", duration, model)

            return AgentResult(
                agent_id=self.agent_id,
                status="completed",
                output=output,
                duration_ms=duration,
                predictions=predictions,
            )
        except Exception as exc:
            duration = int((time.time() - t0) * 1000)
            self.logger.error("Failed after %dms: %s", duration, exc)
            return AgentResult(
                agent_id=self.agent_id,
                status="failed",
                error=str(exc),
                duration_ms=duration,
            )
