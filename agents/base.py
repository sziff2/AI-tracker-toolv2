"""
Base agent class for the investment research agent architecture.

Agents are stateless — they receive inputs, call LLMs, return structured output.
The orchestrator handles DB access, agent ordering, and result persistence.

Tier hierarchy (execution order):
  TASK       (layer 0) — lightweight pre-processing, triage, chunking
  DOCUMENT   (layer 1) — per-document extraction (P&L, BS, CF, segments)
  SPECIALIST (layer 2) — company-level analysis (financial analyst, thesis comparison)
  INDUSTRY   (layer 3) — cross-company within an industry
  SECTOR     (layer 4) — sector-level views
  MACRO      (layer 5) — macro regime, rates, credit cycle
  PORTFOLIO  (layer 6) — portfolio-level risk, allocation
  META       (layer 7) — QC, calibration, orchestration

Model routing by tier:
  TASK, DOCUMENT, META → Haiku (fast, cheap)
  SPECIALIST, INDUSTRY → Sonnet (quality)
  SECTOR, MACRO, PORTFOLIO → Sonnet (quality)
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentTier(str, Enum):
    """Agent tiers — determines execution layer, model routing, and QC requirements."""
    TASK = "task"              # Layer 0: pre-processing, triage, chunking
    DOCUMENT = "document"      # Layer 1: per-document extraction
    SPECIALIST = "specialist"  # Layer 2: company-level analysis
    INDUSTRY = "industry"      # Layer 3: cross-company within industry
    SECTOR = "sector"          # Layer 4: sector-level views
    MACRO = "macro"            # Layer 5: macro regime, rates, credit
    PORTFOLIO = "portfolio"    # Layer 6: portfolio-level risk, allocation
    META = "meta"              # Layer 7: QC, calibration, orchestration


# Default execution layer for each tier
_TIER_LAYERS = {
    AgentTier.TASK: 0,
    AgentTier.DOCUMENT: 1,
    AgentTier.SPECIALIST: 2,
    AgentTier.INDUSTRY: 3,
    AgentTier.SECTOR: 4,
    AgentTier.MACRO: 5,
    AgentTier.PORTFOLIO: 6,
    AgentTier.META: 7,
}

# Tiers that use the fast (cheap) model
_FAST_TIERS = {AgentTier.TASK, AgentTier.DOCUMENT, AgentTier.META}


@dataclass
class AgentResult:
    """Standardised result from any agent run."""
    agent_id: str
    status: str = "completed"       # completed | failed | skipped | degraded
    output: Any = None              # The agent's structured output
    confidence: float = 1.0         # 0-1 self-assessed confidence
    qc_score: float | None = None   # Set by QC agent post-hoc
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)     # e.g. "upstream agent X unavailable"
    predictions: list[dict] = field(default_factory=list)  # Trackable predictions for calibration
    prompt_variant_id: str | None = None


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Subclasses must implement:
      - agent_id: unique identifier (e.g. "financial_analyst")
      - agent_name: display name (e.g. "Financial Analyst")
      - tier: AgentTier
      - build_prompt(inputs): construct the LLM prompt from inputs dict
      - validate_output(raw): validate and structure the LLM response

    Optional overrides:
      - should_run(inputs): return False to skip this agent
      - extract_predictions(output): extract trackable predictions
      - depends_on: list of agent_ids that must run first
      - feeds_into: list of agent_ids that consume this agent's output
      - layer: override default execution layer for this tier
    """

    agent_id: str = "base"
    agent_name: str = "Base Agent"
    tier: AgentTier = AgentTier.TASK
    layer: int | None = None           # Override default layer from tier

    # Orchestration
    depends_on: list[str] = []
    feeds_into: list[str] = []

    # Triggers
    trigger: str = "manual"             # "auto" | "manual" | "scheduled"
    trigger_conditions: list[str] = []  # e.g. ["document_type:transcript"]

    # Caching
    cache_ttl_hours: int = 24

    # Calibration
    tracks_predictions: bool = False
    prediction_horizon_days: int = 90

    # Model routing — None means use default for the tier
    model_override: str | None = None

    # Prompt
    max_tokens: int = 4096

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

    def get_layer(self) -> int:
        """Execution layer — determines ordering within a pipeline run."""
        if self.layer is not None:
            return self.layer
        return _TIER_LAYERS.get(self.tier, 0)

    def should_run(self, inputs: dict) -> bool:
        """Return False to skip this agent for the given inputs."""
        return True

    def build_prompt(self, inputs: dict) -> str:
        """Build the LLM prompt from the inputs dict.

        Default: loads from prompts/agents/{agent_id}.txt via the prompt loader.
        Override in subclass for custom prompt assembly.
        """
        from prompts.loader import load_prompt
        return load_prompt(self.agent_id, inputs=inputs)

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
        if self.tier in _FAST_TIERS:
            return settings.agent_fast_model   # Haiku for speed/cost
        return settings.agent_default_model    # Sonnet for quality

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
                max_tokens=self.max_tokens,
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
