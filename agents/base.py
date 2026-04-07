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

Declaring dependencies on a subclass:
    class BearCaseAgent(BaseAgent):
        agent_id = "bear_case"
        depends_on = ["financial_analyst"]   # class-level — read by registry
        feeds_into = ["debate_agent"]

    These are READ by AgentRegistry at class level for topological sort.
    Instance copies are made in __init__ so runtime mutation never bleeds
    across agent classes.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

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
_TIER_LAYERS: dict[AgentTier, int] = {
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

# Anthropic token pricing (USD per 1M tokens).
# Must stay in sync with llm_client._COST_PER_1M.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":            {"input": 15.00, "output": 75.00},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single LLM call."""
    pricing = _MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


@dataclass
class AgentResult:
    """Standardised result from any agent run."""
    agent_id: str
    status: str = "completed"           # completed | failed | skipped | degraded
    output: Any = None                  # The agent's structured output
    confidence: float = 1.0             # 0-1 self-assessed confidence
    qc_score: float | None = None       # Set by QC agent post-hoc
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)      # e.g. "upstream agent X unavailable"
    predictions: list[dict] = field(default_factory=list)  # Trackable predictions for calibration
    prompt_variant_id: str | None = None                   # Which A/B variant was used


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Subclasses must implement:
      - agent_id: unique identifier (e.g. "financial_analyst")
      - agent_name: display name (e.g. "Financial Analyst")
      - tier: AgentTier
      - validate_output(raw): validate and structure the LLM response

    Optional overrides:
      - build_prompt(inputs): construct the LLM prompt
      - should_run(inputs): return False to skip this agent
      - extract_predictions(output): extract trackable predictions
      - prompt_template: inline prompt string (overrides file-based loading)
      - output_schema: dict describing expected output structure (used by QC agent)
      - depends_on: list of agent_ids that must run first (CLASS-level declaration)
      - feeds_into: list of agent_ids that consume this output (CLASS-level declaration)
      - layer: override default execution layer for this tier
    """

    # ------------------------------------------------------------------ #
    #  Identity — subclasses MUST override agent_id, agent_name, tier
    # ------------------------------------------------------------------ #
    agent_id: str = "base"
    agent_name: str = "Base Agent"
    tier: AgentTier = AgentTier.TASK

    # ------------------------------------------------------------------ #
    #  Orchestration — declare as CLASS attributes on subclasses.
    #  Read by AgentRegistry AT CLASS LEVEL for topological sort.
    # ------------------------------------------------------------------ #
    depends_on: ClassVar[list[str]] = []
    feeds_into: ClassVar[list[str]] = []
    trigger_conditions: ClassVar[list[str]] = []

    # ------------------------------------------------------------------ #
    #  Prompts
    # ------------------------------------------------------------------ #
    prompt_template: str | None = None
    output_schema: dict | None = None

    # ------------------------------------------------------------------ #
    #  Execution config
    # ------------------------------------------------------------------ #
    layer: int | None = None
    trigger: str = "manual"
    cache_ttl_hours: int = 24
    max_tokens: int = 4096

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #
    tracks_predictions: bool = False
    prediction_horizon_days: int = 90

    # ------------------------------------------------------------------ #
    #  Model routing
    # ------------------------------------------------------------------ #
    model_override: str | None = None

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        # Instance copies so runtime mutation never bleeds across classes.
        # Registry reads CLASS attributes directly for topological sort.
        self._depends_on: list[str] = list(self.__class__.depends_on)
        self._feeds_into: list[str] = list(self.__class__.feeds_into)
        self._trigger_conditions: list[str] = list(self.__class__.trigger_conditions)

    # ------------------------------------------------------------------ #
    #  Execution helpers
    # ------------------------------------------------------------------ #

    def get_layer(self) -> int:
        if self.layer is not None:
            return self.layer
        return _TIER_LAYERS.get(self.tier, 0)

    def get_model(self) -> str:
        if self.model_override:
            return self.model_override
        from configs.settings import settings
        if self.tier in _FAST_TIERS:
            return settings.agent_fast_model
        return settings.agent_default_model

    # ------------------------------------------------------------------ #
    #  Overrideable hooks
    # ------------------------------------------------------------------ #

    def should_run(self, inputs: dict) -> bool:
        return True

    def build_prompt(self, inputs: dict) -> tuple[str, str | None]:
        """
        Build the LLM prompt from inputs.
        Returns (prompt_text, prompt_variant_id).

        Priority:
          1. DB prompt variant (A/B experiment override)
          2. Subclass inline prompt_template
          3. File: prompts/agents/{agent_id}.txt
        """
        variant_id: str | None = None

        try:
            from services.prompt_registry import get_active_variant
            variant = get_active_variant(self.agent_id)
            if variant:
                return self._render_template(variant.prompt_text, inputs), str(variant.id)
        except Exception as exc:
            self.logger.warning("prompt_registry lookup failed, falling back: %s", exc)

        if self.prompt_template:
            return self._render_template(self.prompt_template, inputs), variant_id

        try:
            from prompts.loader import load_prompt
            return load_prompt(self.agent_id, inputs=inputs), variant_id
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No prompt found for agent '{self.agent_id}'. "
                f"Create prompts/agents/{self.agent_id}.txt or set prompt_template on the class."
            )

    def _render_template(self, template: str, inputs: dict) -> str:
        try:
            return template.format(**inputs)
        except KeyError as exc:
            raise ValueError(
                f"Agent '{self.agent_id}' prompt template missing input key: {exc}"
            ) from exc

    @abstractmethod
    def validate_output(self, raw: str) -> Any:
        """Validate and parse the raw LLM response into structured output."""
        ...

    def extract_predictions(self, output: Any) -> list[dict]:
        return []

    # ------------------------------------------------------------------ #
    #  Main execution
    # ------------------------------------------------------------------ #

    async def run(self, inputs: dict) -> AgentResult:
        """
        Execute the agent. Handles timing, error catching, model routing,
        cost tracking, and prompt variant recording.

        Uses call_llm_native_async (true async, retry on rate limits) and
        receives token counts for accurate per-agent cost tracking.
        """
        if not self.should_run(inputs):
            return AgentResult(agent_id=self.agent_id, status="skipped")

        t0 = time.time()
        try:
            prompt, variant_id = self.build_prompt(inputs)
            model = self.get_model()

            # Use the NATIVE async client — true asyncio, with retry and
            # returns {"text": str, "input_tokens": int, "output_tokens": int}
            from services.llm_client import call_llm_native_async
            llm_result = await call_llm_native_async(
                prompt,
                model=model,
                max_tokens=self.max_tokens,
                feature=f"agent_{self.agent_id}",
                ticker=inputs.get("ticker"),
                period=inputs.get("period_label"),
            )

            raw_text = llm_result["text"]
            input_tokens = llm_result.get("input_tokens", 0)
            output_tokens = llm_result.get("output_tokens", 0)
            cost = _estimate_cost(model, input_tokens, output_tokens)

            output = self.validate_output(raw_text)
            predictions = self.extract_predictions(output) if self.tracks_predictions else []
            duration = int((time.time() - t0) * 1000)

            self.logger.info(
                "Completed in %dms | model=%s | tokens=%d+%d | cost=$%.4f",
                duration, model, input_tokens, output_tokens, cost,
            )

            return AgentResult(
                agent_id=self.agent_id,
                status="completed",
                output=output,
                duration_ms=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                predictions=predictions,
                prompt_variant_id=variant_id,
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
