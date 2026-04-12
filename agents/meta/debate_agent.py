"""
Debate Agent — adjudicates between Bear and Bull, produces probability-weighted view.

Tier: META (layer 7)
Model: Haiku (orchestration/synthesis — cost-efficient)

Purpose:
  This is the Investment Committee process in agent form.
  It receives both Bear and Bull cases and produces a probability-weighted
  view with explicit adjudication — why is each argument strong or weak,
  and what is the resulting base/bear/bull probability split.

  The prompt should replicate how a good IC chair runs the debate:
  "What's the strongest bear argument? Does the bull have a credible
  counter? Which side has more evidence vs assertion?"

Inputs:
  bear_case           — bear case output (required)
  bull_case           — bull case output (required)
  financial_analyst   — underlying financial assessment
  thesis              — investment thesis
  context_contract    — macro assumptions for consistency check

Output schema:
  debate_summary          — narrative adjudication
  strongest_bear_arg      — the bear argument the bull cannot dismiss
  strongest_bull_arg      — the bull argument the bear cannot dismiss
  base_probability        — % (should sum to 100 with bear/bull)
  bear_probability        — %
  bull_probability        — %
  base_scenario           — implied target and narrative
  verdict                 — buy | hold | watch | avoid
  key_swing_factors       — what would shift the probability split

Prompt file: prompts/agents/debate_agent.txt
  This is the IC process prompt. Write it by answering:
  "How do you chair a good investment debate? What separates strong
  evidence from weak assertion? How do you arrive at a probability
  split without being arbitrarily 33/33/33?"
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class DebateAgent(BaseAgent):

    agent_id   = "debate_agent"
    agent_name = "Debate Agent"
    tier       = AgentTier.META
    # Override to Sonnet — synthesises FA + Bear + Bull into a weighted
    # bull/base/bear probability split. Most analytically demanding
    # step in the pipeline; Haiku struggles with the nuance.
    model_override = "claude-sonnet-4-6"

    depends_on = ["bear_case", "bull_case", "financial_analyst"]
    feeds_into = ["quality_control", "pm_agent"]

    cache_ttl_hours = 24
    tracks_predictions = True
    prediction_horizon_days = 90

    output_schema = {
        "debate_summary":       str,
        "strongest_bear_arg":   str,
        "strongest_bull_arg":   str,
        "base_probability":     float,  # e.g. 55.0
        "bear_probability":     float,  # e.g. 25.0
        "bull_probability":     float,  # e.g. 20.0
        "base_scenario":        dict,
        "verdict":              str,    # buy | hold | watch | avoid
        "key_swing_factors":    list,
        "confidence":           float,
    }

    def should_run(self, inputs: dict) -> bool:
        """Only runs when both Bear and Bull are available."""
        return "bear_case" in inputs and "bull_case" in inputs

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["base_probability", "bear_probability", "bull_probability", "verdict"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Debate Agent output missing required fields: {missing}")

        # Probabilities should sum to ~100
        total = (
            float(data.get("base_probability", 0))
            + float(data.get("bear_probability", 0))
            + float(data.get("bull_probability", 0))
        )
        if abs(total - 100.0) > 5.0:
            logger.warning(
                "Debate Agent probabilities sum to %.1f (expected ~100)", total
            )

        valid_verdicts = {"buy", "hold", "watch", "avoid"}
        if data.get("verdict", "").lower() not in valid_verdicts:
            logger.warning("Unexpected verdict: %s", data.get("verdict"))

        return data

    def extract_predictions(self, output: Any) -> list[dict]:
        predictions = []
        if not isinstance(output, dict):
            return predictions

        verdict = output.get("verdict", "").lower()
        if verdict in ("buy", "avoid"):
            predictions.append({
                "claim":       f"debate_verdict_{verdict}",
                "direction":   "up" if verdict == "buy" else "down",
                "metric":      "price_return",
                "horizon_days": self.prediction_horizon_days,
            })

        return predictions
