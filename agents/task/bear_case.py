"""
Bear Case Agent — steelman the negative case.

Tier: TASK (layer 0)
Model: Sonnet

Purpose:
  Find every reason the stock could underperform — deteriorating fundamentals,
  management credibility gaps, macro headwinds, competitive threats, valuation
  risk. Produces a ranked list of risks with probability × impact scores.

  This is NOT a balanced view. It is deliberately one-sided.
  The Debate Agent adjudicates between Bear and Bull.

Inputs:
  financial_analyst   — upstream agent output (required via depends_on)
  thesis              — investment thesis
  disappearance_flags — metrics/guidance that disappeared (natural bear signals)
  non_gaap_bridge     — GAAP vs adjusted divergence (recurring one-offs = bear)
  confidence_profile  — high hedge rate = management uncertainty = bear signal
  context_contract    — macro assumptions (BEAR CASE MUST USE THESE)
                        e.g. if contract says "higher_for_longer rates",
                        rate-sensitive bear arguments are amplified

Output schema:
  bear_thesis         — narrative steelman of the negative case
  key_risks           — list of {risk, probability, impact, evidence}
  downside_scenario   — implied target price and timeline
  early_warning_signals — what to watch that would confirm the bear case
  macro_headwinds     — risks derived from context_contract assumptions
  confidence          — agent's self-assessed confidence 0-1

Prompt file: prompts/agents/bear_case.txt
  Write this prompt by answering: "How do you steelman a negative case?
  What risks do people most commonly miss? How do you avoid being too
  balanced when the brief is specifically to find reasons to be bearish?"
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class BearCaseAgent(BaseAgent):

    agent_id   = "bear_case"
    agent_name = "Bear Case"
    tier       = AgentTier.TASK
    # Override to Sonnet — adversarial reasoning with probability-weighted
    # risk scoring and citation-backed arguments needs quality model.
    model_override = "claude-sonnet-4-6"

    depends_on = ["financial_analyst"]
    feeds_into = ["debate_agent", "pm_agent"]

    cache_ttl_hours = 24
    tracks_predictions = True
    prediction_horizon_days = 180  # Bear case predictions play out slower

    output_schema = {
        "bear_thesis":           str,
        "key_risks":             list,  # [{risk, probability, impact, evidence}]
        "downside_scenario":     dict,  # {target_price, timeline, implied_return}
        "early_warning_signals": list,
        "macro_headwinds":       list,
        "confidence":            float,
    }

    def should_run(self, inputs: dict) -> bool:
        """
        Runs when financial_analyst output is available.
        Also runs automatically when thesis_comparison returns 'strengthened'
        (contrarian check — if things look great, find the risks).
        """
        return "financial_analyst" in inputs

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["bear_thesis", "key_risks"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Bear Case output missing required fields: {missing}")

        if not isinstance(data.get("key_risks"), list):
            raise ValueError("key_risks must be a list")

        return data

    def extract_predictions(self, output: Any) -> list[dict]:
        """
        Extract verifiable bear case predictions.
        e.g. "cost_of_risk will rise", "margins will compress"
        """
        predictions = []
        if not isinstance(output, dict):
            return predictions

        for risk in output.get("key_risks", [])[:3]:  # top 3 only
            if isinstance(risk, dict) and risk.get("metric"):
                predictions.append({
                    "claim":       f"bear_risk_{risk.get('metric', 'unknown')}",
                    "direction":   risk.get("direction", "down"),
                    "metric":      risk.get("metric"),
                    "horizon_days": self.prediction_horizon_days,
                })

        return predictions
