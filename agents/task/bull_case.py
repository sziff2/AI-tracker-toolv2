"""
Bull Case Agent — steelman the positive case.

Tier: TASK (layer 0)
Model: Sonnet

Purpose:
  Mirror of Bear Case. Find underappreciated positives, optionality,
  and catalysts. Deliberately one-sided upside view.

Inputs:
  financial_analyst   — upstream agent output (required)
  thesis              — investment thesis
  segment_data        — revenue/profit tree (find underappreciated segments)
  guidance            — management guidance (find upside from beat potential)
  context_contract    — macro assumptions (macro tailwinds for this company)

Output schema:
  bull_thesis          — narrative upside case
  upside_catalysts     — list of {catalyst, timeline, magnitude, probability}
  upside_scenario      — implied target and timeline
  what_would_make_you_wrong — honest bear flags embedded in the bull case
  macro_tailwinds      — macro assumptions that help this company
  confidence           — 0-1

Prompt file: prompts/agents/bull_case.txt
  Write this prompt by answering: "How do you identify underappreciated
  positives? What does the market commonly miss? How do you avoid just
  restating the thesis — the bull case should have NEW evidence, not
  just the thesis reworded."
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class BullCaseAgent(BaseAgent):

    agent_id   = "bull_case"
    agent_name = "Bull Case"
    tier       = AgentTier.TASK
    # Override to Sonnet — mirror of bear_case, same reasoning demands.
    model_override = "claude-sonnet-4-6"

    # Mirror bear_case: 8192 to avoid mid-string JSON truncation on
    # data-rich companies. See financial_analyst.py for context.
    max_tokens = 8192

    depends_on = ["financial_analyst"]
    feeds_into = ["debate_agent"]   # pm_agent removed — doesn't exist (Tier 7.6)

    cache_ttl_hours = 24
    tracks_predictions = True
    prediction_horizon_days = 180

    output_schema = {
        "bull_thesis":              str,
        "upside_catalysts":         list,  # [{catalyst, timeline, magnitude, probability}]
        "upside_scenario":          dict,  # {target_price, timeline, implied_return}
        "what_would_make_you_wrong": list,
        "macro_tailwinds":          list,
        "confidence":               float,
    }

    def should_run(self, inputs: dict) -> bool:
        return "financial_analyst" in inputs

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["bull_thesis", "upside_catalysts"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Bull Case output missing required fields: {missing}")

        if not isinstance(data.get("upside_catalysts"), list):
            raise ValueError("upside_catalysts must be a list")

        return data

    def extract_predictions(self, output: Any) -> list[dict]:
        predictions = []
        if not isinstance(output, dict):
            return predictions

        for catalyst in output.get("upside_catalysts", [])[:3]:
            if isinstance(catalyst, dict) and catalyst.get("metric"):
                predictions.append({
                    "claim":       f"bull_catalyst_{catalyst.get('metric', 'unknown')}",
                    "direction":   "up",
                    "metric":      catalyst.get("metric"),
                    "horizon_days": self.prediction_horizon_days,
                })

        return predictions
