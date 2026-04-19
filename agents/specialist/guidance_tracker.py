"""
Guidance Tracker Agent — holds management accountable across periods.

Tier: SPECIALIST (layer 2)
Model: Sonnet

Purpose:
  Management guidance is the most directly falsifiable thing executives say.
  Over time, systematic misses reveal whether a team overpromises, whether
  operating conditions deteriorated faster than they admitted, or whether
  they quietly moved the goalposts.

  This agent scores prior-period guidance against actual results and
  classifies current-period guidance (tightening | loosening | withdrawn |
  new). It surfaces methodology changes and walkbacks that bury bad news
  in footnotes.

  It does NOT reason about whether guidance is right — only about what
  management said, what happened, and how this period's guidance compares
  to last. Bear and Bull agents consume the output.

Inputs (from build_agent_context):
  guidance           — current period guidance items (string-formatted)
  prior_guidance     — prior period guidance items (string-formatted)
  extracted_metrics  — current period actuals (to score prior guidance)
  transcript_deep_dive — ingested transcript analysis, includes
                         guidance_statements with specificity/direction
  presentation_analysis — includes initiatives with timelines/targets
  thesis             — investment thesis (for relevance weighting)
  context_contract   — macro assumptions

Output feeds into:
  bear_case          — missed/walked-back guidance is a bear signal
  bull_case          — tightened guidance with confidence is a bull signal
  debate_agent       — guidance track record shapes conviction weighting
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class GuidanceTrackerAgent(BaseAgent):

    agent_id   = "guidance_tracker"
    agent_name = "Guidance Tracker"
    tier       = AgentTier.SPECIALIST

    # SPECIALIST tier defaults to Sonnet via _TIER_LAYERS routing,
    # but pin explicitly — guidance comparison needs precise language
    # interpretation and Bayesian-style track-record reasoning.
    model_override = "claude-sonnet-4-6"

    # Runs in parallel with financial_analyst (Layer 2). Reads raw
    # guidance rows from context — no upstream agent required.
    depends_on = []
    feeds_into = ["bear_case", "bull_case", "debate_agent"]

    cache_ttl_hours = 24
    tracks_predictions = True
    prediction_horizon_days = 180  # Next 1-2 quarters

    output_schema = {
        "prior_guidance_scorecard": list,   # [{metric, prior_guidance, actual, outcome, gap_pct, notes}]
        "current_guidance":         list,   # [{metric, value_or_range, specificity, direction_vs_prior}]
        "methodology_changes":      list,   # [{change, implication}]
        "notable_walkbacks":        list,   # [{topic, prior_statement, current_statement, why_notable}]
        "new_disclosures":          list,   # metrics / ranges introduced this period
        "withdrawn_guidance":       list,   # metrics / ranges dropped this period
        "track_record_signal":      str,    # strong | mixed | weak | too_early
        "overall_signal":           str,    # tightening | stable | loosening | withdrawn
        "confidence":               float,
    }

    def should_run(self, inputs: dict) -> bool:
        """Run if any guidance data is present for current or prior period.
        Without either, there's nothing to score or classify."""
        has_current = bool(inputs.get("guidance")) and inputs.get("guidance") != "No guidance items found."
        has_prior = bool(inputs.get("prior_guidance")) and inputs.get("prior_guidance") != "No guidance items found."
        has_transcript_guidance = False
        tdd = inputs.get("transcript_deep_dive") or {}
        if isinstance(tdd, dict):
            has_transcript_guidance = bool(tdd.get("guidance_statements"))
        if not (has_current or has_prior or has_transcript_guidance):
            logger.info("Guidance Tracker skipping — no guidance data in context")
            return False
        return True

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["current_guidance", "overall_signal"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Guidance Tracker output missing required fields: {missing}")

        for list_field in ("prior_guidance_scorecard", "current_guidance",
                           "methodology_changes", "notable_walkbacks",
                           "new_disclosures", "withdrawn_guidance"):
            if list_field in data and not isinstance(data[list_field], list):
                raise ValueError(f"{list_field} must be a list")

        return data

    def extract_predictions(self, output: Any) -> list[dict]:
        """Each current_guidance item is a forward-looking claim we can
        score next period. Store them as predictions so the calibration
        worker can compare against actuals later."""
        predictions: list[dict] = []
        if not isinstance(output, dict):
            return predictions

        for g in output.get("current_guidance", [])[:10]:
            if not isinstance(g, dict):
                continue
            metric = g.get("metric")
            if not metric:
                continue
            predictions.append({
                "claim":        f"guidance_{metric}",
                "metric":       metric,
                "direction":    g.get("direction_vs_prior", "unchanged"),
                "value":        g.get("value_or_range"),
                "specificity":  g.get("specificity"),
                "horizon_days": self.prediction_horizon_days,
            })

        return predictions
