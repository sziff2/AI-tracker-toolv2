"""
Competitive Positioning Agent — Tier 5.1.

Tier: INDUSTRY (layer 3)
Model: Sonnet

Purpose:
  The first cross-company agent. Takes the subject company plus an
  analyst-curated peer set and answers three questions:
    1. Where is the subject ahead / behind peers on current-period
       metrics (growth, margin, any sector KPIs)?
    2. Is the gap widening or narrowing vs last period?
    3. Does the peer data strengthen or weaken any pillar of the
       subject's investment thesis?

  We deliberately do NOT compute a composite "quality score". Buy-side
  analysts want to see the comparison; they'll judge it themselves.

Inputs (from orchestrator):
  ticker             — subject company ticker
  company_name       — subject display name
  period_label       — current period (e.g. "2026_Q1")
  subject_metrics    — list of extracted metric rows (current period)
  peer_metrics       — {peer_ticker: [metric rows]} — current period
  prior_subject_metrics — list of extracted metric rows (prior period)
  prior_peer_metrics    — {peer_ticker: [prior metric rows]}
  thesis             — investment thesis text
  context_contract   — macro assumptions
  sector / industry  — subject classification (for prompt framing)

Output feeds into:
  bear_case     — peer outperformance on a thesis pillar = bear signal
  bull_case     — peer underperformance on a thesis pillar = bull signal
  debate_agent  — comparative evidence is direct thesis support/refutation

Skips when:
  - No peers set on the company
  - Zero overlap metrics between subject and any peer
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class CompetitivePositioningAgent(BaseAgent):

    agent_id   = "competitive_positioning"
    agent_name = "Competitive Positioning"
    tier       = AgentTier.INDUSTRY

    model_override = "claude-sonnet-4-6"

    # Runs after financial_analyst so it can reference the company-level
    # analysis when describing the relative position. Feeds bear/bull.
    depends_on = ["financial_analyst"]
    feeds_into = ["bear_case", "bull_case", "debate_agent"]

    cache_ttl_hours = 24
    tracks_predictions = False

    output_schema = {
        "peers_analysed":    list,   # [str] peer tickers actually compared
        "overlap_metrics":   list,   # [str] metric names present for subject AND ≥1 peer
        "leading_metrics":   list,   # [{metric, subject_value, best_peer, best_peer_value, gap_pct, direction}]
        "trailing_metrics":  list,   # same shape — subject worse than peer median
        "trend_vs_prior":    list,   # [{metric, subject_change, peer_change, implication}]
        "thesis_implications": list, # [{pillar, direction: strengthens|weakens|neutral, reasoning}]
        "data_gaps":         list,   # peers or metrics we couldn't compare + reason
        "overall_signal":    str,    # strengthening | stable | weakening | insufficient_data
        "confidence":        float,
        "sources":           list,   # Tier 4.4 — mandatory citation list
    }

    def should_run(self, inputs: dict) -> bool:
        peers = inputs.get("peer_tickers") or []
        peer_metrics = inputs.get("peer_metrics") or {}
        subject_metrics = inputs.get("subject_metrics") or []

        if not peers:
            logger.info("Competitive Positioning skipping — no peers set")
            return False
        if not subject_metrics:
            logger.info("Competitive Positioning skipping — no subject metrics")
            return False
        if not any(peer_metrics.get(p) for p in peers):
            logger.info("Competitive Positioning skipping — no peer metrics available")
            return False
        return True

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["overall_signal", "peers_analysed"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Competitive Positioning output missing required fields: {missing}")

        for list_field in ("peers_analysed", "overlap_metrics", "leading_metrics",
                           "trailing_metrics", "trend_vs_prior", "thesis_implications",
                           "data_gaps"):
            if list_field in data and not isinstance(data[list_field], list):
                raise ValueError(f"{list_field} must be a list")

        # Sources is optional at the model layer — normalise to [] rather
        # than hard-fail, mirroring the bear_case pattern from Tier 4.4.
        srcs = data.get("sources")
        if not isinstance(srcs, list):
            data["sources"] = []

        return data
