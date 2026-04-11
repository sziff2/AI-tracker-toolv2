"""
Presentation Agent — extract strategic signals from investor presentations.

Tier: DOCUMENT (layer 1)
Model: Sonnet

Purpose:
  Investor presentations and Capital Markets Day slides reveal management's
  strategic narrative — what they WANT investors to focus on. This is
  different from what the 10-Q reports and what the transcript reveals
  under questioning.

  Key signals:
  - Slide titles reveal narrative priorities (what's on slide 3 vs slide 30)
  - KPIs management chose to highlight (vs what they left out)
  - Peer comparisons they selected (cherry-picked or fair?)
  - Strategic initiatives with timelines and targets
  - Visual emphasis (charts they built vs tables they buried)
  - New metrics introduced (or old ones quietly dropped)

Inputs:
  presentation_text   — raw presentation text from document_sections
  thesis              — investment thesis
  tracked_kpis        — analyst KPIs
  prior_period        — prior quarter for comparison
  context_contract    — macro assumptions

Output feeds into:
  financial_analyst   — strategic KPIs, management priorities
  bull_case           — catalysts and initiatives highlighted
  bear_case           — what was conspicuously absent
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


# NOT registered as a pipeline agent — runs during ingestion instead.
# Prompt file (prompts/agents/presentation_analysis.txt) is used by
# background_processor._analyse_document_with_llm() during document processing.
class PresentationAgent(BaseAgent):

    agent_id   = "presentation_analysis"
    agent_name = "Presentation Analysis"
    tier       = AgentTier.DOCUMENT

    depends_on = []
    feeds_into = ["financial_analyst", "bear_case", "bull_case"]

    cache_ttl_hours = 24

    output_schema = {
        "narrative_priorities": list,   # [{topic, slide_position, emphasis_level}]
        "strategic_kpis":      list,   # [{metric, value, context, new_or_recurring}]
        "initiatives":         list,   # [{initiative, timeline, target, credibility}]
        "peer_comparisons":    list,   # [{metric, company_position, cherry_picked}]
        "notable_omissions":   list,   # [{topic, why_notable, last_mentioned}]
        "new_metrics":         list,   # [{metric, why_introduced, bullish_or_bearish}]
        "dropped_metrics":     list,   # [{metric, last_shown, likely_reason}]
        "management_message":  str,    # one-paragraph synthesis of what mgmt wants you to believe
        "confidence":          float,
    }

    def should_run(self, inputs: dict) -> bool:
        """Only run if presentation text is available."""
        return bool(inputs.get("presentation_text"))

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["narrative_priorities", "management_message"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Presentation Agent output missing: {missing}")

        return data
