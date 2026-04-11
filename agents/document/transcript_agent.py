"""
Transcript Agent — deep analysis of earnings call transcripts.

Tier: DOCUMENT (layer 1)
Model: Sonnet (narrative analysis requires quality)

Purpose:
  Extract the qualitative signals that structured metric extraction misses:
  management tone, guidance statements, analyst concerns, evasion signals,
  and key direct quotes. This is NOT metric extraction — it's narrative
  intelligence.

  An earnings transcript tells you things the 10-Q never will:
  - What management chose to emphasise (vs what they buried)
  - How they responded under pressure in Q&A
  - Whether guidance language tightened or loosened
  - What analysts are worried about (revealed by their questions)
  - Direct quotes that can be used as thesis evidence

Inputs (from build_agent_context):
  transcript_text     — raw transcript text from document_sections
  thesis              — investment thesis (to assess relevance)
  tracked_kpis        — analyst KPIs (flag when management discusses these)
  prior_period        — prior quarter for comparison
  context_contract    — macro assumptions

Output feeds into:
  financial_analyst   — management_signals, guidance statements
  bear_case           — evasion signals, hedging language
  bull_case           — confidence signals, positive catalysts mentioned
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


# NOT registered as a pipeline agent — runs during ingestion instead.
# Prompt file (prompts/agents/transcript_deep_dive.txt) is used by
# background_processor._analyse_document_with_llm() during document processing.
class TranscriptAgent(BaseAgent):

    agent_id   = "transcript_deep_dive"
    agent_name = "Transcript Deep Dive"
    tier       = AgentTier.DOCUMENT

    depends_on = []
    feeds_into = ["financial_analyst", "bear_case", "bull_case"]

    cache_ttl_hours = 24

    output_schema = {
        "management_tone":     dict,   # {overall, confidence_level, vs_prior_quarter}
        "prepared_remarks":    dict,   # {key_themes, emphasis_order, notable_omissions}
        "guidance_statements": list,   # [{metric, statement, direction, specificity}]
        "analyst_concerns":    list,   # [{topic, question_summary, mgmt_response_quality}]
        "evasion_signals":     list,   # [{topic, what_was_asked, how_they_deflected}]
        "key_quotes":          list,   # [{quote, speaker, context, thesis_relevance}]
        "kpi_mentions":        list,   # [{kpi_name, what_was_said, direction}]
        "sentiment_shift":     dict,   # {vs_prior_quarter, driver}
        "confidence":          float,
    }

    def should_run(self, inputs: dict) -> bool:
        """Only run if transcript text is available."""
        return bool(inputs.get("transcript_text"))

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["management_tone", "guidance_statements", "analyst_concerns"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Transcript Agent output missing: {missing}")

        return data
