"""
Document Triage Agent — classify and prioritise harvester candidates
before they enter the main ingestion flow.

╔══════════════════════════════════════════════════════════════════╗
║  NOT A REGISTERED PIPELINE AGENT.                                  ║
║                                                                    ║
║  Despite the class name and BaseAgent inheritance, this module    ║
║  is NOT decorated with @AgentRegistry.register. It runs at         ║
║  INGESTION TIME from services/harvester/dispatcher.py::_run_triage ║
║  on each candidate, not as part of any pipeline run.               ║
║                                                                    ║
║  If you came here looking for the pipeline agents that produce     ║
║  Bear / Bull / Debate output, see agents/task/ and agents/meta/.   ║
║  If you came here looking for where harvest candidates get         ║
║  classified, you are in the right place — read on.                 ║
╚══════════════════════════════════════════════════════════════════╝

Tier: META (conceptually — runs before the analysis pipeline)
Model: Haiku (short prompt, fast dispatch)

Purpose:
  The harvester finds candidate documents; this agent decides:
  - What type is it, really? (e.g. a PDF titled "Q1-2026 Press Release" that
    is actually a proxy statement amendment)
  - What fiscal period does it cover? (handles Japanese fiscal years,
    amendments, re-issues)
  - Is it thesis-relevant? (priority signal to surface urgent reads)
  - Auto-ingest or flag for analyst review?

Design notes:
  - Stateless; receives candidate + company + thesis, returns classification.
  - Graceful fallback: dispatcher.py catches failures and uses existing
    regex-derived period/type defaults, so a Triage failure never blocks
    the weekly harvest.
  - Every invocation writes an IngestionTriage row for audit + calibration.
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent

logger = logging.getLogger(__name__)


class DocumentTriageAgent(BaseAgent):
    """Ingestion-time classifier. Not a pipeline agent — not auto-registered.
    Invoked directly by services/harvester/dispatcher.py.run_triage()."""

    agent_id   = "document_triage"
    agent_name = "Document Triage"
    tier       = AgentTier.META

    # Haiku is sufficient — this is structured classification, not deep
    # reasoning. Cost matters because Triage runs once per candidate on
    # every harvest run (dozens per week at current scale).
    # No model_override needed — META tier routes to fast model by default.

    depends_on = []
    feeds_into = []

    output_schema = {
        "document_type":    str,    # annual_report | 10-Q | 10-K | transcript | presentation |
                                    # earnings_release | proxy | 8-K | other
        "period_label":     str,    # e.g. "2026_Q1", "2026_FY", or "" if unknown
        "priority":         str,    # immediate | normal | low | skip
        "relevance_score":  int,    # 0-100
        "auto_ingest":      bool,
        "needs_review":     bool,
        "rationale":        str,
        "confidence":       float,
    }

    def should_run(self, inputs: dict) -> bool:
        """Run if a candidate URL is present. Title helps but isn't required —
        EDGAR candidates sometimes arrive with only a URL and filing date."""
        return bool(inputs.get("source_url"))

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["document_type", "priority", "auto_ingest"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Document Triage output missing required fields: {missing}")

        # Normalise — agent may return None for period it genuinely can't determine
        if data.get("period_label") is None:
            data["period_label"] = ""

        # Priority must be one of the allowed values
        allowed_priority = {"immediate", "normal", "low", "skip"}
        if data["priority"] not in allowed_priority:
            logger.warning("Triage returned unexpected priority %r — defaulting to 'normal'", data["priority"])
            data["priority"] = "normal"

        # Types
        data["auto_ingest"] = bool(data.get("auto_ingest", True))
        data["needs_review"] = bool(data.get("needs_review", False))
        if "relevance_score" in data:
            try:
                data["relevance_score"] = int(data["relevance_score"])
            except (TypeError, ValueError):
                data["relevance_score"] = 50

        return data
