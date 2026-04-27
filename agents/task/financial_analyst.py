"""
Financial Analyst Agent — full-spectrum earnings analysis.

Tier: TASK (layer 0) — runs first, all other agents depend on it.
Model: Sonnet (overrides fast tier default for TASK — quality matters here)

Purpose:
  Full-spectrum quarterly result analysis — revenue quality, margin drivers,
  cash conversion, balance sheet changes, segment performance.
  Produces an overall grade (1–5, calibrated vs expectations) and structured breakdown.

Inputs (from build_agent_context):
  extracted_metrics   — structured KPIs from extraction pipeline
  mda_narrative       — raw MD&A text for qualitative signals
  guidance            — management guidance items
  prior_period        — prior quarter summary for comparison
  thesis              — investment thesis
  confidence_profile  — hedge rate, one-off rate, overall signal quality
  disappearance_flags — metrics that dropped vs prior period
  non_gaap_bridge     — GAAP vs adjusted reconciliation
  segment_data        — revenue/profit decomposition
  context_contract    — shared macro assumptions (DO NOT CONTRADICT)

Output schema:
  revenue_assessment  — growth quality, volume vs price decomposition
  margin_analysis     — direction, drivers, sustainability
  cash_flow_quality   — conversion rate, working capital, FCF
  balance_sheet_flags — leverage changes, liquidity
  segment_commentary  — key movers, mix shifts
  guidance_quality    — credibility, implied assumptions
  management_signals  — tone, confidence, evasion signals
  overall_grade       — calibrated vs expectations:
                        1: ≤-10% (significant miss)
                        2: -10% to -5% (miss)
                        3: +/-5% (in line)
                        4: +5% to +10% (beat)
                        5: ≥+10% (significant beat)
  tracked_kpi_assessment — per-KPI score (1-5) for analyst-selected metrics
  key_surprises       — list of material deviations
  thesis_direction    — strengthened | weakened | unchanged | mixed
  key_assumptions     — list of {assumption, probability, prior, direction,
                        magnitude, evidence, key_watch}
                        Bayesian belief updates: for each thesis pillar,
                        what was the prior probability, what new evidence
                        appeared this quarter, and what is the posterior.
                        The Debate Agent uses these to ground its
                        bull/base/bear probability split.

Prompt file: prompts/agents/financial_analyst.txt
  Write this prompt by answering: "How do you grade a quarterly result
  vs expectations? A 3 means in line (+/-5%), a 5 means a significant
  beat (≥+10%). What expectations benchmark do you use — guidance,
  consensus, or prior period?"

Gold standard test case:
  Test against HEIA Q1 2025 (see analyst-contribution-guide.md §3).
  Expected: B+ overall, margin A-, APAC volume concern in risks.
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class FinancialAnalystAgent(BaseAgent):

    agent_id   = "financial_analyst"
    agent_name = "Financial Analyst"
    tier       = AgentTier.TASK

    # Override to Sonnet — quality matters for this foundational agent
    # even though TASK tier defaults to Haiku. Every other agent reads
    # this output, so sloppy reasoning here cascades.
    model_override = "claude-sonnet-4-6"

    # Override BaseAgent's 4096-token default. The output structure
    # (revenue/margin/cash-flow/segment/management commentary +
    # tracked-KPI scores + key surprises + key assumptions) is verbose
    # for data-rich companies — Sanofi 2026_Q1 (288 metrics, granular
    # product franchises, MD&A narrative, RAG passages) ran out of
    # output tokens at ~3500 and the JSON parse failed mid-string
    # ("Unterminated string ... char 14149") on 2026-04-27. 8192 is
    # well within Sonnet's per-call cap and gives ~2x headroom.
    max_tokens = 8192

    # Orchestration
    depends_on = []  # transcript + presentation analyses are pre-built during ingestion
    feeds_into = ["bear_case", "bull_case", "guidance_tracker"]
    # Note: `feeds_into` is informational only — the registry uses
    # `depends_on` for topological sort. Ghost references to unbuilt
    # agents (pm_agent, consensus_comparison, earnings_quality) were
    # removed in Tier 7.6 cleanup to avoid misleading future readers.

    # Cache for 24h — same quarter results don't change
    cache_ttl_hours = 24

    # Predictions tracked for calibration
    tracks_predictions = True
    prediction_horizon_days = 90

    output_schema = {
        "revenue_assessment":  str,
        "margin_analysis":     str,
        "cash_flow_quality":   str,
        "balance_sheet_flags": str,
        "segment_commentary":  str,
        "guidance_quality":    str,
        "management_signals":  str,
        "overall_grade":       int,   # 1: ≤-10% vs expectations | 2: -10% to -5% | 3: +/-5% | 4: +5% to +10% | 5: ≥+10%
        "key_surprises":       list,
        "thesis_direction":    str,   # strengthened | weakened | unchanged | mixed
        "tracked_kpi_assessment": list,  # [{kpi_name, value, prior_value, score, comment}]
        "confidence":          float,
        # Bayesian belief updates + key assumption probabilities
        # Each assumption is a thesis pillar tracked across periods.
        # The Debate Agent consumes these when setting bull/base/bear probabilities.
        "key_assumptions":     list,  # [{assumption, probability, prior, direction,
                                      #   magnitude, evidence, key_watch}]
    }

    def should_run(self, inputs: dict) -> bool:
        """Run if we have any useful data — metrics, transcript, presentation,
        or annual-report narrative analysis. The annual-report branch was
        added after an ARW US 2025_Q4 run skipped every agent because the
        10-K produced annual_report_analysis but zero metric rows (SEC
        inline-XBRL HTML extraction limitation). A valid annual-report
        narrative alone is enough fuel for a financial take."""
        metrics = inputs.get("extracted_metrics", "")
        has_metrics = metrics and metrics != "No metrics extracted for this period."
        has_transcript = bool(inputs.get("transcript_deep_dive") or inputs.get("transcript_text"))
        has_presentation = bool(inputs.get("presentation_analysis") or inputs.get("presentation_text"))
        has_annual_report = bool(inputs.get("annual_report_analysis") or inputs.get("annual_report_text"))
        if not (has_metrics or has_transcript or has_presentation or has_annual_report):
            logger.warning(
                "Financial Analyst skipping — no metrics, transcript, presentation, or annual-report data"
            )
            return False
        return True

    def validate_output(self, raw: str) -> Any:
        """Parse and validate LLM response. Raises on invalid JSON."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        # Validate required fields
        required = ["overall_grade", "thesis_direction", "revenue_assessment"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Financial Analyst output missing required fields: {missing}")

        # Validate grade is 1-5
        grade = data.get("overall_grade")
        try:
            grade = int(grade)
            data["overall_grade"] = grade
        except (TypeError, ValueError):
            logger.warning("Unexpected grade value: %s — accepting anyway", grade)
        if isinstance(grade, int) and grade not in {1, 2, 3, 4, 5}:
            logger.warning("Grade %d outside 1-5 range — accepting anyway", grade)

        return data

    def extract_predictions(self, output: Any) -> list[dict]:
        """
        Extract verifiable predictions for the calibration system.
        Called only when tracks_predictions=True.
        """
        predictions = []
        if not isinstance(output, dict):
            return predictions

        # Thesis direction is a verifiable prediction
        thesis_direction = output.get("thesis_direction")
        if thesis_direction in ("strengthened", "weakened"):
            predictions.append({
                "claim":       f"thesis_{thesis_direction}",
                "direction":   thesis_direction,
                "metric":      "thesis_direction",
                "horizon_days": self.prediction_horizon_days,
            })

        # Key assumption shifts are trackable predictions
        for assumption in output.get("key_assumptions", [])[:5]:
            if isinstance(assumption, dict) and assumption.get("direction") in ("weakened", "strengthened"):
                predictions.append({
                    "claim":       f"assumption_{assumption.get('assumption', 'unknown')[:40]}",
                    "direction":   "up" if assumption["direction"] == "strengthened" else "down",
                    "metric":      assumption.get("assumption", "unknown"),
                    "horizon_days": self.prediction_horizon_days,
                })

        return predictions
