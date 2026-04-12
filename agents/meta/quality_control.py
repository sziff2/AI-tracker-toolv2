"""
Quality Control Agent — scores all other agents and checks macro consistency.

Tier: META (layer 7) — always runs last
Model: Haiku

Purpose:
  Two jobs:
  1. Score every other agent's output for quality (specificity, evidence,
     thesis linkage, actionability, absence of hallucination)
  2. Check every output against the Context Contract — flag any agent that
     contradicted the shared macro assumptions

  This is the gate before results are surfaced to the analyst.
  A low QC score should trigger re-run or analyst review flag.

Inputs:
  all_outputs         — dict of all agent outputs from this pipeline run
                        (special input — see orchestrator._execute_pipeline)
  context_contract    — the macro assumptions ALL agents must respect
  thesis              — investment thesis for thesis-linkage checking

Output schema:
  per_agent_scores    — {agent_id: {score: float, issues: list, passed: bool}}
  contract_violations — list of {agent_id, field, contract_says, agent_assumed}
  overall_score       — 0.0 - 1.0 weighted average
  flags               — list of serious issues requiring analyst attention
  recommendation      — accept | review | rerun
  summary             — narrative QC summary for the analyst

Prompt file: prompts/agents/quality_control.txt
  This is the rubric prompt. Feed in your evaluation rubrics from
  analyst-contribution-guide.md §2. For each agent, what makes a 5/5
  output? What are the deal-breakers?

Context Contract checking:
  The QC agent implements the consistency verification described in
  _thesis-architecture.md §1. It checks for:
  - Rate direction contradictions (agent assumes cuts, contract says higher-for-longer)
  - FX contradictions (agent assumes weak dollar, contract says USD strengthening)
  - Credit contradictions (agent assumes easy refinancing, contract says tightening)
"""

import json
import logging
from typing import Any

from agents.base import AgentTier, BaseAgent, AgentResult
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register
class QualityControlAgent(BaseAgent):

    agent_id   = "quality_control"
    agent_name = "Quality Control"
    tier       = AgentTier.META

    # QC must run AFTER all other agents. Depends on debate_agent
    # to ensure it's placed in the final layer.
    depends_on = ["debate_agent"]
    feeds_into = []

    cache_ttl_hours = 0  # Never cache QC — always re-evaluate fresh

    output_schema = {
        "per_agent_scores":   dict,   # {agent_id: {score, issues, passed}}
        "contract_violations": list,  # [{agent_id, field, contract_says, agent_assumed}]
        "overall_score":      float,  # 0.0 - 1.0
        "flags":              list,   # serious issues
        "recommendation":     str,    # accept | review | rerun
        "summary":            str,
    }

    def should_run(self, inputs: dict) -> bool:
        """Run if at least one other agent completed."""
        all_outputs = inputs.get("all_outputs", {})
        return len(all_outputs) > 0

    def build_prompt(self, inputs: dict) -> tuple[str, str | None]:
        """
        Override to inject all_outputs and context_contract directly
        into the prompt rather than relying on simple {key} substitution.
        The parent build_prompt handles file loading and variant lookup.
        """
        # Serialise all_outputs for the prompt
        all_outputs = inputs.get("all_outputs", {})
        inputs["all_outputs_json"] = json.dumps(all_outputs, indent=2, default=str)[:20000]

        # Serialise context contract
        contract = inputs.get("context_contract", {})
        inputs["context_contract_json"] = json.dumps(contract, indent=2, default=str)

        return super().build_prompt(inputs)

    def validate_output(self, raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned[3:]

        data = json.loads(cleaned)

        required = ["overall_score", "recommendation"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"QC output missing required fields: {missing}")

        score = float(data.get("overall_score", 0))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"overall_score must be 0-1, got {score}")

        valid_recs = {"accept", "review", "rerun"}
        if data.get("recommendation", "").lower() not in valid_recs:
            logger.warning("Unexpected QC recommendation: %s", data.get("recommendation"))

        # Log contract violations if any
        violations = data.get("contract_violations", [])
        if violations:
            logger.warning(
                "QC found %d context contract violation(s): %s",
                len(violations),
                [v.get("agent_id") for v in violations],
            )

        return data

    def _check_contract_consistency(
        self, agent_id: str, agent_output: dict, contract: dict
    ) -> list[dict]:
        """
        Check a single agent's output against the context contract.

        This is a rule-based pre-check that runs before the LLM call.
        The LLM prompt also does a deeper semantic check.
        """
        violations = []
        if not contract or not agent_output:
            return violations

        macro = contract.get("macro_assumptions", {})
        output_text = json.dumps(agent_output).lower()

        # Rate direction check
        rate_direction = macro.get("rate_direction", "")
        if rate_direction == "higher_for_longer":
            if any(phrase in output_text for phrase in
                   ["rate cut", "cuts rates", "rate reduction", "pivot to cuts"]):
                violations.append({
                    "agent_id":       agent_id,
                    "field":          "rate_direction",
                    "contract_says":  "higher_for_longer",
                    "agent_assumed":  "rate cuts",
                    "severity":       "high",
                })

        # USD view check
        usd_view = macro.get("usd_view", "")
        if usd_view == "strengthening":
            if any(phrase in output_text for phrase in
                   ["weak dollar", "usd weakness", "dollar headwind", "fx tailwind from usd"]):
                violations.append({
                    "agent_id":       agent_id,
                    "field":          "usd_view",
                    "contract_says":  "strengthening",
                    "agent_assumed":  "USD weakening",
                    "severity":       "medium",
                })

        # Credit check
        credit_conditions = macro.get("credit_conditions", "")
        if credit_conditions == "tightening":
            if any(phrase in output_text for phrase in
                   ["easy refinanc", "cheap credit", "loose credit"]):
                violations.append({
                    "agent_id":       agent_id,
                    "field":          "credit_conditions",
                    "contract_says":  "tightening",
                    "agent_assumed":  "easy credit",
                    "severity":       "medium",
                })

        return violations
