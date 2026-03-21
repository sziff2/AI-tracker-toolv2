"""
AutoResearch — Two-Pipeline Autonomous Prompt Optimisation
==========================================================

Pipeline A — Extraction
  Evals (no LLM judge for most):
    snippet_recall     (40%) — does the new prompt find the same verbatim source_snippets
                               already stored in ExtractedMetric rows? Pure string match.
    schema_compliance  (30%) — are all required fields present and correctly typed?
                               Pure Python validation, zero LLM cost.
    hallucination_rate (30%) — do extracted numeric values actually appear in the document?
                               LLM check, but grounded against the real source text.

Pipeline B — Output  (synthesis, briefing, ir_questions, surprise, thesis_comparison)
  Evals (LLM judge with Oldfield rubric + hallucination cap):
    specificity         (25%) — precise numbers cited, not vague language
    thesis_linkage      (20%) — company thesis injected from DB, so this is actually testable
    management_scrutiny (20%) — critically assesses vs parrots management
    actionability       (20%) — clear add/hold/trim steer
    hallucination_check (10%) — hard cap at 4.0 if hallucinations found
    conciseness          (5%) — no padding

Both pipelines share a single _state dict with per-pipeline sub-keys.
MIN_IMPROVEMENT = 0.5 on a 0–10 scale for both.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from apps.api.models import (
    ABExperiment, Company, Document, ExtractedMetric,
    PromptVariant, ThesisVersion,
)
from services.llm_client import call_llm, call_llm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AUTORUN/%(pipeline)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autorun")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
MIN_IMPROVEMENT      = 0.5    # score delta to trigger promotion
SLEEP_BETWEEN_TYPES  = 20     # seconds between prompt types in a round
SLEEP_BETWEEN_ROUNDS = 90     # seconds between full sweeps
MAX_TOKENS_REFINE    = 8192
MAX_TOKENS_RUN       = 3000
MAX_TOKENS_HALLUC    = 1024
MAX_TOKENS_RUBRIC    = 1500
DOC_CHAR_LIMIT       = 14000

EXTRACTION_PROMPT_TYPES = [
    "extraction_combined",
    "extraction_earnings",
    "extraction_transcript",
    "extraction_broker",
    "extraction_presentation",
]

OUTPUT_PROMPT_TYPES = [
    "synthesis",
    "briefing",
    "ir_questions",
    "surprise",
    "thesis_comparison",
]

DOC_TYPE_MAP = {
    "extraction_combined":     ["earnings_release", "transcript", "annual_report"],
    "extraction_earnings":     ["earnings_release"],
    "extraction_transcript":   ["transcript"],
    "extraction_broker":       ["broker_note"],
    "extraction_presentation": ["presentation"],
    "synthesis":               ["earnings_release", "transcript"],
    "briefing":                ["earnings_release", "transcript"],
    "ir_questions":            ["earnings_release", "transcript", "presentation"],
    "surprise":                ["earnings_release", "transcript"],
    "thesis_comparison":       ["earnings_release", "transcript"],
}

# Required fields for extraction schema compliance check
EXTRACTION_REQUIRED_FIELDS = {
    "metric_name":   str,
    "source_snippet": str,
    "confidence":    (int, float),
}
EXTRACTION_VALUE_FIELDS = ["metric_value", "metric_text"]   # at least one must be non-null

# ─────────────────────────────────────────────────────────────────
# Eval prompts
# ─────────────────────────────────────────────────────────────────

EXTRACTION_HALLUC_PROMPT = """\
You are verifying whether extracted financial metrics contain invented numbers.

For each item in EXTRACTED ITEMS, check if its numeric value appears anywhere in the SOURCE.

Rules:
- Only flag items where a specific number is stated but CANNOT be found anywhere in the source.
- Do NOT flag items where metric_value is null (those are text-only extractions).
- Do NOT flag reasonable unit conversions (millions to billions etc.).
- Do NOT flag items that use metric_text instead of metric_value.

SOURCE DOCUMENT (truncated):
---
{source}
---

EXTRACTED ITEMS (JSON):
---
{items}
---

Respond ONLY with a JSON object. No preamble, no markdown fences.
{{
  "total_numeric_items": <integer — items where metric_value is not null>,
  "hallucinated_count": <integer — numeric items whose value cannot be found in source>,
  "hallucinated_examples": ["<metric_name: value that cannot be found>"],
  "verdict": "<one sentence>"
}}"""

OUTPUT_HALLUC_PROMPT = """\
You are a strict fact-checker for investment research briefings.

Identify any claim, number, or assertion in the OUTPUT that is NOT supported by the SOURCE DOCUMENT.

Rules:
- Hallucinated = a specific number or named fact that does not appear in the source.
- Vague language ("margins declined") without a number is NOT a hallucination — just weak.
- Inferences from stated numbers (e.g. calculating a percentage change) are NOT hallucinations.

SOURCE DOCUMENT (truncated):
---
{source}
---

OUTPUT TO CHECK:
---
{output}
---

Respond ONLY with a JSON object. No preamble, no markdown fences.
{{
  "hallucinations_found": true | false,
  "count": <integer>,
  "examples": ["<specific hallucinated claim>"],
  "verdict": "<one sentence>"
}}"""

OUTPUT_RUBRIC_PROMPT = """\
You are evaluating an investment research output against the Oldfield Partners house style.

Oldfield Partners is a concentrated, long-term value fund. Their standard:
facts before narrative, scepticism about management, explicit thesis linkage,
opinionated bottom line. No waffle, no PR repetition.

INVESTMENT THESIS ON FILE FOR THIS COMPANY:
---
{thesis}
---

Score the OUTPUT on each dimension from 0–10 using these ANCHORED definitions:

1. SPECIFICITY (0–10)
   10 = every factual claim backed by a precise number with units
        EXAMPLE 10: "EBIT margin compressed 180bps to 14.2%; FCF conversion fell to 67% from 84%"
        EXAMPLE 5:  "margins declined and cash generation weakened"
        EXAMPLE 0:  "the company faced headwinds in several areas"
   0  = entirely vague, zero numbers cited

2. THESIS LINKAGE (0–10)
   Uses the INVESTMENT THESIS ON FILE above.
   10 = bottom line explicitly states whether the thesis is intact, strengthened, or weakened,
        with a specific reason tied to the thesis pillars above
        EXAMPLE 10: "The moat thesis is intact — route density gains of 4% confirm the network
                     effects argument, but FCF miss pushes back the valuation case to 2H26"
        EXAMPLE 5:  "Overall a mixed result; thesis broadly on track"
        EXAMPLE 0:  thesis not mentioned at all
   0  = no reference to the investment thesis

3. MANAGEMENT SCRUTINY (0–10)
   10 = distinguishes what management claims from what numbers show; flags evasive language;
        credits confidence where earned
        EXAMPLE 10: "Management guided 'operational improvement' but operating leverage
                     was actually negative on flat volumes — this is inconsistent with
                     the efficiency story they have been selling"
        EXAMPLE 5:  neutral reporting of management commentary without assessment
        EXAMPLE 0:  uncritically repeats management talking points as fact

4. ACTIONABILITY (0–10)
   10 = ends with explicit add/hold/trim AND a specific forward-looking reason
        EXAMPLE 10: "Hold — thesis intact but valuation now pricing in the recovery;
                     next catalyst is Q3 FCF which will confirm or deny the working
                     capital normalisation story"
        EXAMPLE 5:  "we will continue to monitor developments"
        EXAMPLE 0:  no conclusion; output ends with a description of results

5. CONCISENESS (0–10)
   10 = every sentence adds new information; zero repetition; no hedging for its own sake
   5  = some repetition or filler
   0  = padded throughout

SOURCE DOCUMENT (for reference):
---
{source}
---

OUTPUT TO EVALUATE:
---
{output}
---

Respond ONLY with a JSON object. No preamble, no markdown fences.
{{
  "specificity":         {{"score": <0-10>, "reason": "<one sentence citing evidence from output>"}},
  "thesis_linkage":      {{"score": <0-10>, "reason": "<one sentence citing evidence from output>"}},
  "management_scrutiny": {{"score": <0-10>, "reason": "<one sentence citing evidence from output>"}},
  "actionability":       {{"score": <0-10>, "reason": "<one sentence citing evidence from output>"}},
  "conciseness":         {{"score": <0-10>, "reason": "<one sentence citing evidence from output>"}},
  "overall_verdict":     "<two sentences: strongest dimension and biggest gap>"
}}"""

OUTPUT_RUBRIC_WEIGHTS = {
    "specificity":         0.25,
    "thesis_linkage":      0.20,
    "management_scrutiny": 0.20,
    "actionability":       0.20,
    "conciseness":         0.05,
    # hallucination_check is a cap, not a weighted dimension
}

REFINE_PROMPT_EXTRACTION = """\
You are a prompt engineer improving LLM extraction prompts for a value-oriented buy-side fund.

Outputs are scored on:
  snippet_recall    (40%) — does the prompt find the exact verbatim text from the document?
  schema_compliance (30%) — are all required fields (metric_name, source_snippet,
                            confidence, metric_value or metric_text) always populated?
  hallucination_rate(30%) — are extracted numeric values actually present in the document?

Rewrite the prompt to score higher. Rules:
- Keep the EXACT same JSON output schema.
- Add explicit instructions to always copy verbatim text into source_snippet.
- Add instructions to never infer or calculate values — only state what is written.
- Add instructions for every required field to be populated on every item.
- Remove anything that encourages the LLM to summarise rather than quote exactly.
- The prompt MUST contain {{text}} as a placeholder for document input.

{feedback_section}

CURRENT ACTIVE PROMPT ({variant_name}, generation {generation}):
---
{prompt_text}
---

Return ONLY the improved prompt text. No explanation, no markdown, no preamble."""

REFINE_PROMPT_OUTPUT = """\
You are a prompt engineer improving LLM briefing prompts for Oldfield Partners,
a concentrated, long-term value fund.

Outputs are scored on this rubric:
  specificity         (25%) — every claim backed by a precise number with units
  thesis_linkage      (20%) — bottom line explicitly connects to the investment thesis
  management_scrutiny (20%) — distinguishes claims from numbers; assesses credibility
  actionability       (20%) — ends with an explicit add/hold/trim steer
  conciseness          (5%) — no waffle, no padding

Rewrite the prompt to score higher. Rules:
- Keep the EXACT same JSON output schema.
- Instruct the LLM to always state exact figures with units for every claim.
- Instruct the LLM to assess whether the investment thesis is intact, strengthened, or weakened.
- Instruct the LLM to distinguish what management claims from what numbers show.
- Instruct the LLM to end with an explicit add/hold/trim with one specific forward reason.
- Remove any instruction that encourages hedging, padding, or balancing every point.
- The prompt MUST contain {{text}} as a placeholder for document input.

{feedback_section}

CURRENT ACTIVE PROMPT ({variant_name}, generation {generation}):
---
{prompt_text}
---

Return ONLY the improved prompt text. No explanation, no markdown, no preamble."""

# ─────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────
def _make_pipeline_state():
    return {
        "running": False,
        "started_at": None,
        "stop_requested": False,
        "job_id": None,
        "current_type": None,
        "current_step": None,
        "experiments_run": 0,
        "promotions": 0,
        "rejections": 0,
        "skipped": 0,
        "errors": 0,
        "score_deltas": [],
    }

_state = {
    "extraction": _make_pipeline_state(),
    "output":     _make_pipeline_state(),
    "log": [],   # combined log, newest first
}


def _log(pipeline: str, level: str, msg: str, detail: str = ""):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "level": level,
        "msg": msg,
        "detail": detail,
    }
    _state["log"].insert(0, entry)
    if len(_state["log"]) > 500:
        _state["log"] = _state["log"][:500]
    extra = {"pipeline": pipeline.upper()}
    getattr(logger, level.lower(), logger.info)(
        msg + (" — " + detail if detail else ""), extra=extra
    )


def _step(pipeline: str, step: str):
    _state[pipeline]["current_step"] = step


# ─────────────────────────────────────────────────────────────────
# Document fetch
# ─────────────────────────────────────────────────────────────────

async def _get_test_document(
    db: AsyncSession, prompt_type: str
) -> tuple[Document | None, str]:
    doc_types = DOC_TYPE_MAP.get(prompt_type, ["earnings_release"])

    result = await db.execute(
        select(Document)
        .where(
            Document.doc_type.in_(doc_types),
            Document.parsed_text.isnot(None),
            Document.parsed_text != "",
        )
        .order_by(func.random())
        .limit(6)
    )
    docs = result.scalars().all()

    if not docs:
        return None, ""

    # Prefer document with most high-confidence extracted metrics
    best_doc, best_count = docs[0], 0
    for doc in docs:
        q = await db.execute(
            select(func.count(ExtractedMetric.id)).where(
                ExtractedMetric.document_id == doc.id,
                ExtractedMetric.confidence >= 0.65,
            )
        )
        count = q.scalar() or 0
        if count > best_count:
            best_count = count
            best_doc = doc

    return best_doc, best_doc.parsed_text[:DOC_CHAR_LIMIT]


# ─────────────────────────────────────────────────────────────────
# Thesis fetch (for output pipeline)
# ─────────────────────────────────────────────────────────────────

async def _get_thesis(db: AsyncSession, company_id) -> str:
    q = await db.execute(
        select(ThesisVersion)
        .where(ThesisVersion.company_id == company_id, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc())
        .limit(1)
    )
    thesis = q.scalar_one_or_none()
    if not thesis:
        return "No thesis on file for this company."
    parts = [thesis.core_thesis or ""]
    if thesis.key_risks:
        parts.append(f"Key risks: {thesis.key_risks}")
    if thesis.valuation_framework:
        parts.append(f"Valuation: {thesis.valuation_framework}")
    return "\n".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────
# Ground truth fetch (for extraction recall)
# ─────────────────────────────────────────────────────────────────

async def _get_ground_truth_snippets(
    db: AsyncSession, doc: Document
) -> list[str]:
    """Return verbatim source_snippets from existing high-confidence extractions."""
    q = await db.execute(
        select(ExtractedMetric.source_snippet)
        .where(
            ExtractedMetric.document_id == doc.id,
            ExtractedMetric.confidence >= 0.70,
            ExtractedMetric.source_snippet.isnot(None),
            ExtractedMetric.source_snippet != "",
        )
        .limit(40)
    )
    return [row[0].strip() for row in q.all() if row[0] and len(row[0].strip()) > 10]


# ─────────────────────────────────────────────────────────────────
# EXTRACTION EVALS
# ─────────────────────────────────────────────────────────────────

def _eval_schema_compliance(items: list[dict]) -> dict:
    """
    Pure Python — no LLM cost.
    Check every item has required fields and at least one value field.
    """
    if not items:
        return {"score": 0.0, "reason": "No items extracted", "compliant": 0, "total": 0}

    compliant = 0
    issues = []
    for i, item in enumerate(items):
        item_ok = True
        for field, ftype in EXTRACTION_REQUIRED_FIELDS.items():
            val = item.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                issues.append(f"item[{i}] missing {field}")
                item_ok = False
                break
            if not isinstance(val, ftype):
                issues.append(f"item[{i}].{field} wrong type ({type(val).__name__})")
                item_ok = False
                break
        if item_ok:
            # Check at least one value field is populated
            has_value = any(
                item.get(f) is not None and item.get(f) != ""
                for f in EXTRACTION_VALUE_FIELDS
            )
            if not has_value:
                issues.append(f"item[{i}] has no value (metric_value and metric_text both empty)")
                item_ok = False
        if item_ok:
            compliant += 1

    score = round((compliant / len(items)) * 10, 2)
    reason = (
        f"{compliant}/{len(items)} items fully compliant"
        + (f"; e.g. {issues[0]}" if issues else "")
    )
    return {"score": score, "reason": reason, "compliant": compliant, "total": len(items)}


def _eval_snippet_recall(
    items: list[dict], ground_truth_snippets: list[str]
) -> dict:
    """
    Pure Python string matching — no LLM cost.
    For each ground-truth snippet, check if any extracted item's source_snippet
    contains a significant overlap (≥60% of the ground-truth words).
    """
    if not ground_truth_snippets:
        return {"score": 7.0, "reason": "No ground truth snippets available (new document)", "matched": 0, "total": 0}
    if not items:
        return {"score": 0.0, "reason": "No items extracted — zero recall", "matched": 0, "total": len(ground_truth_snippets)}

    extracted_snippets = [
        (item.get("source_snippet") or "").lower().strip()
        for item in items
    ]

    matched = 0
    for gt in ground_truth_snippets:
        gt_words = set(gt.lower().split())
        if len(gt_words) < 3:
            matched += 1  # too short to test meaningfully, assume matched
            continue
        # Check if any extracted snippet contains ≥60% of gt words
        for ext in extracted_snippets:
            ext_words = set(ext.split())
            overlap = len(gt_words & ext_words) / len(gt_words)
            if overlap >= 0.60:
                matched += 1
                break

    score = round((matched / len(ground_truth_snippets)) * 10, 2)
    reason = f"{matched}/{len(ground_truth_snippets)} ground-truth snippets recovered"
    return {"score": score, "reason": reason, "matched": matched, "total": len(ground_truth_snippets)}


async def _eval_extraction_hallucination(
    doc_text: str, items: list[dict]
) -> dict:
    """LLM check — are extracted numeric values present in the source?"""
    numeric_items = [
        {"metric_name": item.get("metric_name"), "metric_value": item.get("metric_value")}
        for item in items
        if item.get("metric_value") is not None
    ]
    if not numeric_items:
        return {"score": 10.0, "reason": "No numeric values to check", "hallucinated": 0, "total": 0}

    # Only check up to 20 items to manage cost
    sample = numeric_items[:20]
    try:
        result = call_llm_json(
            EXTRACTION_HALLUC_PROMPT.format(
                source=doc_text[:8000],
                items=json.dumps(sample, indent=2),
            ),
            max_tokens=MAX_TOKENS_HALLUC,
        )
        total    = int(result.get("total_numeric_items") or len(sample))
        halluc   = int(result.get("hallucinated_count") or 0)
        examples = result.get("hallucinated_examples", [])
        # Hard cap: >20% hallucination rate → score 0
        if total > 0 and (halluc / total) > 0.20:
            score = 0.0
        else:
            score = round(10 - (halluc / max(total, 1)) * 10, 2)
        reason = (
            f"{halluc}/{total} numeric values not found in source"
            + (f"; e.g. {examples[0]}" if examples else "")
        )
        return {"score": score, "reason": reason, "hallucinated": halluc, "total": total}
    except Exception as e:
        return {"score": 7.0, "reason": f"Hallucination check failed: {str(e)[:80]}", "hallucinated": 0, "total": 0}


def _compute_extraction_score(schema: dict, recall: dict, halluc: dict) -> float:
    return round(
        schema["score"] * 0.30
        + recall["score"] * 0.40
        + halluc["score"] * 0.30,
        2
    )


def _extraction_breakdown(schema: dict, recall: dict, halluc: dict) -> str:
    return (
        f"recall={recall['score']:.1f} "
        f"schema={schema['score']:.1f} "
        f"halluc={halluc['score']:.1f}"
    )


# ─────────────────────────────────────────────────────────────────
# OUTPUT EVALS
# ─────────────────────────────────────────────────────────────────

def _compute_output_rubric_score(rubric: dict) -> float:
    total = 0.0
    for dim, weight in OUTPUT_RUBRIC_WEIGHTS.items():
        dim_data = rubric.get(dim, {})
        score = float(dim_data.get("score", 5)) if isinstance(dim_data, dict) else 5.0
        total += score * weight
    return round(total, 2)


def _output_breakdown(rubric: dict, halluc_count: int) -> str:
    abbrev = {
        "specificity": "spec", "thesis_linkage": "thes",
        "management_scrutiny": "mgmt", "actionability": "act", "conciseness": "conc",
    }
    parts = []
    for dim in OUTPUT_RUBRIC_WEIGHTS:
        dim_data = rubric.get(dim, {})
        score = float(dim_data.get("score", 5)) if isinstance(dim_data, dict) else 5.0
        parts.append(f"{abbrev[dim]}={score:.0f}")
    parts.append(f"halluc={halluc_count}")
    return " ".join(parts)


async def _eval_output(
    doc_text: str, output_text: str, thesis: str
) -> dict:
    """Run hallucination check + Oldfield rubric in parallel."""
    source_snippet = doc_text[:8000]
    output_snippet = output_text[:3000]

    loop = asyncio.get_event_loop()

    halluc_result, rubric_result = await asyncio.gather(
        loop.run_in_executor(None, lambda: call_llm_json(
            OUTPUT_HALLUC_PROMPT.format(source=source_snippet, output=output_snippet),
            max_tokens=MAX_TOKENS_HALLUC,
        )),
        loop.run_in_executor(None, lambda: call_llm_json(
            OUTPUT_RUBRIC_PROMPT.format(
                thesis=thesis[:1500],
                source=source_snippet,
                output=output_snippet,
            ),
            max_tokens=MAX_TOKENS_RUBRIC,
        )),
        return_exceptions=True,
    )

    if isinstance(halluc_result, Exception):
        halluc_result = {"hallucinations_found": False, "count": 0, "verdict": "check failed"}
    if isinstance(rubric_result, Exception):
        rubric_result = {d: {"score": 5, "reason": "eval failed"} for d in OUTPUT_RUBRIC_WEIGHTS}

    rubric_score = _compute_output_rubric_score(rubric_result)
    halluc_count = int(halluc_result.get("count", 0))
    hallucinations_found = halluc_result.get("hallucinations_found", False) and halluc_count > 0

    final_score = min(rubric_score, 4.0) if hallucinations_found else rubric_score

    return {
        "score": final_score,
        "rubric_score": rubric_score,
        "halluc_count": halluc_count,
        "hallucinations_found": hallucinations_found,
        "rubric": rubric_result,
        "breakdown": _output_breakdown(rubric_result, halluc_count),
        "overall_verdict": rubric_result.get("overall_verdict", ""),
    }


# ─────────────────────────────────────────────────────────────────
# Refine — generate improved candidate
# ─────────────────────────────────────────────────────────────────

async def _refine_variant(
    db: AsyncSession,
    prompt_type: str,
    pipeline: str,
    dry_run: bool = False,
) -> PromptVariant | None:

    active_q = await db.execute(
        select(PromptVariant).where(
            PromptVariant.prompt_type == prompt_type,
            PromptVariant.is_active == True,
        ).limit(1)
    )
    active = active_q.scalar_one_or_none()
    if not active:
        _log(pipeline, "warning", f"No active variant for {prompt_type}")
        return None

    # Gather past feedback
    exps_q = await db.execute(
        select(ABExperiment).where(
            ABExperiment.prompt_type == prompt_type,
            ABExperiment.status == "completed",
            ABExperiment.winner.isnot(None),
        ).order_by(ABExperiment.created_at.desc()).limit(15)
    )
    experiments = exps_q.scalars().all()

    if experiments:
        lines = [
            f"Winner: {'A(active)' if e.winner=='a' else 'B(candidate)' if e.winner=='b' else 'Tie'} "
            f"| A={e.rating_a or '?'}/10 B={e.rating_b or '?'}/10 "
            f"| {(e.analyst_feedback or '')[:180]}"
            for e in experiments
        ]
        feedback_section = f"EVIDENCE FROM {len(experiments)} PAST EXPERIMENTS:\n" + "\n".join(lines)
    else:
        feedback_section = "No past experiments yet — use the eval criteria as your primary guide."

    template = REFINE_PROMPT_EXTRACTION if pipeline == "extraction" else REFINE_PROMPT_OUTPUT
    refine_prompt = template.format(
        feedback_section=feedback_section,
        variant_name=active.variant_name,
        generation=active.generation,
        prompt_text=active.prompt_text[:4000],
    )

    try:
        new_text = call_llm(refine_prompt, max_tokens=MAX_TOKENS_REFINE)
        if "{text}" not in new_text:
            new_text += "\n\n--- DOCUMENT TEXT ---\n{text}"

        ts = datetime.now(timezone.utc).strftime("%m%d_%H%M")
        new_variant = PromptVariant(
            id=uuid.uuid4(),
            prompt_type=prompt_type,
            variant_name=f"v{active.generation + 1}_ar{pipeline[:3]}_{ts}",
            prompt_text=new_text,
            is_active=False,
            is_candidate=True,
            parent_variant_id=active.id,
            generation=active.generation + 1,
            notes=(
                f"AutoRun/{pipeline} | job {_state[pipeline]['job_id']} | "
                f"evals: {'recall+schema+halluc' if pipeline == 'extraction' else 'oldfield-rubric+halluc'} | "
                f"based on {len(experiments)} past experiments"
            ),
        )
        if not dry_run:
            db.add(new_variant)
            await db.commit()
            await db.refresh(new_variant)
        return new_variant

    except Exception as e:
        _log(pipeline, "error", f"Refine failed for {prompt_type}", str(e)[:200])
        _state[pipeline]["errors"] += 1
        return None


# ─────────────────────────────────────────────────────────────────
# Run a prompt and parse output
# ─────────────────────────────────────────────────────────────────

def _run_prompt_raw(prompt_text: str, doc_text: str) -> str:
    if "{text}" in prompt_text:
        full = prompt_text.replace("{text}", doc_text)
    else:
        full = prompt_text + "\n\n--- DOCUMENT TEXT ---\n" + doc_text
    try:
        return call_llm(full, max_tokens=MAX_TOKENS_RUN)
    except Exception as e:
        return f"ERROR:{str(e)[:200]}"


def _parse_extraction_output(raw: str) -> list[dict]:
    """Try to parse JSON array from extraction output."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────
# Record result and decide
# ─────────────────────────────────────────────────────────────────

async def _record_and_decide(
    db: AsyncSession,
    active: PromptVariant,
    candidate: PromptVariant,
    score_a: float,
    score_b: float,
    breakdown_a: str,
    breakdown_b: str,
    verdict_b: str,
    prompt_type: str,
    pipeline: str,
    dry_run: bool = False,
) -> bool:
    delta = score_b - score_a
    winner = "b" if delta >= MIN_IMPROVEMENT else ("a" if delta <= -MIN_IMPROVEMENT else "tie")

    feedback = (
        f"[AutoRun/{pipeline}] "
        f"A={score_a:.2f} ({breakdown_a}) | "
        f"B={score_b:.2f} ({breakdown_b}) | "
        f"B verdict: {verdict_b[:150]}"
    )

    if not dry_run:
        exp = ABExperiment(
            id=uuid.uuid4(),
            prompt_type=prompt_type,
            variant_a_id=active.id,
            variant_b_id=candidate.id,
            output_a=json.dumps({"score": score_a, "breakdown": breakdown_a}),
            output_b=json.dumps({"score": score_b, "breakdown": breakdown_b}),
            winner=winner,
            rating_a=round(score_a),
            rating_b=round(score_b),
            analyst_feedback=feedback,
            status="completed",
        )
        db.add(exp)
        active.total_runs   = (active.total_runs or 0) + 1
        candidate.total_runs = (candidate.total_runs or 0) + 1

    promoted = False
    if winner == "b":
        _log(pipeline, "info",
             f"✓ PROMOTED {candidate.variant_name} for {prompt_type}",
             f"B={score_b:.2f} vs A={score_a:.2f} (+{delta:.2f}) | {breakdown_b}")
        if not dry_run:
            active.is_active    = False
            active.loss_count   = (active.loss_count or 0) + 1
            candidate.is_active = True
            candidate.is_candidate = False
            candidate.win_count = (candidate.win_count or 0) + 1
        promoted = True
        _state[pipeline]["promotions"] += 1
    else:
        reason = f"delta={delta:+.2f}, need ≥{MIN_IMPROVEMENT}"
        _log(pipeline, "info",
             f"✗ Rejected {candidate.variant_name} for {prompt_type}",
             f"B={score_b:.2f} vs A={score_a:.2f} | {reason}")
        if not dry_run:
            active.win_count    = (active.win_count or 0) + 1
            candidate.loss_count = (candidate.loss_count or 0) + 1
        _state[pipeline]["rejections"] += 1

    if not dry_run:
        await db.commit()

    _state[pipeline]["experiments_run"] += 1
    _state[pipeline]["score_deltas"].append(delta)
    return promoted


# ─────────────────────────────────────────────────────────────────
# Single experiment — EXTRACTION
# ─────────────────────────────────────────────────────────────────

async def _run_extraction_experiment(
    prompt_type: str, dry_run: bool = False
) -> str:
    async with AsyncSessionLocal() as db:
        _step("extraction", f"{prompt_type}: fetching document")
        doc, doc_text = await _get_test_document(db, prompt_type)
        if not doc_text:
            _log("extraction", "warning", f"{prompt_type}: no documents found — skipping")
            _state["extraction"]["skipped"] += 1
            return "skipped"

        ground_truth = await _get_ground_truth_snippets(db, doc)

        _step("extraction", f"{prompt_type}: refining prompt")
        candidate = await _refine_variant(db, prompt_type, "extraction", dry_run)
        if not candidate:
            _state["extraction"]["skipped"] += 1
            return "skipped"

        active_q = await db.execute(
            select(PromptVariant).where(
                PromptVariant.prompt_type == prompt_type,
                PromptVariant.is_active == True,
            ).limit(1)
        )
        active = active_q.scalar_one_or_none()
        if not active:
            _state["extraction"]["skipped"] += 1
            return "skipped"

        _log("extraction", "info",
             f"{prompt_type}: {active.variant_name} vs {candidate.variant_name}",
             f"{len(doc_text):,} chars | {len(ground_truth)} ground-truth snippets")

        # Run both prompts in parallel
        _step("extraction", f"{prompt_type}: generating outputs")
        loop = asyncio.get_event_loop()
        raw_a, raw_b = await asyncio.gather(
            loop.run_in_executor(None, _run_prompt_raw, active.prompt_text, doc_text),
            loop.run_in_executor(None, _run_prompt_raw, candidate.prompt_text, doc_text),
        )

        if raw_a.startswith("ERROR:") or raw_b.startswith("ERROR:"):
            _log("extraction", "error", f"{prompt_type}: output generation failed")
            _state["extraction"]["errors"] += 1
            return "error"

        items_a = _parse_extraction_output(raw_a)
        items_b = _parse_extraction_output(raw_b)

        # Eval both — schema and recall are free; hallucination costs one LLM call per variant
        _step("extraction", f"{prompt_type}: evaluating")

        schema_a = _eval_schema_compliance(items_a)
        schema_b = _eval_schema_compliance(items_b)
        recall_a = _eval_snippet_recall(items_a, ground_truth)
        recall_b = _eval_snippet_recall(items_b, ground_truth)

        halluc_a, halluc_b = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: asyncio.run(_eval_extraction_hallucination(doc_text, items_a))
            ),
            asyncio.get_event_loop().run_in_executor(
                None, lambda: asyncio.run(_eval_extraction_hallucination(doc_text, items_b))
            ),
            return_exceptions=True,
        )
        if isinstance(halluc_a, Exception):
            halluc_a = {"score": 7.0, "reason": "failed", "hallucinated": 0, "total": 0}
        if isinstance(halluc_b, Exception):
            halluc_b = {"score": 7.0, "reason": "failed", "hallucinated": 0, "total": 0}

        score_a = _compute_extraction_score(schema_a, recall_a, halluc_a)
        score_b = _compute_extraction_score(schema_b, recall_b, halluc_b)
        bd_a = _extraction_breakdown(schema_a, recall_a, halluc_a)
        bd_b = _extraction_breakdown(schema_b, recall_b, halluc_b)

        _log("extraction", "info", f"{prompt_type}: eval done",
             f"A={score_a:.2f} [{bd_a}] | B={score_b:.2f} [{bd_b}]")

        _step("extraction", f"{prompt_type}: recording")
        promoted = await _record_and_decide(
            db, active, candidate,
            score_a, score_b, bd_a, bd_b,
            f"items={len(items_b)} recall={recall_b['reason']}",
            prompt_type, "extraction", dry_run,
        )
        return "promoted" if promoted else "rejected"


# ─────────────────────────────────────────────────────────────────
# Single experiment — OUTPUT
# ─────────────────────────────────────────────────────────────────

async def _run_output_experiment(
    prompt_type: str, dry_run: bool = False
) -> str:
    async with AsyncSessionLocal() as db:
        _step("output", f"{prompt_type}: fetching document")
        doc, doc_text = await _get_test_document(db, prompt_type)
        if not doc_text:
            _log("output", "warning", f"{prompt_type}: no documents found — skipping")
            _state["output"]["skipped"] += 1
            return "skipped"

        thesis = await _get_thesis(db, doc.company_id)

        _step("output", f"{prompt_type}: refining prompt")
        candidate = await _refine_variant(db, prompt_type, "output", dry_run)
        if not candidate:
            _state["output"]["skipped"] += 1
            return "skipped"

        active_q = await db.execute(
            select(PromptVariant).where(
                PromptVariant.prompt_type == prompt_type,
                PromptVariant.is_active == True,
            ).limit(1)
        )
        active = active_q.scalar_one_or_none()
        if not active:
            _state["output"]["skipped"] += 1
            return "skipped"

        _log("output", "info",
             f"{prompt_type}: {active.variant_name} vs {candidate.variant_name}",
             f"{len(doc_text):,} chars | thesis: {thesis[:60]}…")

        _step("output", f"{prompt_type}: generating outputs")
        loop = asyncio.get_event_loop()
        raw_a, raw_b = await asyncio.gather(
            loop.run_in_executor(None, _run_prompt_raw, active.prompt_text, doc_text),
            loop.run_in_executor(None, _run_prompt_raw, candidate.prompt_text, doc_text),
        )

        if raw_a.startswith("ERROR:") or raw_b.startswith("ERROR:"):
            _log("output", "error", f"{prompt_type}: output generation failed")
            _state["output"]["errors"] += 1
            return "error"

        _step("output", f"{prompt_type}: evaluating (rubric + halluc)")
        eval_a, eval_b = await asyncio.gather(
            _eval_output(doc_text, raw_a, thesis),
            _eval_output(doc_text, raw_b, thesis),
        )

        _log("output", "info", f"{prompt_type}: eval done",
             f"A={eval_a['score']:.2f} [{eval_a['breakdown']}] | "
             f"B={eval_b['score']:.2f} [{eval_b['breakdown']}]")

        _step("output", f"{prompt_type}: recording")
        promoted = await _record_and_decide(
            db, active, candidate,
            eval_a["score"], eval_b["score"],
            eval_a["breakdown"], eval_b["breakdown"],
            eval_b.get("overall_verdict", ""),
            prompt_type, "output", dry_run,
        )
        return "promoted" if promoted else "rejected"


# ─────────────────────────────────────────────────────────────────
# Main loop — one per pipeline, run concurrently
# ─────────────────────────────────────────────────────────────────

async def run_pipeline_loop(
    pipeline: str,
    prompt_types: list[str],
    hours: float,
    job_id: str,
    dry_run: bool = False,
):
    run_fn = (
        _run_extraction_experiment if pipeline == "extraction"
        else _run_output_experiment
    )
    deadline = time.time() + hours * 3600

    _state[pipeline].update({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stop_requested": False,
        "job_id": job_id,
        "current_type": None,
        "current_step": None,
        "experiments_run": 0,
        "promotions": 0,
        "rejections": 0,
        "skipped": 0,
        "errors": 0,
        "score_deltas": [],
    })

    _log(pipeline, "info",
         f"Pipeline started — job {job_id}",
         f"Hours: {hours} | Types: {', '.join(prompt_types)} | dry_run={dry_run}")

    round_num = 0
    try:
        while time.time() < deadline and not _state[pipeline]["stop_requested"]:
            round_num += 1
            remaining_h = (deadline - time.time()) / 3600
            _log(pipeline, "info", f"─── Round {round_num} ───", f"{remaining_h:.1f}h remaining")

            for pt in prompt_types:
                if _state[pipeline]["stop_requested"] or time.time() >= deadline:
                    break
                _state[pipeline]["current_type"] = pt
                result = await run_fn(pt, dry_run=dry_run)
                sym = {"promoted": "→", "rejected": "↔", "skipped": "—", "error": "✗"}.get(result, "?")
                _log(pipeline, "info", f"{sym} {pt}: {result.upper()}")
                await asyncio.sleep(SLEEP_BETWEEN_TYPES)

            _state[pipeline]["current_type"] = None
            _state[pipeline]["current_step"] = None

            remaining = deadline - time.time()
            if remaining > SLEEP_BETWEEN_ROUNDS and not _state[pipeline]["stop_requested"]:
                deltas = _state[pipeline]["score_deltas"]
                avg_d = sum(deltas) / len(deltas) if deltas else 0
                _log(pipeline, "info",
                     f"Round {round_num} done | "
                     f"experiments={_state[pipeline]['experiments_run']} "
                     f"promotions={_state[pipeline]['promotions']} "
                     f"avg_delta={avg_d:+.2f}",
                     f"Sleeping {SLEEP_BETWEEN_ROUNDS}s…")
                await asyncio.sleep(SLEEP_BETWEEN_ROUNDS)

    except Exception as e:
        _log(pipeline, "error", f"Pipeline crashed", str(e))
        _state[pipeline]["errors"] += 1
    finally:
        _state[pipeline]["running"] = False
        _state[pipeline]["current_type"] = None
        _state[pipeline]["current_step"] = None
        deltas = _state[pipeline]["score_deltas"]
        avg_d = sum(deltas) / len(deltas) if deltas else 0
        _state[pipeline]["avg_delta"] = round(avg_d, 2)
        _log(pipeline, "info",
             f"Pipeline complete — job {job_id}",
             f"Rounds: {round_num} | "
             f"Experiments: {_state[pipeline]['experiments_run']} | "
             f"Promotions: {_state[pipeline]['promotions']} | "
             f"Avg delta: {avg_d:+.2f}")


async def run_autorun_loop(
    hours: float = 8.0,
    prompt_types: list[str] | None = None,
    pipeline: str = "extraction",
    job_id: str | None = None,
    dry_run: bool = False,
):
    """Entry point called by both API and CLI."""
    if pipeline == "both":
        ext_types = [t for t in (prompt_types or EXTRACTION_PROMPT_TYPES) if t in EXTRACTION_PROMPT_TYPES]
        out_types  = [t for t in (prompt_types or OUTPUT_PROMPT_TYPES)    if t in OUTPUT_PROMPT_TYPES]
        jid = job_id or str(uuid.uuid4())[:8]
        await asyncio.gather(
            run_pipeline_loop("extraction", ext_types or EXTRACTION_PROMPT_TYPES, hours, jid + "_ext", dry_run),
            run_pipeline_loop("output",     out_types  or OUTPUT_PROMPT_TYPES,    hours, jid + "_out", dry_run),
        )
    else:
        default_types = EXTRACTION_PROMPT_TYPES if pipeline == "extraction" else OUTPUT_PROMPT_TYPES
        types = prompt_types or default_types
        jid = job_id or str(uuid.uuid4())[:8]
        await run_pipeline_loop(pipeline, types, hours, jid, dry_run)


def get_state() -> dict:
    s = {
        "extraction": {**_state["extraction"]},
        "output":     {**_state["output"]},
        "log":        _state["log"][:200],
    }
    # Compute avg_delta live if running
    for p in ("extraction", "output"):
        deltas = s[p].get("score_deltas", [])
        s[p]["avg_delta"] = round(sum(deltas) / len(deltas), 2) if deltas else None
        s[p].pop("score_deltas", None)
    return s


def request_stop(pipeline: str = "both"):
    targets = ["extraction", "output"] if pipeline == "both" else [pipeline]
    for p in targets:
        if p in _state:
            _state[p]["stop_requested"] = True
            _log(p, "info", "Stop requested")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch — two-pipeline prompt optimisation")
    parser.add_argument("--hours",    type=float, default=8.0,         help="Run duration in hours")
    parser.add_argument("--pipeline", type=str,   default="extraction", help="extraction | output | both")
    parser.add_argument("--types",    type=str,   default="",           help="Comma-separated prompt types")
    parser.add_argument("--dry-run",  action="store_true",              help="No DB writes")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()] or None
    asyncio.run(run_autorun_loop(
        hours=args.hours,
        prompt_types=types,
        pipeline=args.pipeline,
        dry_run=args.dry_run,
    ))
