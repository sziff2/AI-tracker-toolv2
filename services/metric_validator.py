"""
Metric Validation Service — quality controls for extracted data.

Three layers of validation:
  1. Plausibility checks — flag values that are implausible for the company/metric type
  2. Cross-verification — second LLM pass to verify key numbers against source text
  3. Confidence filtering — minimum threshold for metrics used in outputs

Applied after extraction, before metrics are used in briefings/comparisons.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

MIN_CONFIDENCE_FOR_OUTPUT = 0.6     # Metrics below this are excluded from briefings
MIN_CONFIDENCE_FOR_DISPLAY = 0.4    # Below this, metrics are flagged as unreliable
CROSS_CHECK_TOP_N = 15              # Number of key metrics to cross-verify

# Revenue plausibility ranges by unit (order of magnitude checks)
# If a "Revenue" metric is in USD_M and the value is < 1, something is wrong
PLAUSIBILITY_RULES = {
    "revenue": {"min": 10, "max": 500000, "units": ["USD_M", "EUR_M", "GBP_M"]},
    "operating profit": {"min": 1, "max": 100000, "units": ["USD_M", "EUR_M", "GBP_M"]},
    "net profit": {"min": -50000, "max": 100000, "units": ["USD_M", "EUR_M", "GBP_M"]},
    "eps": {"min": -50, "max": 500, "units": ["USD", "EUR", "GBP", None]},
    "margin": {"min": -100, "max": 100, "units": ["%", "bps"]},
    "growth": {"min": -100, "max": 1000, "units": ["%"]},
    "employees": {"min": 10, "max": 5000000, "units": ["count", None]},
    "dividend": {"min": 0, "max": 500, "units": ["USD", "EUR", "GBP", None]},
    "net debt": {"min": -100000, "max": 500000, "units": ["USD_M", "EUR_M", "GBP_M"]},
}


# ─────────────────────────────────────────────────────────────────
# Layer 1: Plausibility Checks
# ─────────────────────────────────────────────────────────────────

def check_plausibility(metric_name: str, metric_value, unit: str) -> dict:
    """
    Check if a metric value is plausible given its name and unit.
    Returns: {"plausible": bool, "reason": str|None, "adjusted_confidence": float}
    """
    if metric_value is None:
        return {"plausible": True, "reason": None, "confidence_penalty": 0}

    try:
        val = float(metric_value)
    except (ValueError, TypeError):
        return {"plausible": True, "reason": None, "confidence_penalty": 0}

    name_lower = metric_name.lower().strip()

    for keyword, rules in PLAUSIBILITY_RULES.items():
        if keyword in name_lower:
            min_val = rules["min"]
            max_val = rules["max"]

            if val < min_val or val > max_val:
                return {
                    "plausible": False,
                    "reason": f"Value {val} outside plausible range [{min_val}, {max_val}] for '{keyword}'",
                    "confidence_penalty": 0.4,
                }

            # Check unit consistency
            if unit and rules["units"] and unit not in rules["units"]:
                # Unit mismatch might mean wrong scale (e.g. millions vs thousands)
                return {
                    "plausible": True,
                    "reason": f"Unit '{unit}' unusual for '{keyword}' (expected {rules['units']})",
                    "confidence_penalty": 0.15,
                }

            break

    # Check for suspiciously round numbers that might be fabricated
    if val != 0 and val == round(val) and abs(val) > 1000:
        # Large round numbers are less likely to be exact reported figures
        return {
            "plausible": True,
            "reason": "Suspiciously round number — verify against source",
            "confidence_penalty": 0.05,
        }

    return {"plausible": True, "reason": None, "confidence_penalty": 0}


def validate_metrics_batch(items: list[dict]) -> list[dict]:
    """
    Run plausibility checks on a batch of extracted metrics.
    Returns the items with added validation fields.
    """
    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("metric_name", "")
        value = item.get("metric_value")
        unit = item.get("unit")
        confidence = item.get("confidence", 0.8)

        check = check_plausibility(name, value, unit)

        # Apply confidence penalty
        adjusted_confidence = max(0.1, confidence - check["confidence_penalty"])

        item["original_confidence"] = confidence
        item["confidence"] = adjusted_confidence
        item["plausibility_check"] = check["plausible"]
        if check["reason"]:
            item["plausibility_warning"] = check["reason"]
        if not check["plausible"]:
            item["needs_review"] = True
            item["validation_flag"] = f"IMPLAUSIBLE: {check['reason']}"

        validated.append(item)

    return validated


# ─────────────────────────────────────────────────────────────────
# Layer 2: Cross-Verification (second LLM pass)
# ─────────────────────────────────────────────────────────────────

CROSS_CHECK_PROMPT = """\
You are a data verification agent. Your ONLY job is to check whether extracted
numbers match the source document text.

EXTRACTED METRICS TO VERIFY:
{metrics_to_check}

SOURCE DOCUMENT TEXT:
{source_text}

For EACH metric, check:
1. Does this exact number appear in the source text?
2. Is the unit correct (millions, %, bps)?
3. Is the metric name accurately describing what the number represents?
4. Could this number have been confused with a different metric?

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<the metric being checked>",
  "extracted_value": "<the value that was extracted>",
  "verified": true | false,
  "correct_value": <the actual value from the source, or null if extraction is correct>,
  "correct_unit": "<correct unit if different, or null>",
  "issue": "<description of the error, or null if correct>",
  "confidence": <0.0-1.0 confidence in the verification>
}}
"""


async def cross_verify_metrics(items: list[dict], source_text: str, max_items: int = CROSS_CHECK_TOP_N) -> list[dict]:
    """
    Run a second LLM pass to verify the most important extracted metrics
    against the source document text.

    Only checks the top N metrics by importance (revenue, profit, margins first).
    Returns the items with updated confidence and verification flags.
    """
    from services.llm_client import call_llm_json_async

    if not items or not source_text:
        return items

    # Prioritise which metrics to verify — focus on the big numbers
    priority_keywords = ["revenue", "sales", "profit", "ebitda", "ebit", "eps",
                         "margin", "cash flow", "fcf", "dividend", "net debt",
                         "growth", "guidance", "target", "outlook"]

    def _priority(item):
        name = item.get("metric_name", "").lower()
        for i, kw in enumerate(priority_keywords):
            if kw in name:
                return i
        return len(priority_keywords)

    # Sort by priority, take top N
    to_check = sorted(items, key=_priority)[:max_items]
    to_check_names = {item.get("metric_name") for item in to_check}

    # Build the verification prompt
    metrics_text = "\n".join(
        f"- {item.get('metric_name')}: {item.get('metric_value')} {item.get('unit', '')} "
        f"(extracted confidence: {item.get('confidence', 'N/A')})"
        for item in to_check
    )

    # Truncate source text to fit in context
    truncated_source = source_text[:12000]

    prompt = CROSS_CHECK_PROMPT.format(
        metrics_to_check=metrics_text,
        source_text=truncated_source,
    )

    try:
        verification_results = await call_llm_json_async(prompt, max_tokens=4096)
        if not isinstance(verification_results, list):
            verification_results = [verification_results] if isinstance(verification_results, dict) else []
    except Exception as e:
        logger.warning("Cross-verification LLM call failed: %s", str(e)[:100])
        return items

    # Build lookup of verification results
    verification_map = {}
    for v in verification_results:
        if isinstance(v, dict):
            verification_map[v.get("metric_name", "")] = v

    # Apply verification results back to items
    for item in items:
        name = item.get("metric_name", "")
        if name in verification_map:
            v = verification_map[name]
            if v.get("verified") is False:
                # The cross-check found an error
                item["cross_check_failed"] = True
                item["cross_check_issue"] = v.get("issue", "Verification failed")
                if v.get("correct_value") is not None:
                    item["suggested_correction"] = v["correct_value"]
                if v.get("correct_unit"):
                    item["suggested_unit"] = v["correct_unit"]
                # Heavily penalise confidence
                item["confidence"] = max(0.1, item.get("confidence", 0.8) * 0.3)
                item["needs_review"] = True
                item["validation_flag"] = f"CROSS-CHECK FAILED: {v.get('issue', '')}"
                logger.warning("Cross-check failed for %s: %s (extracted: %s, correct: %s)",
                              name, v.get("issue"), item.get("metric_value"), v.get("correct_value"))
            elif v.get("verified") is True:
                # Boost confidence for verified metrics
                item["cross_check_verified"] = True
                item["confidence"] = min(1.0, item.get("confidence", 0.8) * 1.15)

    verified_count = sum(1 for v in verification_results if isinstance(v, dict) and v.get("verified"))
    failed_count = sum(1 for v in verification_results if isinstance(v, dict) and v.get("verified") is False)
    logger.info("Cross-verification: %d verified, %d failed out of %d checked",
                verified_count, failed_count, len(verification_results))

    return items


# ─────────────────────────────────────────────────────────────────
# Layer 3: Confidence Filtering
# ─────────────────────────────────────────────────────────────────

def filter_by_confidence(items: list[dict], min_confidence: float = MIN_CONFIDENCE_FOR_OUTPUT) -> list[dict]:
    """
    Filter metrics to only include those above the minimum confidence threshold.
    Used before feeding metrics into briefings, comparisons, and synthesis.
    """
    passed = []
    filtered_out = []
    for item in items:
        conf = item.get("confidence", 0.8)
        if conf >= min_confidence:
            passed.append(item)
        else:
            filtered_out.append(item)

    if filtered_out:
        logger.info("Confidence filter: passed %d, filtered out %d (threshold: %.2f)",
                    len(passed), len(filtered_out), min_confidence)
        for item in filtered_out:
            logger.debug("  Filtered: %s (confidence: %.2f, reason: %s)",
                        item.get("metric_name"), item.get("confidence", 0),
                        item.get("validation_flag", "below threshold"))

    return passed


# ─────────────────────────────────────────────────────────────────
# Combined validation pipeline
# ─────────────────────────────────────────────────────────────────

async def validate_extraction(
    items: list[dict],
    source_text: str = "",
    run_cross_check: bool = True,
    confidence_threshold: float = MIN_CONFIDENCE_FOR_OUTPUT,
) -> dict:
    """
    Full validation pipeline:
      1. Plausibility checks (fast, no LLM)
      2. Cross-verification (LLM call, optional)
      3. Confidence filtering

    Returns:
      {
        "validated": [...],     # items that passed all checks
        "flagged": [...],       # items that need review
        "rejected": [...],      # items below confidence threshold
        "stats": {...}          # validation statistics
      }
    """
    # Step 1: Plausibility
    items = validate_metrics_batch(items)
    implausible_count = sum(1 for i in items if not i.get("plausibility_check", True))

    # Step 2: Cross-verification
    cross_check_failures = 0
    if run_cross_check and source_text:
        items = await cross_verify_metrics(items, source_text)
        cross_check_failures = sum(1 for i in items if i.get("cross_check_failed"))

    # Step 3: Confidence filtering
    passed = filter_by_confidence(items, confidence_threshold)
    rejected = [i for i in items if i.get("confidence", 0.8) < confidence_threshold]
    flagged = [i for i in passed if i.get("needs_review") or i.get("validation_flag")]

    stats = {
        "total_extracted": len(items),
        "passed": len(passed),
        "flagged": len(flagged),
        "rejected": len(rejected),
        "implausible": implausible_count,
        "cross_check_failures": cross_check_failures,
        "avg_confidence": round(sum(i.get("confidence", 0) for i in items) / max(len(items), 1), 2),
    }

    logger.info("Validation complete: %d passed, %d flagged, %d rejected (avg confidence: %.2f)",
                stats["passed"], stats["flagged"], stats["rejected"], stats["avg_confidence"])

    return {
        "validated": passed,
        "flagged": flagged,
        "rejected": rejected,
        "stats": stats,
    }
