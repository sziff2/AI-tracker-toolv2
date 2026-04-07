"""
Deterministic extraction evals for AutoResearch.
These evaluate extraction prompt accuracy without LLM judges (zero cost).
"""
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def eval_source_recall(extracted_items: list[dict], source_numbers: set[float], tolerance: float = 0.01) -> float:
    """
    What % of source table values were found in the extraction?
    Score: 0.0-1.0 (higher = better recall)

    Args:
        extracted_items: list of dicts with "metric_value" or "value" key
        source_numbers: set of floats from the source tables
        tolerance: relative tolerance for matching
    """
    if not source_numbers:
        return 1.0  # nothing to find

    extracted_values = set()
    for item in extracted_items:
        val = item.get("metric_value") or item.get("value")
        if val is not None and isinstance(val, (int, float)):
            extracted_values.add(float(val))

    found = 0
    non_zero = [s for s in source_numbers if s != 0]
    for src in non_zero:
        if any(abs(src - ext) / max(abs(src), 1) <= tolerance for ext in extracted_values):
            found += 1

    return found / max(len(non_zero), 1)


def eval_source_precision(extracted_items: list[dict], source_numbers: set[float], tolerance: float = 0.01) -> float:
    """
    What % of extracted values actually exist in the source?
    Score: 0.0-1.0 (higher = fewer hallucinations)
    """
    numeric_items = []
    for item in extracted_items:
        val = item.get("metric_value") or item.get("value")
        if val is not None and isinstance(val, (int, float)):
            numeric_items.append(item)

    if not numeric_items:
        return 1.0

    verified = 0
    for item in numeric_items:
        val = float(item.get("metric_value") or item.get("value"))
        if val == 0:
            verified += 1
            continue
        if any(abs(val - src) / max(abs(val), 1) <= tolerance for src in source_numbers):
            verified += 1

    return verified / len(numeric_items)


def eval_name_accuracy(extracted_items: list[dict], canonical_names: set[str] = None) -> float:
    """
    What % of extracted metric names match the canonical vocabulary?
    Score: 0.0-1.0 (higher = more consistent naming)
    """
    if canonical_names is None:
        # Default canonical names from metric_normaliser
        try:
            from services.metric_normaliser import METRIC_NAME_MAP
            canonical_names = set(METRIC_NAME_MAP.values())
        except ImportError:
            return 1.0

    if not extracted_items:
        return 1.0

    matched = 0
    total = 0
    for item in extracted_items:
        name = (item.get("metric_name") or item.get("line_item") or "").strip()
        if not name:
            continue
        total += 1
        name_lower = name.lower()
        if any(name_lower == cn.lower() for cn in canonical_names):
            matched += 1
        elif any(cn.lower() in name_lower or name_lower in cn.lower() for cn in canonical_names):
            matched += 0.5  # partial match

    return matched / max(total, 1)


def eval_period_accuracy(extracted_items: list[dict], expected_period: str = None) -> float:
    """
    What % of extracted items have the correct period assigned?
    Score: 0.0-1.0
    """
    if not expected_period or not extracted_items:
        return 1.0

    correct = 0
    total = 0
    for item in extracted_items:
        period = item.get("period") or item.get("period_label")
        if period:
            total += 1
            if period == expected_period:
                correct += 1

    return correct / max(total, 1)


def run_extraction_evals(
    extracted_items: list[dict],
    source_numbers: set[float] = None,
    expected_period: str = None,
) -> dict:
    """
    Run all 4 deterministic extraction evals.

    Returns:
        {
            "recall": float,      # 0-1, % of source values found
            "precision": float,   # 0-1, % of extracted values verified
            "name_accuracy": float,  # 0-1, % matching canonical names
            "period_accuracy": float,  # 0-1, % correct period
            "composite": float,   # weighted average
        }
    """
    source_numbers = source_numbers or set()

    recall = eval_source_recall(extracted_items, source_numbers)
    precision = eval_source_precision(extracted_items, source_numbers)
    name_acc = eval_name_accuracy(extracted_items)
    period_acc = eval_period_accuracy(extracted_items, expected_period)

    # Weighted composite: precision matters most (hallucination prevention)
    composite = (recall * 0.25 + precision * 0.35 + name_acc * 0.20 + period_acc * 0.20)

    result = {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "name_accuracy": round(name_acc, 3),
        "period_accuracy": round(period_acc, 3),
        "composite": round(composite, 3),
    }

    logger.info("[EVAL] Extraction evals: recall=%.2f precision=%.2f name=%.2f period=%.2f composite=%.2f",
                recall, precision, name_acc, period_acc, composite)
    return result
