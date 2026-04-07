"""
Source anchoring — verifies extracted values exist in source tables.

After LLM extraction, each extracted numeric value is checked against
the actual tables from the document. Values not found get their
confidence penalised, catching hallucinated numbers.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_numbers_from_tables(tables_data: list) -> set[float]:
    """Extract all numeric values from pdfplumber/HTML table data."""
    numbers = set()
    if not tables_data:
        return numbers
    for table_group in tables_data:
        for table in table_group.get("tables", []):
            for row in table:
                for cell in row:
                    if cell is None:
                        continue
                    text = str(cell).strip()
                    cleaned = re.sub(r'[€$£¥,\s]', '', text)
                    if cleaned.startswith('(') and cleaned.endswith(')'):
                        cleaned = '-' + cleaned[1:-1]
                    try:
                        val = float(cleaned)
                        numbers.add(val)
                        numbers.add(abs(val))
                    except (ValueError, TypeError):
                        pass
    return numbers


def _extract_numbers_from_text(text: str) -> set[float]:
    """Extract all numbers mentioned in document text."""
    numbers = set()
    for match in re.finditer(r'[\($]?[\d,]+\.?\d*[%\)]?', text):
        raw = match.group()
        cleaned = re.sub(r'[€$£¥,%\s()]', '', raw)
        neg = '(' in match.group()
        try:
            val = float(cleaned)
            if neg:
                val = -val
            numbers.add(val)
            numbers.add(abs(val))
        except (ValueError, TypeError):
            pass
    return numbers


def anchor_extractions(
    items: list[dict],
    tables_data: list = None,
    source_text: str = "",
    tolerance: float = 0.01,
) -> list[dict]:
    """
    Verify extracted items against source data.

    For each item with a numeric value:
    - Found in source tables → verified=True, confidence unchanged
    - Found in text only → verified=True, slight penalty
    - Not found anywhere → verified=False, confidence * 0.3
    """
    table_numbers = _extract_numbers_from_tables(tables_data)
    text_numbers = _extract_numbers_from_text(source_text) if source_text else set()

    verified_count = 0
    unverified_count = 0

    for item in items:
        # Calculated values are expected to not match source tables
        if item.get("calculated"):
            item["verified"] = True
            item["source_ref"] = "calculated"
            continue

        value = item.get("metric_value") or item.get("value")
        if value is None or not isinstance(value, (int, float)):
            continue

        val = float(value)
        if val == 0:
            item["verified"] = True
            continue

        found_in_table = any(
            abs(val - n) / max(abs(val), 1) <= tolerance
            for n in table_numbers
        ) if table_numbers else False

        found_in_text = any(
            abs(val - n) / max(abs(val), 1) <= tolerance
            for n in text_numbers
        ) if not found_in_table and text_numbers else False

        if found_in_table:
            item["verified"] = True
            item["source_ref"] = "table"
            verified_count += 1
        elif found_in_text:
            item["verified"] = True
            item["source_ref"] = "text"
            if "confidence" in item and item["confidence"]:
                item["confidence"] = min(item["confidence"], 0.85)
            verified_count += 1
        else:
            item["verified"] = False
            item["source_ref"] = None
            if "confidence" in item and item["confidence"]:
                item["confidence"] *= 0.3
            unverified_count += 1
            logger.warning(
                "[ANCHOR] Value %.2f for '%s' not found in source",
                val, item.get("metric_name", "?")
            )

    logger.info("[ANCHOR] Verified: %d, unverified: %d of %d items",
                verified_count, unverified_count, len(items))
    return items
