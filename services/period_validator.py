"""
Period Validation Agent — post-processor to catch period mislabelling.

The most common extraction error is attributing metrics to the wrong period
when a document shows comparative tables (e.g. Q4 2025 vs Q4 2024 side by side).

This runs after extraction as a lightweight Haiku call to validate period labels.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def detect_reporting_period(text: str, doc_period_label: str = "") -> Optional[str]:
    """
    Detect the primary reporting period from document text using regex heuristics.
    Cheaper than an LLM call — used as a first pass.

    Returns a normalised period label like "Q4 2025" or "FY 2025", or None.
    """
    # Look for common period announcements in the first 3000 chars
    header = text[:3000].lower()

    # Patterns: "for the quarter ended December 31, 2025"
    quarter_match = re.search(
        r'(?:for\s+the\s+)?(?:quarter|three.month|period)\s+ended?\s+'
        r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        r'\s+\d{1,2},?\s+(\d{4})',
        header,
    )
    if quarter_match:
        month = quarter_match.group(1)
        year = quarter_match.group(2)
        q = _month_to_quarter(month)
        if q:
            return f"{q} {year}"

    # Patterns: "Q4 2025 Results" or "Fourth Quarter 2025"
    q_match = re.search(
        r'(?:q([1-4])|(first|second|third|fourth)\s+quarter)\s*[\s,]*(\d{4})',
        header,
    )
    if q_match:
        if q_match.group(1):
            q_num = q_match.group(1)
        else:
            q_map = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
            q_num = q_map.get(q_match.group(2), "")
        year = q_match.group(3)
        if q_num:
            return f"Q{q_num} {year}"

    # Patterns: "Full Year 2025" or "FY 2025" or "Annual Results 2025"
    fy_match = re.search(
        r'(?:full\s*year|fy|annual\s+results?|year\s+ended?)\s*[\s,]*(\d{4})',
        header,
    )
    if fy_match:
        return f"FY {fy_match.group(1)}"

    # Patterns: "H1 2025" or "First Half 2025"
    h_match = re.search(
        r'(?:h([12])|(first|second)\s+half)\s*[\s,]*(\d{4})',
        header,
    )
    if h_match:
        if h_match.group(1):
            h_num = h_match.group(1)
        else:
            h_map = {"first": "1", "second": "2"}
            h_num = h_map.get(h_match.group(2), "")
        year = h_match.group(3)
        if h_num:
            return f"H{h_num} {year}"

    # Fall back to the document's own period label
    if doc_period_label:
        return _normalise_period(doc_period_label)

    return None


def _month_to_quarter(month: str) -> Optional[str]:
    """Map month name to quarter label."""
    month_q = {
        "march": "Q1", "april": "Q1",  # fiscal year ends vary
        "june": "Q2", "july": "Q2",
        "september": "Q3", "october": "Q3",
        "december": "Q4", "january": "Q4",
        "february": "Q4",
    }
    return month_q.get(month.lower())


def _normalise_period(label: str) -> str:
    """Normalise period labels to '{shape} {year}' format covering all
    9 canonical shapes (Q1..Q4, H1, H2, L3Q, FY, LTM)."""
    label = label.strip().upper()

    # Canonical "YYYY_SHAPE" form.
    m = re.match(r'^(\d{4})[_\-]?(Q[1-4]|H[12]|L3Q|FY|LTM)$', label)
    if m:
        return f"{m.group(2)} {m.group(1)}"

    return label


def validate_periods(
    items: list[dict],
    detected_period: str,
    doc_period_label: str = "",
) -> list[dict]:
    """
    Validate and fix period labels on extracted items.

    Rules:
    1. If an item has no period, assign the detected reporting period.
    2. If an item's period doesn't match any plausible period for this document,
       flag it with reduced confidence.
    3. Normalise all period formats to be consistent.

    Returns the items with corrected period labels and adjusted confidence.
    """
    if not detected_period:
        detected_period = _normalise_period(doc_period_label) if doc_period_label else ""

    if not detected_period:
        logger.warning("Period validator: no reporting period detected, skipping")
        return items

    normalised_period = _normalise_period(detected_period)

    # Determine plausible periods for this document
    plausible = _get_plausible_periods(normalised_period)

    fixed_count = 0
    flagged_count = 0

    for item in items:
        period = item.get("period", "")

        if not period:
            # No period → assign the detected one
            item["period"] = normalised_period
            fixed_count += 1
            continue

        normalised = _normalise_period(period)
        item["period"] = normalised

        if normalised not in plausible:
            # Period doesn't look right — reduce confidence
            current_conf = item.get("confidence", 0.8)
            item["confidence"] = min(current_conf, 0.5)
            item["_period_flag"] = f"Unusual period '{normalised}' for a {normalised_period} document"
            flagged_count += 1

    if fixed_count or flagged_count:
        logger.info(
            "Period validator: %d items fixed (no period), %d flagged (unusual period)",
            fixed_count, flagged_count,
        )

    return items


def _get_plausible_periods(primary_period: str) -> set[str]:
    """
    Given a primary reporting period, return all plausible periods
    that might appear in the document (including comparatives).
    """
    plausible = {primary_period}

    m = re.match(r'^Q([1-4])\s+(\d{4})$', primary_period)
    if m:
        q = int(m.group(1))
        year = int(m.group(2))
        # Same quarter prior year
        plausible.add(f"Q{q} {year - 1}")
        # Full year if Q4
        if q == 4:
            plausible.add(f"FY {year}")
            plausible.add(f"FY {year - 1}")
        # YTD / half-year
        if q == 2:
            plausible.add(f"H1 {year}")
            plausible.add(f"H1 {year - 1}")
        # Q3 10-Q typically reports YTD figures alongside the quarter.
        if q == 3:
            plausible.add(f"L3Q {year}")
            plausible.add(f"L3Q {year - 1}")
        # Sequential quarter
        if q > 1:
            plausible.add(f"Q{q-1} {year}")
        else:
            plausible.add(f"Q4 {year - 1}")
        return plausible

    m = re.match(r'^FY\s+(\d{4})$', primary_period)
    if m:
        year = int(m.group(1))
        plausible.add(f"FY {year - 1}")
        for q in range(1, 5):
            plausible.add(f"Q{q} {year}")
            plausible.add(f"Q{q} {year - 1}")
        plausible.add(f"H1 {year}")
        plausible.add(f"H2 {year}")
        plausible.add(f"L3Q {year}")
        plausible.add(f"LTM {year}")
        return plausible

    m = re.match(r'^H([12])\s+(\d{4})$', primary_period)
    if m:
        h = int(m.group(1))
        year = int(m.group(2))
        plausible.add(f"H{h} {year - 1}")
        plausible.add(f"FY {year}")
        plausible.add(f"FY {year - 1}")
        return plausible

    # L3Q (9-month YTD) — Q3 10-Q filings. Plausible comparatives
    # include prior-year L3Q and the constituent quarters.
    m = re.match(r'^L3Q\s+(\d{4})$', primary_period)
    if m:
        year = int(m.group(1))
        plausible.add(f"L3Q {year - 1}")
        for q in (1, 2, 3):
            plausible.add(f"Q{q} {year}")
            plausible.add(f"Q{q} {year - 1}")
        plausible.add(f"H1 {year}")
        return plausible

    # LTM (trailing twelve months) — typically appears in management
    # commentary / tracker dashboards. Compare to prior LTM and FY.
    m = re.match(r'^LTM\s+(\d{4})$', primary_period)
    if m:
        year = int(m.group(1))
        plausible.add(f"LTM {year - 1}")
        plausible.add(f"FY {year}")
        plausible.add(f"FY {year - 1}")
        return plausible

    return plausible
