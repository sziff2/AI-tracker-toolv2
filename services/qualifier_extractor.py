"""
Qualifier Language Extractor — detects hedging, attribution, and temporal
qualifiers around extracted metrics.

"Revenue grew 5%" is a hard fact.
"Revenue grew approximately 5%" is hedged.
"Revenue grew 5% supported by one-time contract wins" is attributed.
"Revenue grew 5% in the near term" is temporally qualified.

The pattern of qualification across metrics is a management confidence signal.
A management team that suddenly qualifies previously unqualified metrics is
signalling deterioration before the numbers show it.

Runs as a post-processor after main extraction. Enriches each metric item
with qualifier metadata. Also produces a document-level confidence profile.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Qualifier pattern dictionaries
# ═══════════════════════════════════════════════════════════════════

# Hedging — management is uncertain about the number
HEDGE_PATTERNS = [
    (r"\bapproximately\b", "approximately", 0.8),
    (r"\broughly\b", "roughly", 0.7),
    (r"\babout\b", "about", 0.8),
    (r"\baround\b", "around", 0.8),
    (r"\bbroadly\s+in\s+line\b", "broadly in line", 0.7),
    (r"\bsubstantially\b", "substantially", 0.8),
    (r"\bmore\s+or\s+less\b", "more or less", 0.6),
    (r"\best\.?\b", "estimated", 0.7),
    (r"\bestimated\b", "estimated", 0.7),
    (r"\bpreliminary\b", "preliminary", 0.6),
    (r"\bunaudited\b", "unaudited", 0.7),
    (r"\bsubject\s+to\b", "subject to", 0.6),
    (r"\bin\s+the\s+region\s+of\b", "in the region of", 0.7),
    (r"\bup\s+to\b", "up to", 0.7),
    (r"\bat\s+least\b", "at least", 0.8),
    (r"\bslightly\b", "slightly", 0.8),
    (r"\bmodestly\b", "modestly", 0.8),
    (r"\bmarginal(?:ly)?\b", "marginally", 0.8),
    (r"\bnot\s+insignificant\b", "not insignificant", 0.6),
]

# One-off / attribution — management is explaining away the number
ATTRIBUTION_PATTERNS = [
    (r"\bsupported\s+by\b", "supported by", "positive_attribution"),
    (r"\bdriven\s+by\b", "driven by", "causal_attribution"),
    (r"\bbenefiting\s+from\b", "benefiting from", "positive_attribution"),
    (r"\bdue\s+to\b", "due to", "causal_attribution"),
    (r"\bowing\s+to\b", "owing to", "causal_attribution"),
    (r"\bresulting\s+from\b", "resulting from", "causal_attribution"),
    (r"\breflecting\b", "reflecting", "causal_attribution"),
    (r"\bdespite\b", "despite", "negative_attribution"),
    (r"\bnotwithstanding\b", "notwithstanding", "negative_attribution"),
    (r"\bpartially\s+offset\s+by\b", "partially offset by", "offsetting_attribution"),
    (r"\bmore\s+than\s+offset\b", "more than offset", "offsetting_attribution"),
    (r"\bexcluding\b", "excluding", "exclusion_attribution"),
    (r"\badjusted\s+for\b", "adjusted for", "exclusion_attribution"),
    (r"\bon\s+an?\s+(?:organic|underlying|like.for.like|constant.currency)\s+basis\b",
     "basis adjustment", "basis_attribution"),
    (r"\bone.(?:time|off)\b", "one-time", "one_off_attribution"),
    (r"\bnon.recurring\b", "non-recurring", "one_off_attribution"),
    (r"\bexceptional\b", "exceptional", "one_off_attribution"),
    (r"\bunusual\b", "unusual", "one_off_attribution"),
    (r"\bextraordinary\b", "extraordinary", "one_off_attribution"),
    (r"\btransformation\s+costs?\b", "transformation costs", "one_off_attribution"),
    (r"\brestructuring\b", "restructuring", "one_off_attribution"),
    (r"\bstrategic\s+repositioning\b", "strategic repositioning", "one_off_attribution"),
]

# Temporal — management is qualifying the time horizon
TEMPORAL_PATTERNS = [
    (r"\bin\s+the\s+(?:near|short)\s+term\b", "near term", "short"),
    (r"\bin\s+the\s+medium\s+term\b", "medium term", "medium"),
    (r"\bin\s+the\s+long(?:er)?\s+term\b", "long term", "long"),
    (r"\bcurrently\b", "currently", "present"),
    (r"\bat\s+(?:this|the\s+present)\s+time\b", "at this time", "present"),
    (r"\bgoing\s+forward\b", "going forward", "future"),
    (r"\bover\s+time\b", "over time", "gradual"),
    (r"\btemporarily\b", "temporarily", "temporary"),
    (r"\btransitional\b", "transitional", "temporary"),
    (r"\bseasonal(?:ly|ity)?\b", "seasonal", "cyclical"),
    (r"\bcyclical(?:ly)?\b", "cyclical", "cyclical"),
]

# Confidence / conviction — how sure management sounds
CONFIDENCE_PATTERNS = [
    # High confidence
    (r"\bconfident\b", "confident", "high"),
    (r"\bwell.positioned\b", "well-positioned", "high"),
    (r"\bcommitted\s+to\b", "committed to", "high"),
    (r"\bwe\s+(?:will|shall)\b", "will", "high"),
    (r"\bon\s+track\b", "on track", "high"),
    (r"\bdeliver(?:ing)?\b", "delivering", "high"),
    # Medium confidence
    (r"\bexpect\b", "expect", "medium"),
    (r"\banticipate\b", "anticipate", "medium"),
    (r"\bbelieve\b", "believe", "medium"),
    (r"\bwe\s+see\b", "we see", "medium"),
    (r"\bshould\b", "should", "medium"),
    # Low confidence
    (r"\bhope\b", "hope", "low"),
    (r"\baim\b", "aim", "low"),
    (r"\baspire\b", "aspire", "low"),
    (r"\bmay\b", "may", "low"),
    (r"\bmight\b", "might", "low"),
    (r"\bcould\b", "could", "low"),
    (r"\bpotentially\b", "potentially", "low"),
    (r"\bseek(?:ing)?\s+to\b", "seeking to", "low"),
    (r"\bexploring\b", "exploring", "low"),
]


# ═══════════════════════════════════════════════════════════════════
# Core extraction logic
# ═══════════════════════════════════════════════════════════════════

def _get_context_window(source_text: str, snippet: str, window: int = 200) -> str:
    """Get surrounding context around a source snippet."""
    if not snippet or not source_text:
        return snippet or ""

    # Find the snippet in the source text
    snippet_clean = snippet.strip()[:80]  # Use first 80 chars for matching
    idx = source_text.lower().find(snippet_clean.lower())

    if idx == -1:
        return snippet

    start = max(0, idx - window)
    end = min(len(source_text), idx + len(snippet) + window)
    return source_text[start:end]


def analyse_qualifiers(snippet: str) -> dict:
    """
    Analyse a single source snippet for qualifier patterns.
    Returns a structured qualifier profile.
    """
    if not snippet:
        return {"hedges": [], "attributions": [], "temporals": [], "confidence_level": "neutral"}

    text = snippet.lower()
    result = {
        "hedges": [],
        "attributions": [],
        "temporals": [],
        "confidence_signals": [],
        "confidence_level": "neutral",
        "one_off_flags": [],
    }

    # Hedging
    for pattern, label, strength in HEDGE_PATTERNS:
        if re.search(pattern, text):
            result["hedges"].append({"term": label, "strength": strength})

    # Attribution
    for pattern, label, attr_type in ATTRIBUTION_PATTERNS:
        if re.search(pattern, text):
            entry = {"term": label, "type": attr_type}
            result["attributions"].append(entry)
            if attr_type == "one_off_attribution":
                result["one_off_flags"].append(label)

    # Temporal
    for pattern, label, horizon in TEMPORAL_PATTERNS:
        if re.search(pattern, text):
            result["temporals"].append({"term": label, "horizon": horizon})

    # Confidence
    for pattern, label, level in CONFIDENCE_PATTERNS:
        if re.search(pattern, text):
            result["confidence_signals"].append({"term": label, "level": level})

    # Determine overall confidence level
    levels = [s["level"] for s in result["confidence_signals"]]
    if levels:
        if "low" in levels:
            result["confidence_level"] = "low"
        elif "high" in levels and "low" not in levels:
            result["confidence_level"] = "high"
        else:
            result["confidence_level"] = "medium"

    return result


def enrich_items_with_qualifiers(
    items: list[dict],
    source_text: str = "",
) -> list[dict]:
    """
    Enrich extracted metric items with qualifier metadata.
    Adds a `_qualifiers` field to each item.

    Args:
        items: List of extracted metric dicts
        source_text: Full document text for context expansion

    Returns:
        Items with added qualifier metadata
    """
    hedge_count = 0
    one_off_count = 0
    low_confidence_count = 0

    for item in items:
        snippet = item.get("source_snippet", "") or item.get("guidance_text", "")

        # Expand context if we have the source text
        if source_text:
            context = _get_context_window(source_text, snippet)
        else:
            context = snippet

        qualifiers = analyse_qualifiers(context)
        item["_qualifiers"] = qualifiers

        # Adjust confidence based on hedging
        if qualifiers["hedges"]:
            hedge_count += 1
            # Reduce confidence for heavily hedged items
            current_conf = item.get("confidence", 0.8)
            hedge_penalty = 0.05 * len(qualifiers["hedges"])
            item["confidence"] = max(0.3, current_conf - hedge_penalty)

        # Flag one-off attributions
        if qualifiers["one_off_flags"]:
            one_off_count += 1
            item["_is_one_off"] = True
            item["_one_off_terms"] = qualifiers["one_off_flags"]

        # Track low confidence language
        if qualifiers["confidence_level"] == "low":
            low_confidence_count += 1

    if hedge_count or one_off_count:
        logger.info(
            "Qualifier analysis: %d hedged, %d one-off attributed, %d low-confidence language out of %d items",
            hedge_count, one_off_count, low_confidence_count, len(items),
        )

    return items


def build_document_confidence_profile(items: list[dict]) -> dict:
    """
    Build a document-level confidence profile from qualifier-enriched items.
    Returns a summary of management's language patterns.
    """
    total = len(items)
    if not total:
        return {"total_items": 0, "overall_signal": "insufficient_data"}

    hedged = sum(1 for i in items if i.get("_qualifiers", {}).get("hedges"))
    one_off = sum(1 for i in items if i.get("_is_one_off"))
    low_conf = sum(
        1 for i in items
        if i.get("_qualifiers", {}).get("confidence_level") == "low"
    )
    high_conf = sum(
        1 for i in items
        if i.get("_qualifiers", {}).get("confidence_level") == "high"
    )

    # Collect all unique hedge terms
    all_hedges = set()
    for item in items:
        for h in item.get("_qualifiers", {}).get("hedges", []):
            all_hedges.add(h["term"])

    # Collect all one-off terms
    all_one_offs = set()
    for item in items:
        for t in item.get("_one_off_terms", []):
            all_one_offs.add(t)

    # Determine overall signal
    hedge_rate = hedged / total
    one_off_rate = one_off / total

    if hedge_rate > 0.3 or one_off_rate > 0.2:
        overall = "caution"
    elif high_conf > low_conf and hedge_rate < 0.1:
        overall = "confident"
    else:
        overall = "neutral"

    return {
        "total_items": total,
        "hedged_items": hedged,
        "hedge_rate": round(hedge_rate, 3),
        "one_off_attributed": one_off,
        "one_off_rate": round(one_off_rate, 3),
        "low_confidence_language": low_conf,
        "high_confidence_language": high_conf,
        "hedge_terms_used": sorted(all_hedges),
        "one_off_terms_used": sorted(all_one_offs),
        "overall_signal": overall,
    }
