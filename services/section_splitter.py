"""
Section Splitter — split structured filings into semantic sections.

Splits 10-K, 10-Q, earnings releases, and annual reports into:
  - Financial Statements (income statement, balance sheet, cash flow)
  - MD&A (Management Discussion & Analysis)
  - Notes to Financial Statements
  - Risk Factors
  - Forward-Looking / Guidance
  - Boilerplate (discarded)

Each section gets routed to a specialised prompt + appropriate model tier,
reducing cost (tables → Haiku) and improving accuracy (MD&A → Sonnet with
sector context).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Section types and their characteristics
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FilingSection:
    """A semantically identified section of a filing.

    Named FilingSection to distinguish from the SQLAlchemy DocumentSection
    model in apps/api/models.py (which represents DB rows in document_sections).
    """
    section_type: str           # financial_statements | mda | notes | risk_factors | guidance | cover | boilerplate
    title: str                  # Detected or inferred section title
    text: str                   # The section content
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    has_tables: bool = False    # Hint for model routing
    model_tier: str = "default" # fast | default | advanced
    max_tokens: int = 4096      # Token budget for this section's extraction

# Backwards-compatible alias
DocumentSection = FilingSection


# Section heading patterns — ordered by priority
SECTION_PATTERNS: list[tuple[str, str, str, int]] = [
    # (regex_pattern, section_type, model_tier, max_tokens)

    # ── SEC Item numbers (most reliable signal for 10-K/10-Q) ─
    (r"(?i)item\s+1[\.\s].*financial\s+statements",
     "financial_statements", "fast", 4096),
    (r"(?i)item\s+2[\.\s].*management.{0,5}s?\s+discussion",
     "mda", "default", 8192),
    (r"(?i)item\s+1a[\.\s].*risk\s+factors",
     "risk_factors", "default", 4096),
    (r"(?i)item\s+7[\.\s].*management.{0,5}s?\s+discussion",
     "mda", "default", 8192),
    (r"(?i)item\s+8[\.\s].*financial\s+statements",
     "financial_statements", "fast", 4096),

    # ── Financial Statements (tables → Haiku) ─────────────────
    # Standard + condensed + unaudited variants
    (r"(?i)(condensed\s+)?(consolidated\s+)?(unaudited\s+)?statements?\s+of\s+(income|operations|earnings|net\s+income)",
     "financial_statements", "fast", 4096),
    (r"(?i)(condensed\s+)?(consolidated\s+)?(unaudited\s+)?statements?\s+of\s+comprehensive\s+(income|loss)",
     "financial_statements", "fast", 4096),
    (r"(?i)(condensed\s+)?(consolidated\s+)?(unaudited\s+)?(statements?\s+of\s+)?(financial\s+position|balance\s+sheets?)",
     "financial_statements", "fast", 4096),
    (r"(?i)(condensed\s+)?(consolidated\s+)?(unaudited\s+)?statements?\s+of\s+cash\s+flow",
     "financial_statements", "fast", 4096),
    (r"(?i)(income\s+statement|profit\s+(and|&)\s+loss|p\s*&\s*l)",
     "financial_statements", "fast", 4096),
    (r"(?i)(condensed\s+)?(consolidated\s+)?statement\s+of\s+changes\s+in\s+(equity|stockholders)",
     "financial_statements", "fast", 2048),

    # ── Banking / Insurance specific financial headings ────────
    (r"(?i)(condensed\s+)?(consolidated\s+)?statements?\s+of\s+(financing\s+receivables|loans?\s+and\s+leases?)",
     "financial_statements", "fast", 4096),
    (r"(?i)(net\s+interest\s+(income|margin)|interest\s+income\s+and\s+interest\s+expense)",
     "financial_statements", "fast", 4096),
    (r"(?i)(provision|allowance)\s+(for\s+)?(credit|loan)\s+losses",
     "financial_statements", "fast", 4096),
    (r"(?i)(investment\s+portfolio|securities\s+(available|held))",
     "financial_statements", "fast", 4096),
    (r"(?i)(insurance\s+)?underwriting\s+(results|summary|income)",
     "financial_statements", "fast", 4096),
    (r"(?i)(claims?\s+reserves?|loss\s+reserves?|combined\s+ratio)",
     "financial_statements", "fast", 4096),
    (r"(?i)(capital\s+adequacy|regulatory\s+capital|tier\s+1\s+capital|cet\s*1)",
     "financial_statements", "fast", 4096),
    (r"(?i)(deposit\s+composition|funding\s+sources|liquidity\s+position)",
     "financial_statements", "fast", 4096),

    # ── Summary / highlights ──────────────────────────────────
    (r"(?i)(financial\s+highlights|key\s+figures|results?\s+at\s+a\s+glance|summary\s+financials)",
     "financial_statements", "fast", 4096),
    (r"(?i)(headline\s+results|group\s+results|financial\s+results\s+summary)",
     "financial_statements", "fast", 4096),
    (r"(?i)(selected\s+financial\s+data|five.year\s+summary)",
     "financial_statements", "fast", 4096),

    # ── MD&A (narrative → Sonnet with sector context) ─────────
    (r"(?i)management.{0,5}s?\s+discussion\s+(and|&)\s+analysis",
     "mda", "default", 8192),
    (r"(?i)(business\s+review|operating\s+review|operational\s+review)",
     "mda", "default", 8192),
    (r"(?i)(chief\s+executive.{0,5}s?\s+(review|report|statement))",
     "mda", "default", 6144),
    (r"(?i)(chief\s+financial\s+officer.{0,5}s?\s+(review|report|statement))",
     "mda", "default", 6144),
    (r"(?i)(strategic\s+report|performance\s+review|segment\s+review)",
     "mda", "default", 8192),
    # Banking MD&A sections
    (r"(?i)(credit\s+quality\s+review|asset\s+quality\s+review|loan\s+portfolio\s+review)",
     "mda", "default", 8192),
    (r"(?i)(segment\s+results|business\s+segment\s+discussion|reportable\s+segments?)",
     "mda", "default", 8192),

    # ── Notes to Financial Statements ─────────────────────────
    (r"(?i)notes\s+to\s+(the\s+)?(condensed\s+)?(consolidated\s+)?(unaudited\s+)?financial\s+statements",
     "notes", "default", 6144),
    (r"(?i)(significant\s+accounting\s+policies|basis\s+of\s+preparation)",
     "notes", "default", 4096),

    # ── Risk Factors ──────────────────────────────────────────
    (r"(?i)(risk\s+factors|principal\s+risks|key\s+risks)",
     "risk_factors", "default", 4096),

    # ── Forward-Looking / Guidance ────────────────────────────
    (r"(?i)(outlook|guidance|forward.looking|expectations?\s+for)",
     "guidance", "default", 4096),

    # ── Boilerplate (skip) ────────────────────────────────────
    (r"(?i)(forward.looking\s+statements?\s+disclaimer|safe\s+harbor|legal\s+disclaimer)",
     "boilerplate", "fast", 0),
    (r"(?i)(table\s+of\s+contents|about\s+this\s+report|glossary)",
     "cover", "fast", 0),
]


# ═══════════════════════════════════════════════════════════════════
# Splitting logic
# ═══════════════════════════════════════════════════════════════════

def _detect_section_type(heading: str) -> tuple[str, str, int]:
    """Match a heading against known section patterns."""
    for pattern, section_type, tier, max_tok in SECTION_PATTERNS:
        if re.search(pattern, heading):
            return section_type, tier, max_tok
    return "other", "default", 4096


def _has_table_content(text: str) -> bool:
    """Heuristic: does this text block contain tabular data?"""
    lines = text.split("\n")
    numeric_lines = 0
    for line in lines:
        # Lines with multiple numbers separated by whitespace/tabs = table rows
        nums = re.findall(r'[\d,]+\.?\d*', line)
        if len(nums) >= 3:
            numeric_lines += 1
    return numeric_lines >= 5


def _find_section_boundaries(text: str) -> list[tuple[int, str, str]]:
    """
    Find section heading positions in the text.
    Returns list of (char_position, heading_text, section_type).
    """
    boundaries = []
    lines = text.split("\n")
    pos = 0

    for line in lines:
        stripped = line.strip()

        # Heuristic for section headings:
        # - All caps or title case
        # - Relatively short (< 100 chars)
        # - Not a table row (no multiple numbers)
        if (
            stripped
            and len(stripped) < 120
            and len(re.findall(r'[\d,]+\.?\d*', stripped)) < 3
        ):
            # Check if it matches a known pattern
            section_type, tier, max_tok = _detect_section_type(stripped)
            if section_type not in ("other",):
                boundaries.append((pos, stripped, section_type))

        pos += len(line) + 1  # +1 for \n

    return boundaries


def split_into_sections(
    text: str,
    doc_type: str = "10-K",
    page_texts: list[dict] | None = None,
) -> list[FilingSection]:
    """
    Split a filing into semantic sections.

    Args:
        text: Full document text
        doc_type: Document type for heuristic tuning
        page_texts: Optional list of {page, text} for page tracking

    Returns:
        List of FilingSection objects, each tagged with section_type,
        model_tier, and max_tokens for downstream routing.
    """
    boundaries = _find_section_boundaries(text)

    if not boundaries:
        # No sections detected — fall back to smart chunking
        logger.info("Section splitter: no sections detected, returning full text")
        return [FilingSection(
            section_type="full_document",
            title=doc_type,
            text=text,
            has_tables=_has_table_content(text),
            model_tier="default",
            max_tokens=8192,
        )]

    sections = []

    # Add content before first boundary as "preamble"
    first_pos = boundaries[0][0]
    if first_pos > 200:
        preamble = text[:first_pos].strip()
        if preamble and not _is_boilerplate(preamble):
            sections.append(FilingSection(
                section_type="preamble",
                title="Document Header / Summary",
                text=preamble,
                has_tables=_has_table_content(preamble),
                model_tier="fast",
                max_tokens=2048,
            ))

    # Track covered character ranges for uncovered text detection
    covered_ranges = []

    # Process each section boundary
    for i, (pos, heading, section_type) in enumerate(boundaries):
        # Section text runs from this heading to the next heading
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][0]
        else:
            end_pos = len(text)

        section_text = text[pos:end_pos].strip()

        if not section_text or len(section_text) < 50:
            continue

        covered_ranges.append((pos, end_pos))

        # Skip boilerplate
        if section_type in ("boilerplate", "cover"):
            logger.debug("Section splitter: skipping %s section '%s'", section_type, heading[:60])
            continue

        # Determine model tier and detect tables
        has_tables = _has_table_content(section_text)
        _, tier, max_tok = _detect_section_type(heading)

        # Override: if a section has dense tables, use fast tier
        if has_tables and section_type == "financial_statements":
            tier = "fast"

        sections.append(FilingSection(
            section_type=section_type,
            title=heading[:120],
            text=section_text,
            has_tables=has_tables,
            model_tier=tier,
            max_tokens=max_tok,
        ))

    # If we found sections but they don't cover much of the document,
    # extract only the uncovered ranges (not the full document again)
    covered_chars = sum(end - start for start, end in covered_ranges)
    if covered_chars < len(text) * 0.3:
        logger.info(
            "Section splitter: only %.0f%% covered, extracting uncovered ranges",
            covered_chars / len(text) * 100,
        )
        uncovered_text = _extract_uncovered_text(text, covered_ranges)
        if uncovered_text and len(uncovered_text.strip()) > 200:
            sections.append(FilingSection(
                section_type="uncovered",
                title="Remaining Content",
                text=uncovered_text,
                has_tables=_has_table_content(uncovered_text),
                model_tier="default",
                max_tokens=8192,
            ))

    logger.info(
        "Section splitter: %d sections from %s (types: %s)",
        len(sections),
        doc_type,
        ", ".join(s.section_type for s in sections),
    )

    return sections


def _is_boilerplate(text: str) -> bool:
    """Check if text is likely boilerplate."""
    lower = text.lower()
    boilerplate_signals = [
        "forward-looking statements",
        "safe harbor",
        "this document is not",
        "legal disclaimer",
        "important notice",
        "this report has been prepared",
    ]
    return any(signal in lower for signal in boilerplate_signals)


def _extract_uncovered_text(text: str, covered_ranges: list[tuple[int, int]]) -> str:
    """Extract text not covered by any detected section, skipping boilerplate."""
    if not covered_ranges:
        return text

    # Sort and merge overlapping ranges
    sorted_ranges = sorted(covered_ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Collect gaps
    uncovered_parts = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            gap = text[prev_end:start].strip()
            if gap and len(gap) > 100 and not _is_boilerplate(gap):
                uncovered_parts.append(gap)
        prev_end = end
    # Trailing text
    if prev_end < len(text):
        gap = text[prev_end:].strip()
        if gap and len(gap) > 100 and not _is_boilerplate(gap):
            uncovered_parts.append(gap)

    return "\n\n".join(uncovered_parts)


# ═══════════════════════════════════════════════════════════════════
# Convenience: get sections suitable for specific extraction types
# ═══════════════════════════════════════════════════════════════════

def get_financial_sections(sections: list[FilingSection]) -> list[FilingSection]:
    """Return only sections containing financial statements / tables."""
    return [s for s in sections if s.section_type == "financial_statements"]


def get_narrative_sections(sections: list[FilingSection]) -> list[FilingSection]:
    """Return MD&A, guidance, and other narrative sections."""
    return [s for s in sections if s.section_type in ("mda", "guidance", "preamble")]


def get_notes_sections(sections: list[FilingSection]) -> list[FilingSection]:
    """Return notes to financial statements."""
    return [s for s in sections if s.section_type == "notes"]


def get_risk_sections(sections: list[FilingSection]) -> list[FilingSection]:
    """Return risk factor sections."""
    return [s for s in sections if s.section_type == "risk_factors"]
