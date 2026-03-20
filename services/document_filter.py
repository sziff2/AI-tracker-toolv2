"""
services/document_filter.py

Document-type-aware pre-filter layer.
Sits between raw PDF text extraction and the LLM extraction calls.

Purpose: remove sections that have zero analytical value BEFORE
any LLM token is spent on them. Each doc type has its own
keep/skip rules based on what analysts actually care about.

Usage:
    from services.document_filter import filter_document
    sections = filter_document(pages, document_type)
    # sections is a list of FilteredSection objects ready for extraction

Returns FilteredSection objects with:
    - text: cleaned text for that section
    - section_type: financials | mda | risk_factors | guidance |
                    qa | compensation | strategic | other
    - priority: high | medium | low
    - page_start / page_end: provenance
    - skip_reason: why it was skipped (if applicable)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class FilteredSection:
    text: str
    section_type: str          # financials | mda | risk_factors | guidance | qa | compensation | strategic | other
    priority: str              # high | medium | skip
    page_start: int
    page_end: int
    char_count: int = 0
    skip_reason: Optional[str] = None

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class FilterResult:
    sections: list[FilteredSection] = field(default_factory=list)
    total_pages: int = 0
    pages_kept: int = 0
    pages_skipped: int = 0
    chars_before: int = 0
    chars_after: int = 0
    reduction_pct: float = 0.0

    def summary(self) -> str:
        return (
            f"Filtered {self.total_pages} pages: kept {self.pages_kept}, "
            f"skipped {self.pages_skipped} "
            f"({self.reduction_pct:.0f}% token reduction)"
        )


# ─────────────────────────────────────────────────────────────────
# Universal skip patterns — apply to ALL document types
# ─────────────────────────────────────────────────────────────────

UNIVERSAL_SKIP_PATTERNS = [
    r'(?i)^(this\s+)?forward[\s\-]looking\s+statement',
    r'(?i)safe\s+harbor\s+(statement|notice|language)',
    r'(?i)this\s+(page|slide)\s+(has\s+been\s+)?intentionally\s+left\s+blank',
    r'(?i)^\s*legal\s+disclaimer\s*$',
    r'(?i)^\s*important\s+(notice|information|disclosures?)\s*$',
    r'(?i)reproduction\s+(prohibited|without\s+permission)',
    r'(?i)for\s+institutional\s+investors?\s+only',
    r'(?i)not\s+for\s+(retail|public)\s+(distribution|use)',
    r'(?i)^\s*table\s+of\s+contents?\s*$',
    r'(?i)^\s*index\s*$',
    r'(?i)^\s*page\s+\d+\s*(of\s+\d+)?\s*$',
]

def _is_universally_skippable(text: str) -> Optional[str]:
    """Return skip reason if text matches universal skip patterns, else None."""
    stripped = text.strip()
    if len(stripped) < 30:
        return "too_short"
    # Very low information density — mostly numbers/symbols, no words
    words = re.findall(r'[a-zA-Z]{3,}', stripped)
    if len(words) < 3 and len(stripped) < 200:
        return "no_words"
    for pat in UNIVERSAL_SKIP_PATTERNS:
        if re.search(pat, stripped[:500]):
            return f"pattern:{pat[:40]}"
    return None


# ─────────────────────────────────────────────────────────────────
# Section classifiers — identify what type of section this is
# ─────────────────────────────────────────────────────────────────

SECTION_PATTERNS = {
    "financials": [
        r'(?i)(consolidated\s+)?(statements?\s+of\s+(income|operations|earnings|profit))',
        r'(?i)(consolidated\s+)?balance\s+sheet',
        r'(?i)(consolidated\s+)?cash\s+flow\s+statement',
        r'(?i)income\s+statement',
        r'(?i)profit\s+(and\s+loss|&\s+loss)',
        r'(?i)financial\s+(results|highlights|summary)',
        r'(?i)(selected\s+)?financial\s+data',
        r'(?i)segment\s+(results|revenue|profit|performance|information)',
        r'(?i)(quarterly|annual)\s+(financial\s+)?results',
        r'(?i)key\s+(financial\s+)?metrics',
        r'(?i)revenue\s+breakdown',
        r'(?i)EBITDA|EBIT|operating\s+profit',
    ],
    "mda": [
        r"(?i)management.{0,5}s?\s+discussion\s+and\s+analysis",
        r'(?i)MD&A',
        r'(?i)results\s+of\s+operations',
        r'(?i)operating\s+(results|review|performance)',
        r'(?i)business\s+(overview|review|performance)',
        r'(?i)executive\s+(summary|overview)',
        r'(?i)group\s+(chief|ceo|cfo).{0,10}(review|statement|letter)',
        r'(?i)chairman.{0,5}s?\s+(letter|statement|review)',
        r'(?i)liquidity\s+and\s+capital\s+resources',
        r'(?i)critical\s+accounting\s+(policies|estimates)',
    ],
    "risk_factors": [
        r'(?i)risk\s+factors',
        r'(?i)principal\s+risks',
        r'(?i)key\s+risks',
        r'(?i)risk\s+management',
        r'(?i)risks?\s+and\s+uncertainties',
        r'(?i)material\s+risks?',
    ],
    "guidance": [
        r'(?i)outlook',
        r'(?i)guidance',
        r'(?i)forward.{0,5}looking',
        r'(?i)financial\s+(targets?|objectives?|goals?)',
        r'(?i)medium[\s\-]term\s+(targets?|ambitions?|objectives?)',
        r'(?i)(full[\s\-]year|fiscal\s+(year|20\d\d))\s+(expectations?|guidance|outlook)',
        r'(?i)next\s+(quarter|year|period)\s+(we\s+expect|outlook)',
    ],
    "qa": [
        r'(?i)question[\s\-]and[\s\-]answer',
        r'(?i)q&a\s+session',
        r'(?i)analyst\s+questions?',
        r'(?i)^\s*operator[\s:,]',
        r'(?i)^\s*(question|answer)\s*[\:\-]',
        r'(?i)unidentified\s+analyst',
        r'(?i)(opening|closing)\s+remarks',
    ],
    "compensation": [
        r'(?i)executive\s+compensation',
        r'(?i)compensation\s+discussion\s+and\s+analysis',
        r'(?i)CD&A',
        r'(?i)named\s+executive\s+officers?',
        r'(?i)summary\s+compensation\s+table',
        r'(?i)pay\s+(ratio|mix|structure)',
        r'(?i)long[\s\-]term\s+incentive',
        r'(?i)annual\s+(bonus|incentive)\s+plan',
    ],
    "strategic": [
        r'(?i)strategic\s+(priorities|objectives|review|direction)',
        r'(?i)capital\s+allocation',
        r'(?i)capital\s+(expenditure|investment)\s+(plan|programme|program)',
        r'(?i)M&A\s+(strategy|pipeline|activity)',
        r'(?i)acquisition\s+(strategy|activity)',
        r'(?i)shareholder\s+(returns?|value)',
        r'(?i)(share\s+)?buyback\s+(programme|program)',
        r'(?i)dividend\s+(policy|growth|increase)',
        r'(?i)market\s+(position|share|opportunity)',
        r'(?i)competitive\s+(advantage|position|landscape)',
    ],
}

def _classify_section(text: str) -> tuple[str, str]:
    """
    Returns (section_type, priority).
    priority: high | medium | low
    """
    sample = text[:1000].lower()

    HIGH_PRIORITY_TYPES = {"financials", "mda", "guidance"}
    MEDIUM_PRIORITY_TYPES = {"risk_factors", "strategic", "qa", "compensation"}

    for section_type, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, sample):
                priority = "high" if section_type in HIGH_PRIORITY_TYPES else "medium"
                return section_type, priority

    # Heuristic: high number density likely = financial data
    numbers = re.findall(r'\b\d[\d,\.]+\b', text[:2000])
    words = re.findall(r'[a-zA-Z]{4,}', text[:2000])
    if len(numbers) > 20 and len(words) > 30:
        return "financials", "high"

    return "other", "low"


# ─────────────────────────────────────────────────────────────────
# Per-doc-type skip rules
# ─────────────────────────────────────────────────────────────────

# Patterns that indicate a section should be SKIPPED for each doc type
DOC_TYPE_SKIP_PATTERNS: dict[str, list[str]] = {

    "10-K": [
        r'(?i)^exhibit\s+\d',
        r'(?i)^item\s+\d+[a-z]?\s*[\.\:\-]\s*$',
        r'(?i)list\s+of\s+(subsidiaries|exhibits)',
        r'(?i)subsidiaries\s+of\s+the\s+(company|registrant)',
        r'(?i)consent\s+of\s+independent\s+(registered|certified)',
        r'(?i)certification\s+(pursuant\s+to|of\s+)?(rule|section)\s+\d',
        r'(?i)sarbanes[\s\-]oxley',
        r'(?i)section\s+302\s+certification',
        r'(?i)section\s+906\s+certification',
        r'(?i)power\s+of\s+attorney',
        r'(?i)signatures?\s*page',
        r'(?i)pursuant\s+to\s+(rule|regulation)\s+[\d\-]+',
        r'(?i)incorporated\s+herein\s+by\s+reference',
        r'(?i)index\s+to\s+financial\s+statements',
        r'(?i)report\s+of\s+independent\s+(registered|certified)',
        r'(?i)auditor.{0,5}s?\s+report\s+on\s+internal\s+control',
        r'(?i)internal\s+control\s+over\s+financial\s+reporting\s+(–|—|-)\s+overview',
        r'(?i)changes\s+in\s+and\s+disagreements',
        r'(?i)mine\s+safety\s+disclosures?',
        r'(?i)unresolved\s+staff\s+comments',
        r'(?i)properties?\s*\n',  # "Item 2. Properties" section (real estate disclosures)
        r'(?i)legal\s+proceedings',
        r'(?i)quantitative\s+and\s+qualitative\s+disclosures\s+about\s+market\s+risk',
    ],

    "annual_report": [
        r'(?i)notice\s+of\s+(annual\s+)?general\s+meeting',
        r'(?i)shareholder\s+(notice|letter)\s+for\s+the\s+agm',
        r'(?i)agm\s+notice',
        r'(?i)form\s+of\s+proxy',
        r'(?i)glossary\s+of\s+terms',
        r'(?i)^\s*definitions?\s*$',
        r'(?i)five[\s\-]year\s+(financial\s+)?summary',  # usually duplicated in MD&A
        r'(?i)independent\s+auditor.{0,5}s?\s+report',
        r'(?i)directors?\s+responsibility\s+statement',
    ],

    "earnings_release": [
        r'(?i)(this\s+)?press\s+release\s+contains?\s+forward[\s\-]looking',
        r'(?i)non[\s\-]gaap\s+(financial\s+measures?|reconciliation)',  # reconciliation tables often noise
        r'(?i)about\s+\[?company\s+name\]?',
        r'(?i)about\s+[A-Z][a-z]+\s*\n',  # "About Heineken" boilerplate
        r'(?i)for\s+(further\s+)?information\s+(please\s+)?contact',
        r'(?i)investor\s+relations\s+contact',
        r'(?i)media\s+contact',
        r'(?i)^\s*\*\s*\*\s*\*\s*$',  # *** separator lines
    ],

    "transcript": [
        r'(?i)^\s*operator[\s:,]\s*(please|thank|good|ladies)',
        r'(?i)this\s+conference\s+(call|is)\s+being\s+recorded',
        r'(?i)please\s+(go\s+ahead|proceed|state\s+your\s+name)',
        r'(?i)^\s*thank\s+you[\.\,]\s*(and\s+)?good\s+(morning|afternoon|evening)',
        r'(?i)I\s+would\s+now\s+like\s+to\s+turn\s+the\s+(call|conference)',
        r'(?i)this\s+(concludes|ends)\s+(today.{0,10})?(conference\s+call|presentation|call)',
        r'(?i)participants?\s+(list|on\s+the\s+call)',
        r'(?i)replay\s+(of\s+this\s+call|instructions?)',
        r'(?i)^\s*(good\s+morning|good\s+afternoon|good\s+evening|hello|hi)\s*[,\.]?\s*everyone',
    ],

    "broker_note": [
        r'(?i)important\s+disclosures?',
        r'(?i)analyst\s+certification',
        r'(?i)required\s+disclosures?',
        r'(?i)regulation\s+(ac|fd)',
        r'(?i)(member|member\s+of)\s+(SIPC|FINRA|FCA)',
        r'(?i)this\s+(research\s+)?(report|note)\s+(has\s+been\s+)?prepared\s+by',
        r'(?i)distribution\s+(of\s+)?this\s+(report|note|document)',
        r'(?i)conflicts?\s+of\s+interest',
        r'(?i)price\s+(performance\s+)?chart',
        r'(?i)rating\s+history',
        r'(?i)^\s*disclosures?\s+appendix',
        r'(?i)for\s+important\s+disclosures,?\s+(please|see|visit)',
        r'(?i)this\s+material\s+is\s+(for|intended\s+for)\s+(qualified|sophisticated|institutional)',
    ],

    "proxy_statement": [
        r'(?i)how\s+to\s+vote\s+(your\s+shares)?',
        r'(?i)voting\s+(instructions?|procedures?|mechanics)',
        r'(?i)attending\s+the\s+(annual\s+)?meeting',
        r'(?i)quorum\s+requirements?',
        r'(?i)submission\s+of\s+(future\s+)?shareholder\s+proposals?',
        r'(?i)deadline\s+for\s+submitting',
        r'(?i)how\s+(shares\s+)?are\s+counted',
        r'(?i)appraisal\s+rights',
        r'(?i)cost\s+of\s+proxy\s+solicitation',
        r'(?i)the\s+proxy\s+(card|voting)',
        r'(?i)internet\s+(voting|access)',
        r'(?i)telephon(e|ic)\s+voting',
    ],

    "sustainability_report": [
        r'(?i)about\s+this\s+report',
        r'(?i)reporting\s+(framework|standards?|methodology)',
        r'(?i)GRI\s+(content\s+)?index',
        r'(?i)SASB\s+(index|disclosure)',
        r'(?i)TCFD\s+(index|alignment)',
        r'(?i)assurance\s+(statement|report)',
        r'(?i)independent\s+(assurance|verification)',
        r'(?i)third[\s\-]party\s+(assurance|verification)',
        r'(?i)^\s*glossary\s*$',
        r'(?i)ESG\s+(data\s+)?appendix',
    ],
}


# ─────────────────────────────────────────────────────────────────
# Core filter function
# ─────────────────────────────────────────────────────────────────

def filter_document(
    pages: list[dict],
    document_type: str,
    min_chars: int = 100,
) -> FilterResult:
    """
    Filter raw PDF pages into analytically relevant sections.

    Args:
        pages: list of {"page": int, "text": str} from document_parser
        document_type: e.g. "10-K", "earnings_release", "transcript"
        min_chars: minimum characters for a page to be considered

    Returns:
        FilterResult with kept sections and filtering stats
    """
    result = FilterResult(total_pages=len(pages))
    doc_skip_patterns = DOC_TYPE_SKIP_PATTERNS.get(document_type, [])

    chars_before = sum(len(p.get("text", "")) for p in pages)
    result.chars_before = chars_before

    # Group pages into sections by scanning for section headers
    current_section_pages = []
    current_section_start = 1

    def flush_section(pages_in_section: list[dict], start_page: int):
        """Combine pages into a section and classify it."""
        if not pages_in_section:
            return

        combined_text = "\n\n".join(p.get("text", "") for p in pages_in_section).strip()
        end_page = pages_in_section[-1].get("page", start_page)

        # Universal skip check
        skip_reason = _is_universally_skippable(combined_text[:500])
        if skip_reason and len(combined_text) < 500:
            result.pages_skipped += len(pages_in_section)
            return

        # Doc-type-specific skip check (check first 800 chars = section header area)
        header_text = combined_text[:800]
        for pat in doc_skip_patterns:
            if re.search(pat, header_text):
                result.pages_skipped += len(pages_in_section)
                skip_section = FilteredSection(
                    text="",
                    section_type="skipped",
                    priority="skip",
                    page_start=start_page,
                    page_end=end_page,
                    skip_reason=f"doc_type_rule:{pat[:50]}",
                )
                result.sections.append(skip_section)
                return

        # Classify and keep
        section_type, priority = _classify_section(combined_text)

        kept_section = FilteredSection(
            text=combined_text,
            section_type=section_type,
            priority=priority,
            page_start=start_page,
            page_end=end_page,
        )
        result.sections.append(kept_section)
        result.pages_kept += len(pages_in_section)
        result.chars_after += len(combined_text)

    # Process pages — group into logical sections
    # A new section starts when a page begins with what looks like a heading
    SECTION_BREAK_PATTERN = re.compile(
        r'^(?:'
        r'item\s+\d+|'          # "Item 1A.", "Item 7."
        r'part\s+[IVX]+|'       # "Part II"
        r'section\s+\d+|'       # "Section 4"
        r'[A-Z][A-Z\s]{5,40}\n|'  # ALL CAPS HEADING
        r'(?:management|chairman|ceo|cfo|directors?).{0,30}(?:discussion|review|statement|letter)'
        r')',
        re.IGNORECASE | re.MULTILINE,
    )

    for page in pages:
        page_text = page.get("text", "").strip()
        page_num = page.get("page", 0)

        if len(page_text) < min_chars:
            result.pages_skipped += 1
            continue

        # Check if this page starts a new section
        is_section_break = bool(SECTION_BREAK_PATTERN.match(page_text[:200]))

        if is_section_break and current_section_pages:
            flush_section(current_section_pages, current_section_start)
            current_section_pages = [page]
            current_section_start = page_num
        else:
            current_section_pages.append(page)
            if not current_section_pages or page_num < current_section_start:
                current_section_start = page_num

    # Flush remaining pages
    flush_section(current_section_pages, current_section_start)

    # Compute stats
    result.total_pages = len(pages)
    if chars_before > 0:
        result.reduction_pct = (1 - result.chars_after / chars_before) * 100

    logger.info(
        "Document filter [%s]: %d pages → %d kept / %d skipped | "
        "%d chars → %d chars (%.0f%% reduction)",
        document_type, result.total_pages, result.pages_kept, result.pages_skipped,
        result.chars_before, result.chars_after, result.reduction_pct,
    )

    return result


def get_high_value_text(filter_result: FilterResult, max_chars: int = 150_000) -> str:
    """
    Get a single string of the highest-priority content.
    Prioritises: high > medium > low sections.
    Respects max_chars budget.
    Used when you want to pass filtered content directly to extraction.
    """
    priority_order = {"high": 0, "medium": 1, "low": 2, "skip": 99}
    kept = [s for s in filter_result.sections if s.priority != "skip" and s.text]
    kept.sort(key=lambda s: priority_order.get(s.priority, 99))

    parts = []
    total = 0
    for section in kept:
        if total + section.char_count > max_chars:
            remaining = max_chars - total
            if remaining > 500:
                parts.append(section.text[:remaining])
            break
        parts.append(section.text)
        total += section.char_count

    return "\n\n---\n\n".join(parts)


def get_sections_by_type(
    filter_result: FilterResult,
    section_type: str,
) -> list[FilteredSection]:
    """Get all kept sections of a specific type. Useful for targeted extraction."""
    return [
        s for s in filter_result.sections
        if s.section_type == section_type and s.priority != "skip" and s.text
    ]
