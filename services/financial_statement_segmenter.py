"""
Financial statement segmenter — structural parser (NO LLM calls).

Pre-segments financial documents into classified tables with period/currency/unit
metadata so downstream LLM extraction can be targeted and focused.

Uses pdfplumber table data + page text heuristics.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Enums & Data Models ─────────────────────────────────────────

class StatementType(str, Enum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    SEGMENT_BREAKDOWN = "segment"
    EQUITY_CHANGES = "equity_changes"
    KPI_TABLE = "kpi"
    GUIDANCE = "guidance"
    NARRATIVE = "narrative"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


@dataclass
class FinancialTable:
    statement_type: StatementType
    period: str                        # "Q4 2025" or "FY 2025" or "Dec 2025"
    period_type: str                   # "quarter" | "annual" | "point_in_time"
    segment: str | None = None         # "Europe" | None = consolidated
    currency: str = "USD"
    unit_scale: str = "millions"
    rows: list[dict] = field(default_factory=list)   # [{label, value, raw_text}]
    source_page: int = 0
    is_current: bool = False
    raw_table: list[list] = field(default_factory=list)


@dataclass
class FinancialDocumentStructure:
    tables: list[FinancialTable] = field(default_factory=list)
    narrative_sections: list[dict] = field(default_factory=list)
    footnotes: list[dict] = field(default_factory=list)
    document_metadata: dict = field(default_factory=dict)


# ── Keyword sets ─────────────────────────────────────────────────

_INCOME_KEYWORDS = {
    # Standard industrial P&L
    "revenue", "net sales", "total revenue", "operating profit", "operating income",
    "ebit", "ebitda", "net income", "net profit", "net loss", "eps",
    "earnings per share", "gross profit", "gross margin", "cost of sales",
    "cost of goods sold", "cost of revenue", "sg&a",
    "selling general and administrative", "operating expenses",
    "income from operations", "profit before tax", "income before tax",
    "diluted eps", "basic eps",
    # Banking P&L
    "net interest income", "net financing revenue", "total net revenue",
    "provision for credit losses", "noninterest income", "noninterest expense",
    "non-interest income", "non-interest expense", "total interest expense",
    "total interest income", "net interest margin",
    "financing revenue", "total financing revenue",
    "insurance premiums", "compensation and benefits",
    "pre-provision profit", "pre-tax income",
    # Insurance P&L
    "net premiums written", "net premiums earned", "combined ratio",
    "loss ratio", "expense ratio", "underwriting income",
    "claims incurred", "investment income",
}

_BALANCE_SHEET_KEYWORDS = {
    "total assets", "total liabilities", "shareholders equity",
    "stockholders equity", "shareholders' equity", "stockholders' equity",
    "current assets", "non-current assets", "noncurrent assets",
    "current liabilities", "non-current liabilities", "noncurrent liabilities",
    "cash and cash equivalents", "total debt", "net debt", "goodwill",
    "intangible assets", "property plant and equipment",
    "accounts receivable", "accounts payable", "inventories",
    "retained earnings", "total equity",
}

# Keywords that are P&L for banks but look like BS to generic classifier
_BANK_PL_OVERRIDE_KEYWORDS = {
    "net interest income", "net financing revenue", "total net revenue",
    "provision for credit losses", "noninterest expense", "noninterest income",
    "total interest expense", "total interest income",
    "financing revenue", "total financing revenue",
    "compensation and benefits", "insurance premiums",
    "income from continuing operations", "net depreciation expense",
}

# Keywords that are P&L for insurance
_INSURANCE_PL_OVERRIDE_KEYWORDS = {
    "net premiums written", "net premiums earned", "claims incurred",
    "underwriting income", "combined ratio", "loss ratio", "expense ratio",
    "acquisition costs", "investment income", "realised gains",
}

_CASH_FLOW_KEYWORDS = {
    "operating cash flow", "cash from operations", "cash flow from operations",
    "cash generated from operations", "capital expenditure", "capex",
    "free cash flow", "investing activities", "financing activities",
    "cash from investing", "cash from financing", "dividends paid",
    "purchase of property", "depreciation and amortization",
    "depreciation & amortization", "change in working capital",
    "net cash from operating", "net cash used in investing",
    "net cash used in financing", "repurchase of shares",
}

_SEGMENT_PATTERNS = re.compile(
    r"segment|by\s+region|by\s+division|by\s+geography|by\s+business\s+line"
    r"|regional\s+breakdown|divisional",
    re.IGNORECASE,
)

_EQUITY_KEYWORDS = {
    "changes in equity", "statement of equity", "changes in shareholders",
    "equity attributable", "comprehensive income",
}

_KPI_KEYWORDS = {
    "key performance", "kpi", "operating metrics", "operating statistics",
    "store count", "same-store", "like-for-like", "subscriber",
}

_GUIDANCE_KEYWORDS = {
    "guidance", "outlook", "forecast", "expects", "expected to",
    "full year target", "range of", "anticipated",
}

# ── Month helpers ────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _extract_month(text: str) -> int | None:
    """Extract month number from text."""
    lower = text.lower().strip()
    for name, num in _MONTH_MAP.items():
        if name in lower:
            return num
    # Try numeric month in date patterns like 12/31/2025 or 31/12/2025
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", lower)
    if m:
        month_candidate = int(m.group(1))
        if 1 <= month_candidate <= 12:
            return month_candidate
    return None


def _month_to_quarter(month: int) -> str:
    """Map month number to quarter label."""
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"


def _extract_year(text: str) -> int | None:
    """Extract a 4-digit year from text."""
    m = re.search(r"(20\d{2})", text)
    if m:
        return int(m.group(1))
    # 2-digit year: 24, 25, 26
    m = re.search(r"(?<!\d)(\d{2})(?!\d)", text)
    if m:
        yr = int(m.group(1))
        if 15 <= yr <= 40:
            return 2000 + yr
    return None


# ── Core functions ───────────────────────────────────────────────

def _flatten_table_text(table: list[list]) -> str:
    """Flatten all table cells into one lowercase string for keyword matching."""
    parts = []
    for row in table:
        for cell in row:
            if cell is not None:
                parts.append(str(cell).lower().strip())
    return " ".join(parts)


def classify_table(table: list[list], page_text: str, page_num: int, sector: str = None) -> StatementType:
    """
    Classify a table as income statement, balance sheet, cash flow, etc.
    Uses keyword matching against row labels and surrounding page text.
    Sector-aware: banks and insurers get boosted P&L detection.
    """
    if not table or len(table) < 2:
        return StatementType.UNKNOWN

    flat = _flatten_table_text(table)
    context = (flat + " " + page_text.lower()).strip()

    # Score each statement type
    scores = {
        StatementType.INCOME_STATEMENT: 0,
        StatementType.BALANCE_SHEET: 0,
        StatementType.CASH_FLOW: 0,
    }

    for kw in _INCOME_KEYWORDS:
        if kw in context:
            scores[StatementType.INCOME_STATEMENT] += 1

    for kw in _BALANCE_SHEET_KEYWORDS:
        if kw in context:
            scores[StatementType.BALANCE_SHEET] += 1

    for kw in _CASH_FLOW_KEYWORDS:
        if kw in context:
            scores[StatementType.CASH_FLOW] += 1

    # Sector-specific P&L boost: banks and insurers have P&L keywords
    # that the generic classifier mistakes for BS
    sector_lower = (sector or "").lower()
    is_bank = any(k in sector_lower for k in ["financ", "bank", "lending", "credit"])
    is_insurance = any(k in sector_lower for k in ["insur", "underwrit"])

    if is_bank:
        for kw in _BANK_PL_OVERRIDE_KEYWORDS:
            if kw in context:
                scores[StatementType.INCOME_STATEMENT] += 2  # strong boost
    if is_insurance:
        for kw in _INSURANCE_PL_OVERRIDE_KEYWORDS:
            if kw in context:
                scores[StatementType.INCOME_STATEMENT] += 2

    # Check special types first (before falling through to the big 3)
    if _SEGMENT_PATTERNS.search(context):
        # Only classify as segment if there aren't strong P&L/BS/CF signals
        best_score = max(scores.values())
        if best_score < 3:
            return StatementType.SEGMENT_BREAKDOWN

    for kw in _EQUITY_KEYWORDS:
        if kw in context:
            return StatementType.EQUITY_CHANGES

    for kw in _KPI_KEYWORDS:
        if kw in context:
            best_score = max(scores.values())
            if best_score < 3:
                return StatementType.KPI_TABLE

    for kw in _GUIDANCE_KEYWORDS:
        if kw in context:
            best_score = max(scores.values())
            if best_score < 3:
                return StatementType.GUIDANCE

    # Boost from page text headings (strong signal)
    page_lower = page_text.lower()
    if "cash flow" in page_lower:
        scores[StatementType.CASH_FLOW] += 2
    if "income statement" in page_lower or "profit and loss" in page_lower or "statement of operations" in page_lower:
        scores[StatementType.INCOME_STATEMENT] += 2
    if "balance sheet" in page_lower or "financial position" in page_lower:
        scores[StatementType.BALANCE_SHEET] += 2

    # Pick highest-scoring main statement type
    best_type = max(scores, key=scores.get)
    if scores[best_type] >= 2:
        return best_type

    return StatementType.UNKNOWN


def parse_period_label(text: str) -> dict | None:
    """
    Parse a period label from a column header.

    Returns {"label": "Q4 2025", "type": "quarter"} or None.
    Handles many common formats used in financial filings.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    lower = text.lower()

    # "Q4 2025", "Q4 25", "Q1 2025"
    m = re.search(r"q([1-4])\s*['\-]?\s*(\d{2,4})", lower)
    if m:
        q = m.group(1)
        year = _extract_year(m.group(2)) or _extract_year(text)
        if year:
            return {"label": f"Q{q} {year}", "type": "quarter"}

    # "4Q25", "4Q 2025", "4Q2025"
    m = re.search(r"([1-4])q\s*['\-]?\s*(\d{2,4})", lower)
    if m:
        q = m.group(1)
        year = _extract_year(m.group(2)) or _extract_year(text)
        if year:
            return {"label": f"Q{q} {year}", "type": "quarter"}

    # "H1 2025", "H2 2025", "1H 2025", "1H25"
    m = re.search(r"h([12])\s*['\-]?\s*(\d{2,4})", lower)
    if not m:
        m = re.search(r"([12])h\s*['\-]?\s*(\d{2,4})", lower)
    if m:
        h = m.group(1)
        year = _extract_year(m.group(2)) or _extract_year(text)
        if year:
            return {"label": f"H{h} {year}", "type": "half"}

    # "FY2025", "FY 2025", "FY25"
    m = re.search(r"fy\s*['\-]?\s*(\d{2,4})", lower)
    if m:
        year = _extract_year(m.group(1)) or _extract_year(text)
        if year:
            return {"label": f"FY {year}", "type": "annual"}

    # "Year ended December 31, 2025" / "Year ended Dec 2025"
    if "year ended" in lower or "year ending" in lower:
        year = _extract_year(text)
        if year:
            return {"label": f"FY {year}", "type": "annual"}

    # "Three months ended Dec 31, 2025"
    m_three = re.search(r"three\s+months?\s+ended", lower)
    if m_three:
        month = _extract_month(text)
        year = _extract_year(text)
        if month and year:
            q = _month_to_quarter(month)
            return {"label": f"{q} {year}", "type": "quarter"}

    # "Six months ended Jun 30, 2025"
    m_six = re.search(r"six\s+months?\s+ended", lower)
    if m_six:
        month = _extract_month(text)
        year = _extract_year(text)
        if month and year:
            h = "H1" if month <= 6 else "H2"
            return {"label": f"{h} {year}", "type": "half"}

    # "As at Dec 2025" / "As at 31 December 2025" (balance sheet date)
    if "as at" in lower or "as of" in lower:
        month = _extract_month(text)
        year = _extract_year(text)
        if month and year:
            q = _month_to_quarter(month)
            return {"label": f"{q} {year}", "type": "point_in_time"}

    # "31 December 2025" / "December 31, 2025" (bare date)
    month = _extract_month(text)
    year = _extract_year(text)
    if month and year and re.search(r"\d{1,2}", text):
        # Has a day number — this is a date
        q = _month_to_quarter(month)
        return {"label": f"{q} {year}", "type": "point_in_time"}

    # Plain year: "2025" (standalone)
    m = re.match(r"^\s*(20\d{2})\s*$", text)
    if m:
        return {"label": f"FY {m.group(1)}", "type": "annual"}

    return None


def extract_periods_from_headers(table: list[list]) -> list[dict]:
    """
    Parse first row (or first two rows) as column headers.
    Returns list of {"column_index", "period", "period_type", "is_current"}.
    """
    if not table or len(table) < 1:
        return []

    periods = []
    header_row = table[0]

    # Try parsing each cell in the first row
    for i, cell in enumerate(header_row):
        parsed = parse_period_label(str(cell) if cell is not None else "")
        if parsed:
            periods.append({
                "column_index": i,
                "period": parsed["label"],
                "period_type": parsed["type"],
                "is_current": False,
            })

    # If no periods found in first row, try second row (merged headers)
    if not periods and len(table) >= 2:
        header_row = table[1]
        for i, cell in enumerate(header_row):
            parsed = parse_period_label(str(cell) if cell is not None else "")
            if parsed:
                periods.append({
                    "column_index": i,
                    "period": parsed["label"],
                    "period_type": parsed["type"],
                    "is_current": False,
                })

    # Also try combining row 0 + row 1 cells (e.g., row 0 = year, row 1 = quarter)
    if not periods and len(table) >= 2:
        for i in range(len(table[0])):
            cell0 = str(table[0][i]) if i < len(table[0]) and table[0][i] is not None else ""
            cell1 = str(table[1][i]) if i < len(table[1]) and table[1][i] is not None else ""
            combined = f"{cell0} {cell1}".strip()
            parsed = parse_period_label(combined)
            if parsed:
                periods.append({
                    "column_index": i,
                    "period": parsed["label"],
                    "period_type": parsed["type"],
                    "is_current": False,
                })

    # Mark most recent period as is_current
    if periods:
        # Sort by extracting year and quarter for comparison
        def _sort_key(p):
            label = p["period"]
            year = _extract_year(label) or 0
            # Extract quarter/half number for ordering
            m = re.search(r"Q(\d)", label)
            if m:
                return (year, int(m.group(1)))
            m = re.search(r"H(\d)", label)
            if m:
                return (year, int(m.group(1)) * 2)  # H1→2, H2→4
            if "FY" in label:
                return (year, 5)  # FY sorts after all quarters
            return (year, 0)

        best_idx = max(range(len(periods)), key=lambda i: _sort_key(periods[i]))
        periods[best_idx]["is_current"] = True

    return periods


def parse_financial_value(text: str) -> float | None:
    """
    Parse a financial value from a table cell.
    Handles: parenthetical negatives, commas, currency symbols, dashes, n/a.
    """
    if text is None:
        return None

    text = str(text).strip()

    # Null-like values
    if not text or text in ("—", "–", "-", "−", "n/a", "N/A", "nm", "NM", "n.m.", "N.M.", "—", "…", ".."):
        return None

    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[€$£¥₩kr\s]", "", text)

    # Handle parenthetical negatives: (1,234) → -1234
    m = re.match(r"^\(([0-9,.]+)\)$", cleaned)
    if m:
        cleaned = "-" + m.group(1)

    # Remove commas
    cleaned = cleaned.replace(",", "")

    # Handle minus signs (various unicode dashes)
    cleaned = cleaned.replace("−", "-").replace("–", "-")

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def detect_units(table: list[list], page_text: str) -> tuple[str, str]:
    """
    Detect currency and unit scale from table headers and surrounding page text.
    Returns (currency, scale) e.g. ("EUR", "millions").
    """
    flat = _flatten_table_text(table)
    context = (flat + " " + page_text.lower()).strip()

    # Detect currency
    currency = "USD"  # default
    if "€" in context or "eur" in context or "euro" in context:
        currency = "EUR"
    elif "£" in context or "gbp" in context or "pound" in context or "sterling" in context:
        currency = "GBP"
    elif "chf" in context or "swiss franc" in context:
        currency = "CHF"
    elif "¥" in context or "jpy" in context or "yen" in context:
        currency = "JPY"
    elif "₩" in context or "krw" in context or "won" in context:
        currency = "KRW"
    elif "sek" in context or "swedish" in context or "kronor" in context:
        currency = "SEK"
    elif "nok" in context or "norwegian" in context:
        currency = "NOK"
    elif "dkk" in context or "danish" in context:
        currency = "DKK"

    # Detect scale
    scale = "millions"  # default
    if "billion" in context or "bn" in context or "bns" in context:
        scale = "billions"
    elif "thousand" in context or "'000" in context or "000s" in context:
        scale = "thousands"
    elif "unit" in context and "million" not in context:
        # Only override to units if "million" isn't also present
        if re.search(r"\bunit\b", context):
            scale = "units"
    # "million" is the default so no explicit check needed

    return currency, scale


def split_by_period(
    table: list[list],
    periods: list[dict],
    statement_type: StatementType,
) -> list[FinancialTable]:
    """
    Split a multi-period table into one FinancialTable per period.
    Extracts label + value rows for each period column.
    """
    if not periods or not table:
        return []

    # Determine which row is the data start (skip header rows)
    data_start = 1
    # If second row also parsed as periods, start from row 2
    if len(table) >= 2:
        second_row_periods = []
        for cell in table[1]:
            if cell is not None and parse_period_label(str(cell)):
                second_row_periods.append(True)
        if len(second_row_periods) >= len(periods) // 2:
            data_start = 2

    results = []
    for period_info in periods:
        col_idx = period_info["column_index"]
        rows = []

        for row in table[data_start:]:
            # First column is typically the label
            label = str(row[0]).strip() if row and row[0] is not None else ""
            if not label:
                continue

            # Get value from the period's column
            raw_text = ""
            value = None
            if col_idx < len(row) and row[col_idx] is not None:
                raw_text = str(row[col_idx]).strip()
                value = parse_financial_value(raw_text)

            rows.append({
                "label": label,
                "value": value,
                "raw_text": raw_text,
            })

        ft = FinancialTable(
            statement_type=statement_type,
            period=period_info["period"],
            period_type=period_info["period_type"],
            rows=rows,
            is_current=period_info.get("is_current", False),
            raw_table=table,
        )
        results.append(ft)

    return results


def segment_document(
    pages: list[dict],
    tables_by_page: dict,
    sector: str = None,
) -> FinancialDocumentStructure:
    """
    Main entry point. Takes parsed document pages and pdfplumber tables.
    Pass sector (e.g. "Financials") for bank/insurance-aware classification.

    Args:
        pages: list of {"page_num": int, "text": str}
        tables_by_page: dict mapping page_num -> list of tables (list[list[list]])

    Returns:
        Complete FinancialDocumentStructure
    """
    structure = FinancialDocumentStructure()
    all_page_text = " ".join(p.get("text", "") for p in pages)

    # Detect document-level metadata
    currency, scale = detect_units([[]], all_page_text)
    structure.document_metadata = {
        "default_currency": currency,
        "default_scale": scale,
        "page_count": len(pages),
    }

    pages_with_tables = set()

    for page in pages:
        page_num = page.get("page_num", 0)
        page_text = page.get("text", "")
        page_tables = tables_by_page.get(page_num, [])

        if page_tables:
            pages_with_tables.add(page_num)

        for raw_table in page_tables:
            if not raw_table or len(raw_table) < 2:
                continue

            # Classify the table
            stmt_type = classify_table(raw_table, page_text, page_num, sector=sector)

            # Extract periods from headers
            periods = extract_periods_from_headers(raw_table)

            # Detect units for this specific table
            tbl_currency, tbl_scale = detect_units(raw_table, page_text)

            if periods:
                # Split by period
                financial_tables = split_by_period(raw_table, periods, stmt_type)
                for ft in financial_tables:
                    ft.currency = tbl_currency
                    ft.unit_scale = tbl_scale
                    ft.source_page = page_num
                structure.tables.extend(financial_tables)
            else:
                # No periods detected — store as a single table
                ft = FinancialTable(
                    statement_type=stmt_type,
                    period="unknown",
                    period_type="unknown",
                    currency=tbl_currency,
                    unit_scale=tbl_scale,
                    source_page=page_num,
                    raw_table=raw_table,
                )
                # Extract rows anyway (use col 1 as value)
                for row in raw_table[1:]:
                    label = str(row[0]).strip() if row and row[0] is not None else ""
                    if not label:
                        continue
                    raw_text = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
                    ft.rows.append({
                        "label": label,
                        "value": parse_financial_value(raw_text),
                        "raw_text": raw_text,
                    })
                structure.tables.append(ft)

    # Identify narrative sections (pages without tables) and footnotes
    for page in pages:
        page_num = page.get("page_num", 0)
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        if page_num not in pages_with_tables:
            lower = page_text.lower()
            # Check if it's a footnote page
            is_footnote = bool(re.search(
                r"^note\s+\d|^notes\s+to|footnote", lower[:200]
            ))
            if is_footnote:
                structure.footnotes.append({
                    "page_num": page_num,
                    "text": page_text,
                })
            else:
                structure.narrative_sections.append({
                    "page_num": page_num,
                    "text": page_text,
                })

    logger.info(
        "Segmented document: %d tables, %d narrative sections, %d footnotes",
        len(structure.tables),
        len(structure.narrative_sections),
        len(structure.footnotes),
    )
    return structure
