"""
Metric Normalisation & Deduplication Service.

Three capabilities:
  1. Name normalisation — maps variant names to a controlled vocabulary
  2. Deduplication — keeps highest-confidence metric per (company, metric, segment, period)
  3. Table-to-schema — structured table extraction before LLM interpretation

Applied after extraction and validation, before metrics are persisted.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 1. Metric Name Normalisation — controlled vocabulary
# ─────────────────────────────────────────────────────────────────

# Maps raw extracted names to canonical names
# Keys are lowercase patterns, values are the canonical name
METRIC_NAME_MAP = {
    # Revenue
    "total revenue": "Revenue",
    "net revenue": "Revenue",
    "net sales": "Revenue",
    "total net sales": "Revenue",
    "total sales": "Revenue",
    "sales": "Revenue",
    "turnover": "Revenue",
    "net turnover": "Revenue",
    "group revenue": "Revenue",
    "consolidated revenue": "Revenue",
    "revenue": "Revenue",
    # Operating Profit
    "operating income": "Operating Profit",
    "operating profit": "Operating Profit",
    "operating earnings": "Operating Profit",
    "income from operations": "Operating Profit",
    "profit from operations": "Operating Profit",
    "ebit": "EBIT",
    # EBITDA
    "ebitda": "EBITDA",
    "adjusted ebitda": "Adjusted EBITDA",
    # Net Income
    "net income": "Net Income",
    "net profit": "Net Income",
    "net earnings": "Net Income",
    "profit after tax": "Net Income",
    "profit for the period": "Net Income",
    "profit for the year": "Net Income",
    "profit attributable to equity holders": "Net Income (Attributable)",
    "net income attributable": "Net Income (Attributable)",
    # EPS
    "earnings per share": "EPS",
    "eps": "EPS",
    "diluted earnings per share": "EPS (Diluted)",
    "eps diluted": "EPS (Diluted)",
    "eps_diluted": "EPS (Diluted)",
    "diluted eps": "EPS (Diluted)",
    "basic earnings per share": "EPS (Basic)",
    "basic eps": "EPS (Basic)",
    # Margins
    "operating margin": "Operating Margin",
    "gross margin": "Gross Margin",
    "gross profit margin": "Gross Margin",
    "net margin": "Net Margin",
    "net profit margin": "Net Margin",
    "ebitda margin": "EBITDA Margin",
    "ebit margin": "EBIT Margin",
    # Cash Flow
    "operating cash flow": "Operating Cash Flow",
    "cash flow from operations": "Operating Cash Flow",
    "cash from operations": "Operating Cash Flow",
    "free cash flow": "Free Cash Flow",
    "fcf": "Free Cash Flow",
    "capital expenditure": "Capex",
    "capex": "Capex",
    "capital expenditures": "Capex",
    # Balance Sheet
    "total assets": "Total Assets",
    "total debt": "Total Debt",
    "net debt": "Net Debt",
    "total equity": "Total Equity",
    "shareholders equity": "Shareholders' Equity",
    "shareholder equity": "Shareholders' Equity",
    "book value per share": "Book Value Per Share",
    "tangible book value per share": "Tangible Book Value Per Share",
    "cash and equivalents": "Cash & Equivalents",
    "cash and cash equivalents": "Cash & Equivalents",
    # Dividends
    "dividend per share": "DPS",
    "dps": "DPS",
    "dividends per share": "DPS",
    "total dividend": "Total Dividends",
    "payout ratio": "Payout Ratio",
    # Returns
    "return on equity": "ROE",
    "roe": "ROE",
    "return on invested capital": "ROIC",
    "roic": "ROIC",
    "return on assets": "ROA",
    "roa": "ROA",
    # Growth
    "revenue growth": "Revenue Growth",
    "organic growth": "Organic Revenue Growth",
    "organic revenue growth": "Organic Revenue Growth",
    "like for like growth": "Organic Revenue Growth",
    "volume growth": "Volume Growth",
    # Leverage
    "net debt to ebitda": "Net Debt/EBITDA",
    "net debt/ebitda": "Net Debt/EBITDA",
    "leverage ratio": "Net Debt/EBITDA",
    "debt to equity": "Debt/Equity",
    # Employees
    "employees": "Headcount",
    "headcount": "Headcount",
    "number of employees": "Headcount",
    "fte": "Headcount (FTE)",
}

# Patterns for partial matching (if exact match fails)
METRIC_PATTERN_MAP = [
    (r"revenue.*organic", "Organic Revenue Growth"),
    (r"organic.*revenue", "Organic Revenue Growth"),
    (r"organic.*growth", "Organic Revenue Growth"),
    (r"like.for.like", "Organic Revenue Growth"),
    (r"operating.*margin", "Operating Margin"),
    (r"gross.*margin", "Gross Margin"),
    (r"ebitda.*margin", "EBITDA Margin"),
    (r"net.*margin", "Net Margin"),
    (r"operating.*profit", "Operating Profit"),
    (r"operating.*income", "Operating Profit"),
    (r"net.*income", "Net Income"),
    (r"net.*profit", "Net Income"),
    (r"earnings.*per.*share", "EPS"),
    (r"diluted.*eps", "EPS (Diluted)"),
    (r"free.*cash.*flow", "Free Cash Flow"),
    (r"operating.*cash.*flow", "Operating Cash Flow"),
    (r"net.*debt", "Net Debt"),
    (r"dividend.*per.*share", "DPS"),
    (r"return.*equity", "ROE"),
    (r"return.*invested", "ROIC"),
    (r"capital.*expend", "Capex"),
    (r"book.*value.*per", "Book Value Per Share"),
]


def normalise_metric_name(raw_name: str) -> str:
    """
    Normalise a metric name to the controlled vocabulary.
    Returns the canonical name, or the original if no match found.
    """
    if not raw_name:
        return raw_name

    # Strip period prefixes like "[Q4 2024] "
    clean = re.sub(r'^\[.*?\]\s*', '', raw_name).strip()
    lower = clean.lower().strip()

    # Exact match
    if lower in METRIC_NAME_MAP:
        # Preserve any period prefix
        prefix = raw_name[:len(raw_name) - len(clean)] if raw_name != clean else ""
        return prefix + METRIC_NAME_MAP[lower]

    # Pattern match
    for pattern, canonical in METRIC_PATTERN_MAP:
        if re.search(pattern, lower):
            prefix = raw_name[:len(raw_name) - len(clean)] if raw_name != clean else ""
            return prefix + canonical

    # No match — return original with cleaned formatting
    return raw_name


def normalise_unit(unit: str, metric_name: str = "") -> str:
    """Normalise unit strings to consistent format."""
    if not unit:
        return unit

    u = unit.strip().upper()
    name_lower = metric_name.lower()

    # Common mappings
    unit_map = {
        "USD_M": "USD_M", "USDM": "USD_M", "US$M": "USD_M", "$M": "USD_M", "USD M": "USD_M",
        "EUR_M": "EUR_M", "EURM": "EUR_M", "€M": "EUR_M", "EUR M": "EUR_M",
        "GBP_M": "GBP_M", "GBPM": "GBP_M", "£M": "GBP_M", "GBP M": "GBP_M",
        "USD_B": "USD_B", "US$B": "USD_B", "$B": "USD_B",
        "EUR_B": "EUR_B", "€B": "EUR_B",
        "GBP_B": "GBP_B", "£B": "GBP_B",
        "%": "%", "PERCENT": "%", "PCT": "%",
        "BPS": "bps", "BASIS POINTS": "bps",
        "X": "x", "TIMES": "x",
    }

    if u in unit_map:
        return unit_map[u]

    return unit


def normalise_metrics_batch(items: list[dict]) -> list[dict]:
    """
    Normalise metric names and units across a batch of extracted items.
    """
    for item in items:
        if not isinstance(item, dict):
            continue

        original_name = item.get("metric_name", "")
        normalised = normalise_metric_name(original_name)
        if normalised != original_name:
            item["original_metric_name"] = original_name
            item["metric_name"] = normalised

        original_unit = item.get("unit", "")
        normalised_unit = normalise_unit(original_unit, normalised)
        if normalised_unit != original_unit:
            item["unit"] = normalised_unit

    return items


# ─────────────────────────────────────────────────────────────────
# 2. Deduplication — keep highest confidence per key
# ─────────────────────────────────────────────────────────────────

def deduplicate_metrics(items: list[dict]) -> list[dict]:
    """
    Remove duplicate metrics, keeping the one with highest confidence.
    Key: (metric_name, period, segment)
    """
    if not items:
        return items

    unique = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("metric_name", "")
        period = item.get("period", "")
        segment = item.get("segment", "") or "Total"
        key = (name.lower(), period.lower(), segment.lower())

        confidence = item.get("confidence", 0)
        if key not in unique or confidence > unique[key].get("confidence", 0):
            unique[key] = item

    deduped = list(unique.values())
    removed = len(items) - len(deduped)
    if removed > 0:
        logger.info("Deduplication: removed %d duplicates, kept %d metrics", removed, len(deduped))

    return deduped


# ─────────────────────────────────────────────────────────────────
# 3. Segment Sum Validation
# ─────────────────────────────────────────────────────────────────

def validate_segment_sums(items: list[dict], tolerance_pct: float = 5.0) -> list[dict]:
    """
    Check if segment values sum to the total for the same metric and period.
    Flags items where segments don't reconcile.
    """
    from collections import defaultdict

    # Group by (metric, period)
    groups = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("metric_name", "").lower()
        period = item.get("period", "").lower()
        groups[(name, period)].append(item)

    for (name, period), group_items in groups.items():
        total_item = None
        segment_items = []
        for item in group_items:
            seg = (item.get("segment") or "").lower()
            if seg in ("total", "consolidated", "group", ""):
                total_item = item
            elif item.get("metric_value") is not None:
                segment_items.append(item)

        if total_item and segment_items and total_item.get("metric_value") is not None:
            try:
                total_val = float(total_item["metric_value"])
                seg_sum = sum(float(s["metric_value"]) for s in segment_items)
                if total_val != 0:
                    diff_pct = abs((seg_sum - total_val) / total_val) * 100
                    if diff_pct > tolerance_pct:
                        logger.warning(
                            "Segment sum mismatch for %s (%s): segments=%.1f, total=%.1f (diff=%.1f%%)",
                            name, period, seg_sum, total_val, diff_pct
                        )
                        # Penalise confidence on segment items
                        for s in segment_items:
                            s["confidence"] = max(0.3, s.get("confidence", 0.8) - 0.15)
                            s["segment_sum_warning"] = f"Segments sum to {seg_sum:.1f} vs total {total_val:.1f}"
            except (ValueError, TypeError):
                pass

    return items


# ─────────────────────────────────────────────────────────────────
# 4. Table-First Extraction Prompt
# ─────────────────────────────────────────────────────────────────

TABLE_TO_SCHEMA_PROMPT = """\
You are a financial data parser. Convert this structured table into normalised
financial datapoints.

COMPANY: {company} ({ticker})
DOCUMENT: {document_title}
PAGE: {page_number}

TABLE DATA:
{table_data}

RULES:
1. Create one entry per cell that contains a numeric financial value.
2. Identify the metric name from the row header.
3. Identify the period from the column header (e.g. Q4 2025, FY 2024).
4. Detect the currency from the table caption or surrounding context.
5. Set is_current_period=true for the most recent period, false for comparatives.
6. Normalise metric names: "Net Sales"→"Revenue", "Profit After Tax"→"Net Income",
   "Operating Income"→"Operating Profit", etc.
7. If a value has parentheses like (500), it's negative: -500.
8. Skip non-numeric rows (headers, section labels, blank rows).
9. Set confidence based on clarity: 1.0 if unambiguous, lower if the period or metric is unclear.

Respond ONLY with a JSON array. No preamble, no markdown fences.

Item schema:
{{
  "metric_name": "<normalised metric name>",
  "metric_value": <number>,
  "unit": "<USD_M | EUR_M | GBP_M | % | bps | x | null>",
  "period": "<e.g. Q4 2025 | FY 2024>",
  "is_current_period": true | false,
  "segment": "<segment name or Total>",
  "source_snippet": "<the row text>",
  "confidence": <0.0-1.0>
}}
"""


def format_table_for_llm(table: list[list], page: int = 0) -> str:
    """Convert a pdfplumber table (list of rows) to a formatted string for LLM."""
    if not table:
        return ""

    lines = []
    for i, row in enumerate(table):
        cells = [str(c).strip() if c else "" for c in row]
        lines.append(" | ".join(cells))

    return "\n".join(lines)


async def extract_from_tables(
    tables_data: list[dict],
    company_name: str,
    ticker: str,
    document_title: str = "",
) -> list[dict]:
    """
    Extract metrics from structured tables BEFORE the general text extraction.
    Returns a list of metric dicts in the standard schema.
    """
    from services.llm_client import call_llm_json_async

    all_items = []

    for table_info in tables_data:
        page = table_info.get("page", 0)
        tables = table_info.get("tables", [])

        for table in tables:
            if not table or len(table) < 2:
                continue

            # Skip tiny tables (likely not financial data)
            if len(table) < 3 or len(table[0]) < 2:
                continue

            # Check if table likely contains financial data
            header_text = " ".join(str(c) for c in table[0] if c).lower()
            all_text = " ".join(str(c) for row in table for c in row if c).lower()

            financial_keywords = [
                "revenue", "sales", "profit", "income", "ebitda", "ebit",
                "margin", "eps", "earnings", "cash flow", "assets", "debt",
                "dividend", "capex", "fy", "q1", "q2", "q3", "q4",
                "2024", "2025", "2026",
            ]
            if not any(kw in all_text for kw in financial_keywords):
                continue

            table_text = format_table_for_llm(table, page)
            prompt = TABLE_TO_SCHEMA_PROMPT.format(
                company=company_name, ticker=ticker,
                document_title=document_title, page_number=page,
                table_data=table_text,
            )

            try:
                items = await call_llm_json_async(prompt, max_tokens=4096)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item["page_number"] = page
                            item["extraction_method"] = "table_first"
                            all_items.append(item)
                logger.info("Table extraction (page %d): %d items", page, len(items) if isinstance(items, list) else 0)
            except Exception as e:
                logger.warning("Table extraction failed (page %d): %s", page, str(e)[:100])

    return all_items


# ─────────────────────────────────────────────────────────────────
# 5. Period Normalisation
# ─────────────────────────────────────────────────────────────────

def normalise_period(raw_period: str) -> str:
    """
    Normalise period format: "Q4 2025" -> "2025_Q4", "FY 2025" -> "2025_FY".
    Returns underscore-joined format suitable for period_label fields.
    """
    if not raw_period or not raw_period.strip():
        return raw_period or ""

    parts = raw_period.strip().split()
    if len(parts) == 2:
        if parts[0] in ("Q1", "Q2", "Q3", "Q4", "FY", "HY"):
            return f"{parts[1]}_{parts[0]}"
        return f"{parts[0]}_{parts[1]}"
    return raw_period.strip().replace(" ", "_")


# ─────────────────────────────────────────────────────────────────
# 6. Combined pipeline: normalise → validate segments → dedup
# ─────────────────────────────────────────────────────────────────

def post_process_metrics(items: list[dict]) -> list[dict]:
    """
    Full post-processing pipeline:
      1. Normalise metric names and units
      2. Validate segment sums
      3. Deduplicate
    """
    items = normalise_metrics_batch(items)
    items = validate_segment_sums(items)
    items = deduplicate_metrics(items)
    return items
