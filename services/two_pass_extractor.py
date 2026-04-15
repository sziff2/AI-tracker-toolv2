"""
Two-pass extraction: structure first, numbers second.

Pass 1 (LLM - Haiku): "What are the row labels in this table?"
  -> Returns list of {label, category} without seeing numbers
Pass 2 (no LLM): Match labels to parser's pre-extracted numeric values

The LLM never sees the actual numbers, so it cannot hallucinate them.
"""

import json
import logging
from typing import Any

from services.financial_statement_segmenter import FinancialTable, StatementType
from services.llm_client import call_llm_json_async, set_llm_context
from configs.settings import settings
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)


# ── Sector detection ────────────────────────────────────────────

def _sector_key(sector: str | None, industry: str | None = None) -> str | None:
    """Map sector + industry to a prompt suffix, or None for generic.

    Industry takes precedence because it's more specific:
      sector="Financials", industry="Banks"         → "bank"
      sector="Financials", industry="P&C Insurance" → "insurance"
      sector="Financials", industry=""              → None (ambiguous — use generic)

    Sector-only matches are accepted for unambiguous cases ("Banks" as a
    sector label, "Insurance" as a sector label).
    """
    s = (sector or "").lower()
    i = (industry or "").lower()
    combined = f"{i} {s}"  # industry first so it dominates

    if any(k in combined for k in (
        "bank", "lending", "consumer finance", "credit services",
        "mortgage", "thrift", "capital markets",
    )):
        return "bank"
    if any(k in combined for k in ("insur", "reinsur", "underwrit")):
        return "insurance"
    # "Financials" / "Financial Services" alone is too broad — fall through
    # to generic rather than guess.
    return None


# ── Statement type → base prompt name ───────────────────────────

_PROMPT_BASE: dict[StatementType, str] = {
    StatementType.INCOME_STATEMENT:  "labels_income_statement",
    StatementType.BALANCE_SHEET:     "labels_balance_sheet",
    StatementType.CASH_FLOW:         "labels_cash_flow",
    StatementType.SEGMENT_BREAKDOWN: "labels_segment",
}

# ── Category mappings per statement type ────────────────────────

_CATEGORY_HINTS: dict[StatementType, list[str]] = {
    StatementType.INCOME_STATEMENT: [
        "revenue", "cost_of_sales", "gross_profit", "operating_expense",
        "operating_profit", "ebitda", "ebit", "interest", "tax",
        "net_income", "eps", "shares_outstanding", "other_income",
    ],
    StatementType.BALANCE_SHEET: [
        "cash", "receivables", "inventory", "current_assets",
        "ppe", "goodwill", "intangibles", "non_current_assets", "total_assets",
        "payables", "short_term_debt", "current_liabilities",
        "long_term_debt", "non_current_liabilities", "total_liabilities",
        "equity", "retained_earnings", "total_equity",
    ],
    StatementType.CASH_FLOW: [
        "net_income", "depreciation", "working_capital_change",
        "operating_cash_flow", "capex", "acquisitions",
        "investing_cash_flow", "debt_issued", "debt_repaid",
        "dividends_paid", "share_buyback", "financing_cash_flow",
        "free_cash_flow", "net_cash_change",
    ],
    StatementType.SEGMENT_BREAKDOWN: [
        "revenue", "operating_profit", "ebit", "ebitda", "assets",
    ],
}


def _build_label_prompt(
    labels: list[str],
    statement_type: StatementType,
    sector: str | None = None,
    industry: str | None = None,
) -> str:
    """Build a Haiku prompt for pass 1 (label classification only).

    Resolution order (first hit wins):
      1. prompts/extraction/labels_<statement>_<sector>.txt   (e.g. labels_income_statement_bank)
      2. prompts/extraction/labels_<statement>.txt            (generic)
      3. Inline fallback using _CATEGORY_HINTS
    """
    labels_text = "\n".join(f"- {label}" for label in labels)
    base = _PROMPT_BASE.get(statement_type)

    if base:
        sector_key = _sector_key(sector, industry)
        candidates = [f"{base}_{sector_key}", base] if sector_key else [base]
        for name in candidates:
            try:
                return load_prompt(
                    name,
                    inputs={"labels_block": labels_text},
                    include_context_contract=False,
                    include_output_constraints=False,
                )
            except FileNotFoundError:
                continue

    # Inline fallback — statement types with no file prompt (KPI, guidance, etc.)
    type_name = statement_type.value.replace("_", " ")
    categories = _CATEGORY_HINTS.get(statement_type, [])
    cat_hint = ", ".join(categories) if categories else "use your best judgement"
    return (
        f"Classify these financial statement row labels. Statement type: {type_name}.\n"
        f"Categories: {cat_hint}\n"
        f"Return ONLY a JSON array with original, normalised, and category for each.\n\n"
        f"Labels:\n{labels_text}\n\n"
        f'JSON:\n[{{"original": "...", "normalised": "...", "category": "..."}}]'
    )


async def extract_labels_only(
    table_rows: list[dict],
    statement_type: StatementType,
    company_name: str,
    ticker: str,
    sector: str | None = None,
    industry: str | None = None,
) -> list[dict]:
    """
    Pass 1: Send ONLY row labels (no numbers) to Haiku for classification.

    Returns list of:
        {"original_label": str, "normalised": str, "category": str}
    """
    labels = [row.get("label", "").strip() for row in table_rows if row.get("label", "").strip()]

    if not labels:
        return []

    prompt = _build_label_prompt(labels, statement_type, sector=sector, industry=industry)

    set_llm_context(feature="two_pass_labels", ticker=ticker)

    result = await call_llm_json_async(
        prompt,
        model=settings.agent_fast_model,
        max_tokens=2048,
        temperature=0.0,
    )

    # Validate and normalise the response
    if not isinstance(result, list):
        logger.warning("Label extraction returned non-list for %s: %s", ticker, type(result))
        return []

    classified = []
    for item in result:
        if not isinstance(item, dict):
            continue
        classified.append({
            "original_label": item.get("original", ""),
            "normalised": item.get("normalised", item.get("normalized", "")),
            "category": item.get("category", "unknown"),
        })

    logger.info(
        "Pass 1 classified %d/%d labels for %s (%s)",
        len(classified), len(labels), ticker, statement_type.value,
    )
    return classified


def match_labels_to_values(
    labels: list[dict],
    table: FinancialTable,
) -> list[dict]:
    """
    Pass 2 (no LLM): Match classified labels to parser's numeric values.

    Takes the LLM's label classification and matches against the
    parser's pre-extracted rows. Values come from the structural parser,
    never from the LLM.

    Returns list of:
        {"metric": str, "category": str, "value": float|None,
         "original_label": str, "source": "parser"}
    """
    # Build a lookup: lowercase label -> row dict
    row_lookup: dict[str, dict] = {}
    for row in table.rows:
        label = row.get("label", "").strip().lower()
        if label:
            row_lookup[label] = row

    matched = []
    for label_info in labels:
        original = label_info.get("original_label", "")
        normalised = label_info.get("normalised", "")
        category = label_info.get("category", "unknown")

        # Try exact match on original label first, then fuzzy
        row = row_lookup.get(original.lower())

        if row is None:
            # Try partial match: find the best matching row
            original_lower = original.lower()
            for row_label, row_data in row_lookup.items():
                if original_lower in row_label or row_label in original_lower:
                    row = row_data
                    break

        value = row.get("value") if row else None

        matched.append({
            "metric": normalised or original,
            "category": category,
            "value": value,
            "original_label": original,
            "source": "parser",
        })

    return matched


async def two_pass_extract(
    table: FinancialTable,
    company_name: str,
    ticker: str,
    sector: str | None = None,
    industry: str | None = None,
) -> list[dict]:
    """
    Main entry point for two-pass extraction.

    Pass 1: LLM classifies row labels (no numbers shown)
    Pass 2: Match labels to parser's numeric values

    Falls back to returning raw parser rows if label extraction fails.
    ``sector`` + ``industry`` select bank / insurance specialised label
    prompts. Industry is preferred (more specific) when both are given.

    Returns list of:
        {"metric": str, "category": str, "value": float|None,
         "original_label": str, "source": "parser"|"fallback"}
    """
    if not table.rows:
        return []

    try:
        # Pass 1: classify labels via LLM
        labels = await extract_labels_only(
            table.rows,
            table.statement_type,
            company_name,
            ticker,
            sector=sector,
            industry=industry,
        )

        if not labels:
            raise ValueError("No labels returned from pass 1")

        # Pass 2: match labels to parser values (no LLM)
        results = match_labels_to_values(labels, table)

        logger.info(
            "Two-pass extraction complete for %s: %d items from %s %s",
            ticker, len(results), table.statement_type.value, table.period,
        )
        return results

    except Exception as e:
        logger.warning(
            "Two-pass extraction failed for %s (%s %s), falling back to raw rows: %s",
            ticker, table.statement_type.value, table.period, e,
        )
        # Fallback: return raw parser rows without LLM classification
        fallback = []
        for row in table.rows:
            label = row.get("label", "").strip()
            if not label:
                continue
            fallback.append({
                "metric": label,
                "category": "unknown",
                "value": row.get("value"),
                "original_label": label,
                "source": "fallback",
            })
        return fallback
