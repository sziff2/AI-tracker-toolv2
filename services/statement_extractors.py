"""
Targeted LLM extraction — one call per FinancialTable.

Runs parallel LLM calls with statement-type-specific prompts loaded
from prompts/extraction/{statement_type}.txt.
"""

import asyncio
import logging
from typing import Any

from services.financial_statement_segmenter import (
    FinancialDocumentStructure,
    FinancialTable,
    StatementType,
)
from services.llm_client import call_llm_json_async, call_llm_json_parallel, set_llm_context
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

# Map StatementType to prompt file name (under prompts/extraction/)
_PROMPT_MAP = {
    StatementType.INCOME_STATEMENT: "income_statement",
    StatementType.BALANCE_SHEET: "balance_sheet",
    StatementType.CASH_FLOW: "cash_flow",
    StatementType.SEGMENT_BREAKDOWN: "segment",
    StatementType.KPI_TABLE: "income_statement",      # fallback to income_statement
    StatementType.EQUITY_CHANGES: "balance_sheet",     # fallback to balance_sheet
    StatementType.GUIDANCE: "narrative",
    StatementType.NARRATIVE: "narrative",
    StatementType.FOOTNOTE: "notes",
    StatementType.UNKNOWN: "income_statement",         # best-effort
}


def _format_table_rows(table: FinancialTable) -> str:
    """Format rows as label: value text for the LLM prompt."""
    lines = []
    for row in table.rows:
        label = row.get("label", "")
        raw = row.get("raw_text", "")
        if label:
            lines.append(f"{label}: {raw}")
    return "\n".join(lines)


def _build_prompt_for_table(
    table: FinancialTable,
    company_name: str,
    ticker: str,
) -> str:
    """Build a focused extraction prompt for a single FinancialTable."""
    prompt_name = _PROMPT_MAP.get(table.statement_type, "income_statement")

    try:
        template = load_prompt(
            prompt_name,
            inputs={
                "company_name": company_name,
                "ticker": ticker,
                "period": table.period,
                "period_type": table.period_type,
                "currency": table.currency,
                "unit_scale": table.unit_scale,
                "segment": table.segment or "consolidated",
            },
            include_context_contract=False,
            include_output_constraints=True,
        )
    except FileNotFoundError:
        logger.warning("No prompt template for %s, using inline prompt", prompt_name)
        template = (
            f"Extract financial metrics from this {table.statement_type.value} "
            f"for {company_name} ({ticker}).\n"
            f"Period: {table.period} ({table.period_type})\n"
            f"Currency: {table.currency}, Scale: {table.unit_scale}\n"
            "Return JSON with metric names as keys and numeric values."
        )

    rows_text = _format_table_rows(table)

    prompt = (
        f"{template}\n\n"
        f"--- Data ---\n"
        f"Company: {company_name} ({ticker})\n"
        f"Statement: {table.statement_type.value}\n"
        f"Period: {table.period} ({table.period_type})\n"
        f"Currency: {table.currency}\n"
        f"Scale: {table.unit_scale}\n"
        f"Segment: {table.segment or 'consolidated'}\n\n"
        f"{rows_text}"
    )

    return prompt


async def extract_all_statements(
    structure: FinancialDocumentStructure,
    company_name: str,
    ticker: str,
) -> dict:
    """
    Run parallel LLM extraction for each table in the structure.

    Returns dict grouped by statement type:
    {
        "income_statements": [...],
        "balance_sheets": [...],
        "cash_flows": [...],
        "segments": [...],
        "notes": [...],
        "narratives": [...],
    }
    """
    set_llm_context(feature="statement_extraction", ticker=ticker)

    results: dict[str, list] = {
        "income_statements": [],
        "balance_sheets": [],
        "cash_flows": [],
        "segments": [],
        "notes": [],
        "narratives": [],
    }

    # ── Build prompts for all tables ──
    table_prompts = []
    table_metadata = []  # track which result bucket each prompt maps to

    for table in structure.tables:
        prompt = _build_prompt_for_table(table, company_name, ticker)
        table_prompts.append(prompt)

        # Map statement type to result key
        if table.statement_type == StatementType.INCOME_STATEMENT:
            key = "income_statements"
        elif table.statement_type == StatementType.BALANCE_SHEET:
            key = "balance_sheets"
        elif table.statement_type == StatementType.CASH_FLOW:
            key = "cash_flows"
        elif table.statement_type == StatementType.SEGMENT_BREAKDOWN:
            key = "segments"
        elif table.statement_type == StatementType.FOOTNOTE:
            key = "notes"
        else:
            key = "narratives"

        table_metadata.append({
            "key": key,
            "period": table.period,
            "period_type": table.period_type,
            "segment": table.segment,
            "currency": table.currency,
            "unit_scale": table.unit_scale,
            "is_current": table.is_current,
        })

    # ── Build prompts for narrative sections ──
    for section in structure.narrative_sections:
        page_text = section.get("text", "")
        if len(page_text.strip()) < 50:
            continue
        try:
            template = load_prompt(
                "narrative",
                inputs={
                    "company_name": company_name,
                    "ticker": ticker,
                },
                include_context_contract=False,
                include_output_constraints=True,
            )
        except FileNotFoundError:
            template = (
                f"Extract key insights and forward-looking statements from this narrative "
                f"section for {company_name} ({ticker}). Return JSON."
            )

        prompt = f"{template}\n\n--- Narrative Text (page {section.get('page_num', '?')}) ---\n{page_text[:3000]}"
        table_prompts.append(prompt)
        table_metadata.append({
            "key": "narratives",
            "period": "unknown",
            "period_type": "unknown",
            "segment": None,
            "currency": structure.document_metadata.get("default_currency", "USD"),
            "unit_scale": structure.document_metadata.get("default_scale", "millions"),
            "is_current": False,
        })

    # ── Build prompts for footnotes ──
    for note in structure.footnotes:
        page_text = note.get("text", "")
        if len(page_text.strip()) < 50:
            continue
        try:
            template = load_prompt(
                "notes",
                inputs={
                    "company_name": company_name,
                    "ticker": ticker,
                },
                include_context_contract=False,
                include_output_constraints=True,
            )
        except FileNotFoundError:
            template = (
                f"Extract key details from these financial statement notes "
                f"for {company_name} ({ticker}). Return JSON."
            )

        prompt = f"{template}\n\n--- Notes (page {note.get('page_num', '?')}) ---\n{page_text[:3000]}"
        table_prompts.append(prompt)
        table_metadata.append({
            "key": "notes",
            "period": "unknown",
            "period_type": "unknown",
            "segment": None,
            "currency": structure.document_metadata.get("default_currency", "USD"),
            "unit_scale": structure.document_metadata.get("default_scale", "millions"),
            "is_current": False,
        })

    if not table_prompts:
        logger.warning("No prompts to extract for %s (%s)", company_name, ticker)
        return results

    logger.info(
        "Running %d parallel LLM extractions for %s (%s)",
        len(table_prompts), company_name, ticker,
    )

    # ── Execute in parallel ──
    llm_results = await call_llm_json_parallel(
        table_prompts,
        max_concurrency=6,
        timeout_seconds=90,
    )

    # ── Collect results ──
    for i, llm_result in enumerate(llm_results):
        meta = table_metadata[i]
        key = meta["key"]

        if isinstance(llm_result, Exception):
            logger.warning("Extraction failed for prompt %d (%s): %s", i, key, llm_result)
            continue

        entry = {
            "data": llm_result,
            "period": meta["period"],
            "period_type": meta["period_type"],
            "segment": meta["segment"],
            "currency": meta["currency"],
            "unit_scale": meta["unit_scale"],
            "is_current": meta["is_current"],
        }
        results[key].append(entry)

    logger.info(
        "Extraction complete for %s: %s",
        ticker,
        {k: len(v) for k, v in results.items()},
    )

    return results
