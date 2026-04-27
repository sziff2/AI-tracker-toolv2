"""
Consensus extractor — turns an analyst-uploaded consensus document
(Bloomberg screen, VARA / Visible Alpha export, broker note table,
Excel paste) into structured consensus_expectations rows.

Wired into services/background_processor._process_one_document via
the dtype="consensus" branch. Reads parsed full_text, calls Claude
with a focused JSON-extraction prompt, upserts each row keyed on
(company_id, period_label, metric_name).

Per-row schema returned by the LLM:
    {metric_name, consensus_value, unit, source, notes}

Use cases the prompt handles:
    • Single-source consensus (one broker's estimate)
    • Median / mean consensus across multiple brokers
    • Tabular layouts with both Actual and Consensus columns —
      we keep ONLY the consensus column (the Actual already lives
      in extracted_metrics from the company's own filing)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from services.consensus_storage import upsert_consensus_row
from services.llm_client import call_llm_json_async, call_llm_native_async

logger = logging.getLogger(__name__)


# Conservative cap — consensus docs are typically a single page or two,
# so a 60K-char ceiling covers the largest realistic input without
# blowing the prompt budget on garbage trailing pages.
_MAX_TEXT_CHARS = 60_000


_PROMPT = """You are extracting CONSENSUS / STREET ESTIMATE values from a research document.

Company: {company_name} ({ticker})
Period:  {period_label}

The document is an analyst-curated consensus pack — typically Bloomberg /
VARA / Visible Alpha exports with one row per metric and many columns per
broker (Mean Consensus, Median Consensus, High, Low, then individual broker
estimates from Barclays, DB, Citi, JPM, etc.).

═══════════════════════════════════════════════════════════════
ABSOLUTE OUTPUT CONSTRAINTS — read carefully:

  1. ONE row per metric (per segment if applicable). NEVER emit one row
     per broker. The output should be ~15-40 items total, NOT 200+.

  2. Pick the BEST single consensus value per metric using this priority:
       a) "Median Consensus" column   → source="consensus (median)"
       b) "Mean Consensus" column     → source="consensus (mean)"
       c) Single broker if no median  → source="<broker name>"
     Ignore High / Low / individual broker columns when a median or mean
     is present.

  3. Skip "Actuals" / "Reported" columns — actuals live elsewhere.

  4. Skip per-period columns that don't match {period_label} (e.g. if
     the workbook has Q1/Q2/Q3 columns, take only the {period_label} one).

  5. Skip rows with no numeric estimate (qualitative outlook → skip).

  6. Do not invent. If unclear whether a number is consensus or actual,
     leave it out.
═══════════════════════════════════════════════════════════════

Output JSON: {{"items": [...]}}. Each item:
  metric_name      — canonical name (e.g. "Revenue", "EPS", "Operating Profit",
                     "Sales: Dupixent" for segment lines)
  segment          — segment / product / business-unit name OR null
  consensus_value  — number ONLY (no currency or unit symbol)
  unit             — "USD_M" / "EUR_M" / "GBP_M" / "SEK_M" / "%" / "x" /
                     "bps" / "EUR" (per-share), or null when ambiguous
  source           — "consensus (median)" | "consensus (mean)" | broker name
  notes            — optional ≤1-line context (e.g. "median of 14 brokers")

Respond ONLY with the JSON object. No preamble, no markdown fences.

DOCUMENT TEXT:
{document_text}
"""


async def extract_consensus(
    db: AsyncSession,
    document,
    full_text: str,
    *,
    company_name: str = "",
    ticker: str = "",
    uploaded_by: Optional[str] = None,
) -> dict:
    """Run the LLM extractor against parsed consensus-document text and
    upsert each item into consensus_expectations.

    Returns: {"extracted": int, "items": [...], "extraction_method": str}
    Never raises — failures land in the return dict so the caller (the
    background processor) keeps aggregating other docs."""
    period_label = (document.period_label or "").strip()
    if not period_label:
        return {
            "extracted": 0, "items": [], "extraction_method": "consensus",
            "error": "document has no period_label — cannot anchor consensus",
        }
    if not full_text or len(full_text) < 50:
        return {
            "extracted": 0, "items": [], "extraction_method": "consensus",
            "error": f"document text too short ({len(full_text or '')} chars)",
        }

    text = full_text[:_MAX_TEXT_CHARS]
    prompt = _PROMPT.format(
        company_name=company_name or "—",
        ticker=ticker or "—",
        period_label=period_label,
        document_text=text,
    )

    # Native-PDF path when the source is a PDF — Claude reads the layout
    # directly (merged cells, multi-column tables, headers) which is
    # materially better at consensus tables than the flattened pdfplumber
    # text. For .xlsx / .docx / text the parsed text already preserves
    # row/column alignment via openpyxl tabbed output, so the text path
    # is fine.
    pdf_path = None
    fp = (document.file_path or "").lower()
    if fp.endswith(".pdf"):
        pdf_path = document.file_path

    try:
        if pdf_path:
            from configs.settings import settings as _settings
            try:
                native = await call_llm_native_async(
                    prompt,
                    model=_settings.agent_default_model,
                    max_tokens=16384,
                    feature="consensus_extraction_pdf",
                    ticker=ticker,
                    pdf_path=pdf_path,
                )
                # Robust JSON parse from the native response
                from services.llm_client import _parse_json
                result = _parse_json(native["text"])
            except Exception as native_exc:
                logger.warning("Native-PDF consensus path failed for doc %s, falling back to text: %s",
                               document.id, str(native_exc)[:200])
                result = await call_llm_json_async(
                    prompt, max_tokens=16384,
                    feature="consensus_extraction", ticker=ticker, tier="standard",
                )
        else:
            result = await call_llm_json_async(
                prompt,
                max_tokens=16384,
                feature="consensus_extraction",
                ticker=ticker,
                tier="standard",
            )
    except Exception as exc:
        logger.warning("Consensus extraction LLM call failed for doc %s: %s",
                       document.id, str(exc)[:200])
        return {
            "extracted": 0, "items": [], "extraction_method": "consensus",
            "error": str(exc)[:200],
        }

    # Robust to either {"items": [...]} or a bare list.
    raw_items: list[dict] = []
    if isinstance(result, dict):
        raw_items = result.get("items") or []
    elif isinstance(result, list):
        raw_items = result
    if not isinstance(raw_items, list):
        raw_items = []

    persisted: list[dict] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        name = (it.get("metric_name") or "").strip()
        if not name:
            continue
        v_raw = it.get("consensus_value")
        try:
            value = float(v_raw) if v_raw is not None and v_raw != "" else None
        except (TypeError, ValueError):
            continue
        if value is None:
            continue
        try:
            row = await upsert_consensus_row(
                db,
                company_id=document.company_id,
                period_label=period_label,
                metric_name=name,
                consensus_value=value,
                unit=(it.get("unit") or None),
                source=(it.get("source") or None),
                notes=(it.get("notes") or None),
                uploaded_by=uploaded_by,
            )
            persisted.append(row)
        except Exception as exc:
            logger.warning("Consensus upsert failed for %s/%s: %s",
                           name, period_label, str(exc)[:200])

    if persisted:
        await db.commit()

    logger.info("Consensus extraction: doc=%s period=%s extracted=%d",
                document.id, period_label, len(persisted))
    return {
        "extracted":         len(persisted),
        "items":             persisted,
        "extraction_method": "consensus",
    }
