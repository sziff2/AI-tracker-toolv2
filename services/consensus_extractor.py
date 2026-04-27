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

The document below is an analyst-curated consensus pack — it typically contains:
  • Broker / sell-side mean or median estimates
  • Per-metric consensus tables (Bloomberg / VARA / Visible Alpha export)
  • Comparison tables with Actual / Reported and Consensus / Estimate columns

Your job: extract the CONSENSUS column ONLY. Do NOT extract the actual /
reported figures — those already live elsewhere in the system.

Output a JSON object: {{"items": [...]}}. Each item has:
  metric_name      — canonical metric name (e.g. "Total Income", "NII", "EPS",
                     "Operating Profit", "ROE", "Cost/Income Ratio", "CET1 Ratio")
  consensus_value  — number ONLY (no currency or unit symbol)
  unit             — e.g. "SEK_M", "USD_M", "EUR_M", "GBP_M", "%", "x", "bps",
                     "SEK" for per-share metrics. Use null when ambiguous.
  source           — broker name when single-source ("Goldman Sachs",
                     "Morgan Stanley"). Use "consensus" when median/mean
                     across multiple analysts. Use the document's own
                     description otherwise.
  notes            — optional one-line context (e.g. "median of 14 brokers",
                     "ex-VAT refund")

CRITICAL rules:
  1. Consensus only — never copy actuals, prior-period figures, or YoY change.
  2. Period scope — only extract values that match {period_label}. If the
     document covers multiple periods, ignore non-matching ones.
  3. Numeric only — if a row has no numeric estimate (qualitative outlook),
     skip it.
  4. Don't invent — if you can't tell whether a number is consensus or
     actual, leave it out.
  5. Per-share figures stay in the per-share unit — if EPS consensus is
     "SEK 2.98", emit consensus_value=2.98, unit="SEK".

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
                    max_tokens=4096,
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
                    prompt, max_tokens=4096,
                    feature="consensus_extraction", ticker=ticker, tier="standard",
                )
        else:
            result = await call_llm_json_async(
                prompt,
                max_tokens=4096,
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
