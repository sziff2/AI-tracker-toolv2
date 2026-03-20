"""
Optimised Annual Report / 10-K extraction strategy.

ADD THIS FILE as: services/annual_report_extractor.py

Then in services/metric_extractor.py, replace the 10-K/annual_report
branch in extract_by_document_type() with a call to this module.
See INTEGRATION NOTE at the bottom.

Strategy:
  1. Triage chunks — fast regex filter to skip low-value sections
     (exhibits, signatures, legal boilerplate, footnote-only pages)
  2. Priority routing — financials/MD&A/risk factors get the full
     ANNUAL_REPORT_EXTRACTOR; other sections get the lighter
     COMBINED_EXTRACTOR
  3. Larger chunks (30k chars) — fewer LLM calls
  4. Skip cross-check validation (saves one full LLM pass)
  5. Cap total chunks — never run more than 12 LLM calls on one doc
"""

import logging
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric
from prompts import ANNUAL_REPORT_EXTRACTOR, COMBINED_EXTRACTOR
from services.llm_client import call_llm_json_parallel

logger = logging.getLogger(__name__)

# ── Section priority patterns ──────────────────────────────────────────
# Sections matching HIGH_VALUE get the full annual report prompt.
# Sections matching LOW_VALUE are skipped entirely.
HIGH_VALUE_PATTERNS = [
    r'(?i)(management.{0,10}discussion|MD&A|operating results)',
    r'(?i)(results of operations|financial condition)',
    r'(?i)(consolidated.{0,20}(income|balance|cash flow|statement))',
    r'(?i)(risk factors|principal risks)',
    r'(?i)(segment.{0,20}(results|revenue|profit|performance))',
    r'(?i)(capital (allocation|resources|expenditure))',
    r'(?i)(dividends|share (repurchase|buyback))',
    r'(?i)(outlook|guidance|forward.looking)',
    r'(?i)(revenue|turnover|EBITDA|operating profit|net income|earnings per share)',
    r'(?i)(return on (invested capital|equity|assets)|ROIC|ROE|ROA)',
    r'(?i)(free cash flow|net debt|leverage)',
]

LOW_VALUE_PATTERNS = [
    r'(?i)^exhibit\s+\d',
    r'(?i)signatures?\s*$',
    r'(?i)(incorporated herein by reference)',
    r'(?i)(pursuant to (rule|section|regulation)\s+\d)',
    r'(?i)(power of attorney)',
    r'(?i)(certification.{0,20}(sarbanes|sox|section 302|section 906))',
    r'(?i)(consent of independent)',
    r'(?i)(list of (subsidiaries|exhibits))',
    r'(?i)(index to financial statements)',
    r'(?i)^(item\s+\d+[a-z]?\s*\.?\s*)?(legal proceedings|mine safety)',
]

# 10-K/annual report gets bigger chunks — fewer calls
ANNUAL_CHUNK_SIZE = 30_000   # chars per chunk (vs 20k default)
MAX_CHUNKS = 12              # never run more than 12 LLM calls


def _section_priority(text: str) -> str:
    """Return 'high', 'low', or 'normal' based on content patterns."""
    for pat in LOW_VALUE_PATTERNS:
        if re.search(pat, text[:500]):  # check section header
            return 'low'
    for pat in HIGH_VALUE_PATTERNS:
        if re.search(pat, text):
            return 'high'
    # Check raw number density — high density = probably financial tables
    numbers = re.findall(r'\b\d[\d,\.]+\b', text)
    if len(numbers) / max(len(text) / 100, 1) > 2:
        return 'high'
    return 'normal'


def _chunk_annual_report(text: str) -> list[tuple[str, str]]:
    """
    Split text into (chunk, priority) tuples.
    Uses larger chunks and assigns priority to each.
    Returns list of (text, priority) — 'low' chunks are excluded.
    """
    if len(text) <= ANNUAL_CHUNK_SIZE:
        priority = _section_priority(text)
        return [(text, priority)] if priority != 'low' else []

    # Split by double newlines
    paragraphs = text.split('\n\n')
    chunks = []
    current_paras = []
    current_len = 0

    for para in paragraphs:
        # Skip obvious boilerplate paragraphs
        if len(para.strip()) < 50:
            continue

        if current_len + len(para) > ANNUAL_CHUNK_SIZE and current_paras:
            chunk_text = '\n\n'.join(current_paras)
            priority = _section_priority(chunk_text)
            if priority != 'low':
                chunks.append((chunk_text, priority))
            current_paras = []
            current_len = 0

        current_paras.append(para)
        current_len += len(para)

    if current_paras:
        chunk_text = '\n\n'.join(current_paras)
        priority = _section_priority(chunk_text)
        if priority != 'low':
            chunks.append((chunk_text, priority))

    return chunks


async def extract_annual_report(
    db: AsyncSession,
    document: Document,
    text: str,
    tables_data: list = None,
) -> dict:
    """
    Optimised extraction for 10-K / Annual Report documents.
    2-3x faster than the generic extractor on large filings.
    """
    # ── Step 1: Triage chunks ──────────────────────────────────
    chunks_with_priority = _chunk_annual_report(text)
    logger.info(
        "Annual report triage: %d total chunks from %d chars (doc: %s)",
        len(chunks_with_priority), len(text), document.id,
    )

    if not chunks_with_priority:
        return {"document_type": "annual_report", "items_extracted": 0, "raw_items": []}

    # ── Step 2: Priority routing ───────────────────────────────
    # High-value chunks → ANNUAL_REPORT_EXTRACTOR (full 5-category prompt)
    # Normal chunks → COMBINED_EXTRACTOR (lighter, faster)
    # Cap at MAX_CHUNKS total — prioritise 'high' over 'normal'
    high_chunks = [(c, p) for c, p in chunks_with_priority if p == 'high']
    normal_chunks = [(c, p) for c, p in chunks_with_priority if p == 'normal']

    # Fill up to MAX_CHUNKS: high first, then normal
    selected = high_chunks[:MAX_CHUNKS]
    remaining_slots = MAX_CHUNKS - len(selected)
    if remaining_slots > 0:
        selected += normal_chunks[:remaining_slots]

    logger.info(
        "Chunk selection: %d high-value + %d normal = %d total (capped at %d, skipped %d)",
        len([c for c in selected if c[1] == 'high']),
        len([c for c in selected if c[1] == 'normal']),
        len(selected), MAX_CHUNKS,
        len(chunks_with_priority) - len(selected),
    )

    # ── Step 3: Table-first extraction ────────────────────────
    table_items = []
    if tables_data:
        try:
            from services.metric_normaliser import extract_from_tables
            from sqlalchemy import select
            from apps.api.models import Company
            company_q = await db.execute(
                select(Company).where(Company.id == document.company_id)
            )
            company = company_q.scalar_one_or_none()
            table_items = await extract_from_tables(
                tables_data,
                company_name=company.name if company else "",
                ticker=company.ticker if company else "",
                document_title=document.title or "",
            )
            logger.info("Table-first: %d items from %d table groups", len(table_items), len(tables_data))
        except Exception as e:
            logger.warning("Table-first failed: %s", str(e)[:100])

    # ── Step 4: Build prompts and run in parallel ──────────────
    prompts = []
    for chunk_text, priority in selected:
        if priority == 'high':
            prompts.append(ANNUAL_REPORT_EXTRACTOR.format(text=chunk_text))
        else:
            prompts.append(COMBINED_EXTRACTOR.format(text=chunk_text))

    results = await call_llm_json_parallel(prompts, max_tokens=4096)  # 4k not 8k — faster

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    if table_items:
        all_items = table_items + all_items

    logger.info("Annual report extraction: %d raw items from %d prompts", len(all_items), len(prompts))

    # ── Step 5: Normalise + dedup (no cross-check — too slow for 10-Ks) ──
    try:
        from services.metric_normaliser import post_process_metrics
        before = len(all_items)
        all_items = post_process_metrics(all_items)
        logger.info("Post-processing: %d → %d items", before, len(all_items))
    except Exception as e:
        logger.warning("Post-processing failed: %s", str(e)[:100])

    # ── Step 6: Confidence filter only (skip LLM cross-check) ─
    try:
        from services.metric_validator import filter_by_confidence, validate_metrics_batch
        all_items = validate_metrics_batch(all_items)   # fast plausibility only
        all_items = filter_by_confidence(all_items, min_confidence=0.55)  # slightly lower threshold
        logger.info("Validation: %d items passed confidence filter", len(all_items))
    except Exception as e:
        logger.warning("Validation failed: %s", str(e)[:100])

    # ── Step 7: Persist ───────────────────────────────────────
    persisted = 0
    for item in all_items:
        if not isinstance(item, dict):
            continue
        # Handle both financial_metric and generic metric schemas
        metric_name = (
            item.get("metric_name") or
            item.get("topic") or
            item.get("item") or
            item.get("risk_name") or
            "unknown"
        )
        metric_value = item.get("metric_value")
        metric_text = (
            item.get("metric_text") or
            item.get("management_view") or
            item.get("disclosure") or
            item.get("description") or
            item.get("value") or
            ""
        )
        category = item.get("category", "")
        segment = category if category else item.get("segment")

        try:
            m = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=item.get("period") or document.period_label,
                metric_name=metric_name,
                metric_value=float(metric_value) if metric_value is not None else None,
                metric_text=str(metric_text)[:1000] if metric_text else None,
                unit=item.get("unit"),
                segment=str(segment)[:100] if segment else None,
                geography=item.get("geography"),
                source_snippet=str(item.get("source_snippet", ""))[:500],
                confidence=float(item.get("confidence", 0.7)),
                needs_review=float(item.get("confidence", 0.7)) < 0.7,
            )
            db.add(m)
            persisted += 1
        except Exception as e:
            logger.warning("Failed to persist item: %s", str(e)[:100])

    await db.commit()
    logger.info("Persisted %d metrics for annual report %s", persisted, document.id)

    return {
        "document_type": "annual_report",
        "items_extracted": len(all_items),
        "chunks_processed": len(selected),
        "chunks_skipped": len(chunks_with_priority) - len(selected),
        "raw_items": all_items,
    }


# ══════════════════════════════════════════════════════════════════
# INTEGRATION NOTE
# ══════════════════════════════════════════════════════════════════
#
# In services/metric_extractor.py, find extract_by_document_type()
# and add this import at the top of the function (or file):
#
#   from services.annual_report_extractor import extract_annual_report
#
# Then add this block BEFORE the generic extraction path:
#
#   # ── Annual report / 10-K: use optimised extractor ─────────
#   if doc_type in ("10-K", "annual_report"):
#       return await extract_annual_report(db, document, text, tables_data)
#
# That's it. The rest of the function handles all other doc types unchanged.
# ══════════════════════════════════════════════════════════════════
