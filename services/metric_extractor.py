"""
Metric Extraction Service — optimised for speed.

Optimisations:
  1. PARALLEL chunk processing (all chunks at once)
  2. COMBINED prompts (KPIs + guidance in one pass)
  3. SMART chunking (skip empty/boilerplate pages)
"""

import json
import logging
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric, ReviewQueueItem
from prompts import (
    KPI_EXTRACTOR, GUIDANCE_EXTRACTOR, COMBINED_EXTRACTOR,
    EARNINGS_RELEASE_EXTRACTOR, TRANSCRIPT_EXTRACTOR,
    BROKER_NOTE_EXTRACTOR, PRESENTATION_EXTRACTOR,
    ESG_ENVIRONMENTAL_EXTRACTOR, ESG_SOCIAL_EXTRACTOR, ESG_GOVERNANCE_EXTRACTOR,
    ANNUAL_REPORT_EXTRACTOR,
)
from schemas import ExtractedKPI, GuidanceItem
from services.llm_client import call_llm_json, call_llm_json_async, call_llm_json_parallel

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.8

DOCTYPE_PROMPTS = {
    "earnings_release": EARNINGS_RELEASE_EXTRACTOR,
    "10-Q": EARNINGS_RELEASE_EXTRACTOR,
    "10-K": ANNUAL_REPORT_EXTRACTOR,
    "annual_report": ANNUAL_REPORT_EXTRACTOR,
    "transcript": TRANSCRIPT_EXTRACTOR,
    "broker_note": BROKER_NOTE_EXTRACTOR,
    "presentation": PRESENTATION_EXTRACTOR,
}

# ESG doc types run three extractors in parallel
ESG_DOC_TYPES = {"proxy_statement", "annual_report_esg", "sustainability_report"}

# ─────────────────────────────────────────────────────────────────
# Smart chunking — skip low-value content
# ─────────────────────────────────────────────────────────────────

SKIP_PATTERNS = [
    r'(?i)forward.looking\s+statements?\s+disclaimer',
    r'(?i)safe\s+harbor',
    r'(?i)this\s+(page|slide)\s+(is\s+)?intentionally\s+left\s+blank',
    r'(?i)legal\s+disclaimer',
    r'(?i)important\s+notice',
    r'(?i)©\s*\d{4}',
]

def _is_low_value(text: str) -> bool:
    """Check if text is likely boilerplate with no financial content."""
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    # Check for numbers — financial content almost always has them
    numbers = re.findall(r'\d+\.?\d*', stripped)
    if len(numbers) < 2 and len(stripped) < 200:
        return True
    # Check for known boilerplate
    for pat in SKIP_PATTERNS:
        if re.search(pat, stripped):
            return True
    return False


def _smart_chunk(text: str, max_chars: int = 20000) -> list[str]:
    """
    Split text into chunks, skipping low-value content.
    Uses larger chunks (20k vs 15k) since we're doing combined extraction.
    """
    if len(text) <= max_chars:
        return [text] if not _is_low_value(text) else []

    # Split by double newlines (paragraph boundaries)
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if _is_low_value(para) and len(para) > 500:
            continue  # skip large boilerplate blocks

        if current_len + len(para) > max_chars and current:
            chunk_text = "\n\n".join(current)
            if not _is_low_value(chunk_text):
                chunks.append(chunk_text)
            current = []
            current_len = 0

        current.append(para)
        current_len += len(para)

    if current:
        chunk_text = "\n\n".join(current)
        if not _is_low_value(chunk_text):
            chunks.append(chunk_text)

    return chunks


# ─────────────────────────────────────────────────────────────────
# Type-specific extraction with parallel chunks
# ─────────────────────────────────────────────────────────────────

async def extract_by_document_type(
    db: AsyncSession, document: Document, text: str, tables_data: list = None
) -> dict:
    """
    Run type-specific extraction with PARALLEL chunk processing.
    If tables_data is provided, runs table-first extraction and merges results.
    """
    doc_type = document.document_type or "other"
    prompt_template = DOCTYPE_PROMPTS.get(doc_type, COMBINED_EXTRACTOR)

    # ── Pre-filter: remove boilerplate BEFORE chunking ────────
    # Removes exhibits, legal boilerplate, signatures, disclaimers
    # before any LLM token is spent. Saves 30-60% on large docs.
    FILTER_DOC_TYPES = {
        "10-K", "annual_report", "proxy_statement",
        "sustainability_report", "10-Q", "transcript", "broker_note",
    }
    if doc_type in FILTER_DOC_TYPES and len(text) > 15000:
        try:
            from services.document_filter import filter_document, get_high_value_text
            # Use stored parsed pages if available (set by document_parser)
            # Otherwise fall back to splitting raw text into 3000-char pseudo-pages
            if hasattr(document, '_parsed_pages') and document._parsed_pages:
                pages = document._parsed_pages
            else:
                page_size = 3000
                raw_chunks = [text[i:i+page_size] for i in range(0, len(text), page_size)]
                pages = [{"page": i + 1, "text": c} for i, c in enumerate(raw_chunks)]
            filter_result = filter_document(pages, doc_type)
            logger.info("Pre-filter [%s]: %s", doc_type, filter_result.summary())
            filtered_text = get_high_value_text(filter_result, max_chars=200_000)
            if filtered_text and len(filtered_text) > 500:
                text = filtered_text
        except Exception as e:
            logger.warning("Pre-filter failed, using raw text: %s", str(e)[:150])

    chunks = _smart_chunk(text)
    logger.info("Smart chunking: %d chunks from %d chars (type: %s)", len(chunks), len(text), doc_type)

    if not chunks:
        return {"document_type": doc_type, "items_extracted": 0, "raw_items": []}

    # ── Table-first extraction (if tables available) ─────────
    table_items = []
    if tables_data and doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
        try:
            from services.metric_normaliser import extract_from_tables
            from apps.api.models import Company
            company_q = await db.execute(
                __import__("sqlalchemy").select(Company).where(Company.id == document.company_id)
            )
            company = company_q.scalar_one_or_none()
            table_items = await extract_from_tables(
                tables_data,
                company_name=company.name if company else "",
                ticker=company.ticker if company else "",
                document_title=document.title or "",
            )
            logger.info("Table-first extraction: %d items from %d table groups", len(table_items), len(tables_data))
        except Exception as e:
            logger.warning("Table-first extraction failed: %s", str(e)[:100])

    # Build all prompts and run in parallel
    prompts = [prompt_template.format(text=chunk) for chunk in chunks]
    results = await call_llm_json_parallel(prompts, max_tokens=8192)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    # Merge table-first items with text extraction items
    if table_items:
        all_items = table_items + all_items

    logger.info("Extracted %d items from %s document %s (%d from tables, %d from text)",
                len(all_items), doc_type, document.id, len(table_items), len(all_items) - len(table_items))

    # ── Post-processing: normalise + dedup ────────────────────
    try:
        from services.metric_normaliser import post_process_metrics
        before = len(all_items)
        all_items = post_process_metrics(all_items)
        logger.info("Post-processing: %d → %d items (normalised + deduped)", before, len(all_items))
    except Exception as e:
        logger.warning("Post-processing failed: %s", str(e)[:100])

    # ── Validation pipeline ──────────────────────────────────
    try:
        from services.metric_validator import validate_extraction
        validation = await validate_extraction(
            items=all_items,
            source_text=text,
            run_cross_check=True,
            confidence_threshold=0.6,
        )
        all_items = validation["validated"]
        logger.info("Type-specific validation: %d passed, %d rejected",
                    validation["stats"]["passed"], validation["stats"]["rejected"])
    except Exception as e:
        logger.warning("Validation failed, using unvalidated: %s", str(e)[:100])

    # Persist based on type
    if doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
        await _persist_earnings_metrics(db, document, all_items)
    elif doc_type == "transcript":
        await _persist_transcript_items(db, document, all_items)

    return {
        "document_type": doc_type,
        "items_extracted": len(all_items),
        "raw_items": all_items,
    }


# ─────────────────────────────────────────────────────────────────
# Combined extraction (KPIs + guidance in one pass, parallel)
# ─────────────────────────────────────────────────────────────────

async def extract_combined(db: AsyncSession, document: Document, text: str) -> dict:
    """
    Single-pass extraction: KPIs and guidance together.
    Chunks processed in parallel. Returns split results.
    """
    chunks = _smart_chunk(text)
    logger.info("Combined extraction: %d chunks from %d chars", len(chunks), len(text))

    if not chunks:
        return {"metrics": [], "guidance": []}

    prompts = [COMBINED_EXTRACTOR.format(text=chunk) for chunk in chunks]
    results = await call_llm_json_parallel(prompts, max_tokens=8192)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    # Split into metrics and guidance
    metrics_raw = [i for i in all_items if i.get("type") == "metric" or "metric_value" in i]
    guidance_raw = [i for i in all_items if i.get("type") == "guidance" or "guidance_type" in i]

    # ── Post-processing: normalise + dedup ────────────────────
    try:
        from services.metric_normaliser import post_process_metrics
        before = len(metrics_raw)
        metrics_raw = post_process_metrics(metrics_raw)
        logger.info("Combined post-processing: %d → %d items", before, len(metrics_raw))
    except Exception as e:
        logger.warning("Post-processing failed: %s", str(e)[:100])

    # ── Validation pipeline ──────────────────────────────────
    try:
        from services.metric_validator import validate_extraction
        validation = await validate_extraction(
            items=metrics_raw,
            source_text=text,
            run_cross_check=True,
            confidence_threshold=0.6,
        )
        metrics_raw = validation["validated"]
        validation_stats = validation["stats"]
        logger.info("Validation: %d passed, %d rejected, %d flagged",
                    validation_stats["passed"], validation_stats["rejected"], validation_stats["flagged"])
    except Exception as e:
        logger.warning("Validation failed, using unvalidated metrics: %s", str(e)[:100])
        validation_stats = None

    # Persist metrics using shared helper
    metrics = await _persist_metrics(db, document, metrics_raw)

    # Persist guidance as metrics with guidance segment
    guidance_records = await _persist_metrics(db, document, guidance_raw, guidance_mode=True)

    logger.info("Combined extraction: %d metrics, %d guidance from document %s",
                len(metrics), len(guidance_records), document.id)

    return {
        "metrics": metrics,
        "guidance": guidance_raw,
        "total_items": len(all_items),
    }


# ─────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────

def _resolve_period_and_name(item: dict, document_period: str) -> tuple[str, str]:
    """Resolve the period label and metric name, normalising extracted period info."""
    from services.metric_normaliser import normalise_period

    item_period = item.get("period")
    metric_period = document_period
    metric_name = item.get("metric_name", "unknown")

    if item_period and item_period.strip():
        normalised = normalise_period(item_period)
        if item.get("is_current_period") is False:
            metric_period = normalised
            metric_name = f"[{item_period}] {metric_name}"

    return metric_period, metric_name


async def _persist_metrics(
    db, document, raw_items: list[dict], *, segment_filter: str | None = None, guidance_mode: bool = False
) -> list[ExtractedMetric]:
    """Shared persistence for all metric types."""
    metrics = []
    for item in raw_items:
        try:
            if segment_filter and item.get("category", "") != segment_filter:
                continue

            if guidance_mode:
                metric = ExtractedMetric(
                    id=uuid.uuid4(), company_id=document.company_id, document_id=document.id,
                    period_label=document.period_label,
                    metric_name=f"GUIDANCE: {item.get('metric_name', 'unknown')}",
                    metric_value=item.get("high") or item.get("low"),
                    metric_text=item.get("guidance_text", ""),
                    unit=item.get("unit"), segment="guidance",
                    source_snippet=item.get("source_snippet", ""),
                    confidence=item.get("confidence", 0.8), needs_review=False,
                )
            else:
                confidence = item.get("confidence", 1.0)
                metric_period, metric_name = _resolve_period_and_name(item, document.period_label)

                metric = ExtractedMetric(
                    id=uuid.uuid4(), company_id=document.company_id, document_id=document.id,
                    period_label=metric_period, metric_name=metric_name,
                    metric_value=item.get("metric_value"), metric_text=item.get("metric_text", ""),
                    unit=item.get("unit"), segment=item.get("segment"), geography=item.get("geography"),
                    source_snippet=item.get("source_snippet", ""), page_number=item.get("page_number"),
                    confidence=confidence, needs_review=confidence < REVIEW_THRESHOLD,
                )

            db.add(metric)
            metrics.append(metric)

            if not guidance_mode:
                confidence = item.get("confidence", 1.0)
                if confidence < REVIEW_THRESHOLD:
                    db.add(ReviewQueueItem(
                        id=uuid.uuid4(), entity_type="metric", entity_id=metric.id,
                        queue_reason=f"Low confidence ({confidence:.2f}) on {item.get('metric_name')}",
                        priority="high" if confidence < 0.5 else "normal",
                    ))
        except Exception as e:
            logger.warning("Failed to persist metric: %s", e)
    await db.commit()
    return metrics


async def _persist_earnings_metrics(db, document, raw_items):
    await _persist_metrics(db, document, raw_items)


async def _persist_transcript_items(db, document, raw_items):
    await _persist_metrics(db, document, raw_items, segment_filter="guidance", guidance_mode=True)


# ─────────────────────────────────────────────────────────────────
# Public API — thin wrappers around extract_combined
# ─────────────────────────────────────────────────────────────────

async def extract_metrics(db: AsyncSession, document: Document, text: str) -> list[ExtractedMetric]:
    """Combined extraction used by single-doc pipeline."""
    result = await extract_combined(db, document, text)
    return result["metrics"]


# ─────────────────────────────────────────────────────────────────
# ESG-specific extraction — runs E, S, G extractors in parallel
# ─────────────────────────────────────────────────────────────────

async def extract_esg(db: AsyncSession, document: Document, text: str) -> dict:
    """
    Run three ESG extractors in parallel (Environmental, Social, Governance).
    Returns raw items grouped by category + auto-populates ESG data table.
    """
    import asyncio

    chunks = _smart_chunk(text, max_chars=12000)
    full_text = "\n\n".join(chunks[:3])  # First 3 chunks for ESG (proxy docs are long)

    # Run all three in parallel
    env_prompt = ESG_ENVIRONMENTAL_EXTRACTOR.format(text=full_text)
    soc_prompt = ESG_SOCIAL_EXTRACTOR.format(text=full_text)
    gov_prompt = ESG_GOVERNANCE_EXTRACTOR.format(text=full_text)

    results = await asyncio.gather(
        call_llm_json_async(env_prompt, max_tokens=4096),
        call_llm_json_async(soc_prompt, max_tokens=4096),
        call_llm_json_async(gov_prompt, max_tokens=8192),  # Governance gets more tokens
        return_exceptions=True,
    )

    env_items = results[0] if isinstance(results[0], list) else []
    soc_items = results[1] if isinstance(results[1], list) else []
    gov_items = results[2] if isinstance(results[2], list) else []

    all_items = env_items + soc_items + gov_items

    # Persist as ExtractedMetric rows
    metrics = []
    for item in all_items:
        if not isinstance(item, dict):
            continue
        name = item.get("metric_name", "")
        if not name:
            continue
        cat = item.get("category", "esg")
        subcat = item.get("subcategory", "")
        m = ExtractedMetric(
            id=uuid.uuid4(),
            company_id=document.company_id,
            document_id=document.id,
            period_label=document.period_label,
            metric_name=f"ESG:{cat}:{name}",
            metric_value=item.get("metric_value"),
            metric_text=item.get("metric_text"),
            unit=item.get("unit"),
            segment=f"{cat}/{subcat}",
            source_snippet=item.get("source_snippet", "")[:500],
            confidence=item.get("confidence", 0.7),
        )
        db.add(m)
        metrics.append(m)

    # Auto-populate ESG data table
    esg_updates = {}
    for item in all_items:
        if not isinstance(item, dict):
            continue
        key = item.get("esg_field_key")
        val = item.get("metric_text") or str(item.get("metric_value", ""))
        if key and val:
            esg_updates[key] = val

    if esg_updates:
        try:
            from apps.api.models import ESGData
            import json as _json
            esg_q = await db.execute(
                __import__("sqlalchemy").select(ESGData).where(ESGData.company_id == document.company_id)
            )
            esg_row = esg_q.scalar_one_or_none()
            if esg_row:
                existing = _json.loads(esg_row.data) if isinstance(esg_row.data, str) else (esg_row.data or {})
                existing.update(esg_updates)
                esg_row.data = _json.dumps(existing)
            else:
                esg_row = ESGData(
                    id=uuid.uuid4(), company_id=document.company_id,
                    data=_json.dumps(esg_updates),
                )
                db.add(esg_row)
        except Exception as e:
            logger.warning("ESG auto-populate failed: %s", e)

    await db.commit()
    logger.info("ESG extraction: %d env, %d soc, %d gov items. Auto-populated %d fields.",
                len(env_items), len(soc_items), len(gov_items), len(esg_updates))

    return {
        "metrics": metrics,
        "raw_items": all_items,
        "environmental": env_items,
        "social": soc_items,
        "governance": gov_items,
        "esg_fields_populated": esg_updates,
    }
