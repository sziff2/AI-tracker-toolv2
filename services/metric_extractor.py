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
)
from schemas import ExtractedKPI, GuidanceItem
from services.llm_client import call_llm_json, call_llm_json_async, call_llm_json_parallel

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.8

DOCTYPE_PROMPTS = {
    "earnings_release": EARNINGS_RELEASE_EXTRACTOR,
    "10-Q": EARNINGS_RELEASE_EXTRACTOR,
    "10-K": EARNINGS_RELEASE_EXTRACTOR,
    "annual_report": EARNINGS_RELEASE_EXTRACTOR,
    "transcript": TRANSCRIPT_EXTRACTOR,
    "broker_note": BROKER_NOTE_EXTRACTOR,
    "presentation": PRESENTATION_EXTRACTOR,
}

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
    db: AsyncSession, document: Document, text: str
) -> dict:
    """
    Run type-specific extraction with PARALLEL chunk processing.
    """
    doc_type = document.document_type or "other"
    prompt_template = DOCTYPE_PROMPTS.get(doc_type, COMBINED_EXTRACTOR)

    chunks = _smart_chunk(text)
    logger.info("Smart chunking: %d chunks from %d chars (type: %s)", len(chunks), len(text), doc_type)

    if not chunks:
        return {"document_type": doc_type, "items_extracted": 0, "raw_items": []}

    # Build all prompts and run in parallel
    prompts = [prompt_template.format(text=chunk) for chunk in chunks]
    results = await call_llm_json_parallel(prompts, max_tokens=8192)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    logger.info("Extracted %d items from %s document %s (parallel)", len(all_items), doc_type, document.id)

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

    # Persist metrics
    metrics = []
    for item in metrics_raw:
        try:
            confidence = item.get("confidence", 1.0)
            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=document.period_label,
                metric_name=item.get("metric_name", "unknown"),
                metric_value=item.get("metric_value"),
                metric_text=item.get("metric_text", ""),
                unit=item.get("unit"),
                segment=item.get("segment"),
                geography=item.get("geography"),
                source_snippet=item.get("source_snippet", ""),
                page_number=item.get("page_number"),
                confidence=confidence,
                needs_review=confidence < REVIEW_THRESHOLD,
            )
            db.add(metric)
            metrics.append(metric)
            if confidence < REVIEW_THRESHOLD:
                db.add(ReviewQueueItem(
                    id=uuid.uuid4(), entity_type="metric", entity_id=metric.id,
                    queue_reason=f"Low confidence ({confidence:.2f}) on {item.get('metric_name')}",
                    priority="high" if confidence < 0.5 else "normal",
                ))
        except Exception as e:
            logger.warning("Failed to persist metric: %s", str(e)[:100])

    # Persist guidance as metrics with guidance segment
    guidance_records = []
    for item in guidance_raw:
        try:
            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=document.period_label,
                metric_name=f"GUIDANCE: {item.get('metric_name', 'unknown')}",
                metric_value=item.get("high") or item.get("low"),
                metric_text=item.get("guidance_text", ""),
                unit=item.get("unit"),
                segment="guidance",
                source_snippet=item.get("source_snippet", ""),
                confidence=item.get("confidence", 0.8),
                needs_review=False,
            )
            db.add(metric)
            guidance_records.append(item)
        except Exception as e:
            logger.warning("Failed to persist guidance: %s", str(e)[:100])

    await db.commit()
    logger.info("Combined extraction: %d metrics, %d guidance from document %s",
                len(metrics), len(guidance_records), document.id)

    return {
        "metrics": metrics,
        "guidance": guidance_records,
        "total_items": len(all_items),
    }


# ─────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────

async def _persist_earnings_metrics(db, document, raw_items):
    for item in raw_items:
        try:
            confidence = item.get("confidence", 1.0)
            metric = ExtractedMetric(
                id=uuid.uuid4(), company_id=document.company_id, document_id=document.id,
                period_label=document.period_label, metric_name=item.get("metric_name", "unknown"),
                metric_value=item.get("metric_value"), metric_text=item.get("metric_text", ""),
                unit=item.get("unit"), segment=item.get("segment"), geography=item.get("geography"),
                source_snippet=item.get("source_snippet", ""), page_number=item.get("page_number"),
                confidence=confidence, needs_review=confidence < REVIEW_THRESHOLD,
            )
            db.add(metric)
        except Exception as e:
            logger.warning("Failed to persist metric: %s", str(e)[:100])
    await db.commit()


async def _persist_transcript_items(db, document, raw_items):
    for item in raw_items:
        try:
            cat = item.get("category", "")
            if cat == "guidance":
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
                db.add(metric)
        except Exception as e:
            logger.warning("Failed to persist transcript item: %s", str(e)[:100])
    await db.commit()


# ─────────────────────────────────────────────────────────────────
# Legacy functions (kept for single-doc pipeline compatibility)
# ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 15000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    lines = text.split("\n")
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


async def extract_metrics(db: AsyncSession, document: Document, text: str) -> list[ExtractedMetric]:
    """Combined extraction used by single-doc pipeline."""
    result = await extract_combined(db, document, text)
    return result["metrics"]


async def extract_guidance(db: AsyncSession, document: Document, text: str) -> list[dict]:
    """Returns guidance already extracted by extract_metrics/extract_combined."""
    # If called after extract_combined, guidance is already persisted
    # Just return empty — the combined extraction handles it
    return []
