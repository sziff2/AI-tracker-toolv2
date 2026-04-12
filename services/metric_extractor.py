"""
Metric Extraction Service — v2 with section splitting, model tiers, and sector context.

Architecture:
  1. Section splitter breaks filings into semantic sections
  2. Each section routed to specialised prompt + appropriate model tier
  3. Sector-specific KPI context injected into prompts
  4. Segment decomposition runs as dedicated pass
  5. Period validation catches mislabelling
  6. Post-processing: normalise + dedup + validate

Cost optimisation:
  - Financial statement tables → Haiku (fast tier)
  - MD&A / notes → Sonnet (default tier)
  - Synthesis / judgement → Sonnet or Opus (advanced tier)
"""

import asyncio
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
)
from prompts.section_prompts import SECTION_PROMPT_MAP
from schemas import ExtractedKPI, GuidanceItem
from services.llm_client import (
    call_llm_json, call_llm_json_async, call_llm_json_parallel,
    call_llm_native_async,
    TIER_FAST, TIER_DEFAULT, TIER_ADVANCED,
)
# Import JSON parser from llm_client for parsing native async responses
from services.llm_client import _parse_json as _llm_parse_json

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.8

# Document types that benefit from section splitting
SECTION_SPLIT_TYPES = {"earnings_release", "10-Q", "10-K", "annual_report"}

# Document types that use the original monolithic prompts
DOCTYPE_PROMPTS = {
    "earnings_release": EARNINGS_RELEASE_EXTRACTOR,
    "10-Q": EARNINGS_RELEASE_EXTRACTOR,
    "10-K": EARNINGS_RELEASE_EXTRACTOR,
    "annual_report": EARNINGS_RELEASE_EXTRACTOR,
    "transcript": TRANSCRIPT_EXTRACTOR,
    "broker_note": BROKER_NOTE_EXTRACTOR,
    "presentation": PRESENTATION_EXTRACTOR,
}

# ESG doc types run three extractors in parallel
ESG_DOC_TYPES = {"proxy_statement", "annual_report_esg", "sustainability_report"}


# ─────────────────────────────────────────────────────────────────
# Smart chunking — skip low-value content (kept for non-section types)
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
    numbers = re.findall(r'\d+\.?\d*', stripped)
    if len(numbers) < 2 and len(stripped) < 200:
        return True
    for pat in SKIP_PATTERNS:
        if re.search(pat, stripped):
            return True
    return False


def _smart_chunk(text: str, max_chars: int = 20000) -> list[str]:
    """Split text into chunks, skipping low-value content."""
    if len(text) <= max_chars:
        return [text] if not _is_low_value(text) else []

    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if _is_low_value(para) and len(para) > 500:
            continue

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
# v2: Section-aware extraction
# ─────────────────────────────────────────────────────────────────

async def _extract_with_sections(
    db: AsyncSession,
    document: Document,
    text: str,
    tables_data: list = None,
    sector: str = "",
    industry: str = "",
    country: str = "",
) -> dict:
    """
    Section-aware extraction for structured filings (10-K, 10-Q, earnings release).

    1. Split document into semantic sections
    2. Route each section to specialised prompt + model tier
    3. Run segment decomposition in parallel
    4. Validate periods
    5. Merge, normalise, dedup
    """
    from services.section_splitter import split_into_sections
    from services.sector_kpi_config import get_sector_context
    from services.period_validator import detect_reporting_period, validate_periods
    from services.segment_extractor import extract_segments, segments_to_metrics

    doc_type = document.document_type or "other"
    sector_context = get_sector_context(sector, industry, country)

    # ── Step 1: Split into sections ──────────────────────────
    sections = split_into_sections(text, doc_type=doc_type)
    logger.info(
        "Section-aware extraction: %d sections (types: %s)",
        len(sections),
        ", ".join(s.section_type for s in sections),
    )

    # Capture MD&A narrative for synthesis (raw text, not just extracted metrics)
    mda_narrative = ""
    for s in sections:
        if s.section_type in ("mda", "guidance", "preamble"):
            mda_narrative += s.text[:15000] + "\n\n"  # cap at 15K per section
    if mda_narrative:
        logger.info("Captured %d chars of MD&A narrative for synthesis", len(mda_narrative))

    # ── Step 2: Build extraction tasks per section ───────────
    extraction_tasks = []

    for section in sections:
        prompt_template = SECTION_PROMPT_MAP.get(section.section_type)

        if prompt_template is None:
            # Fall back to document-type prompt
            prompt_template = DOCTYPE_PROMPTS.get(doc_type, COMBINED_EXTRACTOR)

        # Inject sector context
        prompt = prompt_template.format(
            text=section.text[:section.max_tokens * 4],  # ~4 chars per token
            sector_context=sector_context,
        )

        extraction_tasks.append({
            "prompt": prompt,
            "tier": section.model_tier,
            "section_type": section.section_type,
            "max_tokens": section.max_tokens,
        })

    # ── Step 3: Run all section extractions in parallel ──────
    # Uses call_llm_native_async (true async + retry) instead of
    # call_llm_json_async (ThreadPoolExecutor, no retry) to avoid
    # silent failures on rate limits / transient errors.
    async def run_section_extraction(task):
        try:
            result = await call_llm_native_async(
                task["prompt"],
                max_tokens=task["max_tokens"],
                feature="section_extraction",
            )
            return _llm_parse_json(result["text"])
        except Exception as e:
            logger.warning(
                "Section extraction failed (%s): %s",
                task["section_type"], str(e)[:200],
            )
            return []

    # Also run segment decomposition in parallel
    segment_text = ""
    for s in sections:
        if s.section_type in ("financial_statements", "mda"):
            segment_text += "\n\n" + s.text
    if not segment_text:
        segment_text = text[:30000]

    # Also run table-first extraction if tables available
    table_task = None
    if tables_data and doc_type in SECTION_SPLIT_TYPES:
        table_task = _extract_from_tables(db, document, tables_data)

    # Two-pass structural extraction for financial statements
    # (splits into IS/BS/CF/segments, LLM only classifies labels, numbers from parser)
    two_pass_task = None
    if tables_data and doc_type in SECTION_SPLIT_TYPES:
        two_pass_task = _extract_two_pass(db, document, tables_data, sector)

    # Gather all tasks
    section_results = await asyncio.gather(
        *[run_section_extraction(t) for t in extraction_tasks],
        extract_segments(segment_text, sector_context),
        *([] if table_task is None else [table_task]),
        *([] if two_pass_task is None else [two_pass_task]),
        return_exceptions=True,
    )

    # ── Step 4: Collect all items ────────────────────────────
    all_items = []

    # Section extraction results
    num_section_tasks = len(extraction_tasks)
    for i, result in enumerate(section_results[:num_section_tasks]):
        if isinstance(result, Exception):
            logger.warning("Section %d failed: %s", i, str(result)[:200])
            continue
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    # Segment decomposition result
    segment_result = section_results[num_section_tasks]
    if isinstance(segment_result, dict) and not isinstance(segment_result, Exception):
        segment_items = segments_to_metrics(segment_result, document.period_label or "")
        all_items.extend(segment_items)
        logger.info("Segment decomposition added %d items", len(segment_items))

    # Table extraction result
    idx = num_section_tasks + 1
    if table_task is not None and len(section_results) > idx:
        table_result = section_results[idx]
        if isinstance(table_result, list) and not isinstance(table_result, Exception):
            all_items.extend(table_result)
            logger.info("Table-first extraction added %d items", len(table_result))
        idx += 1

    # Two-pass structural extraction result
    if two_pass_task is not None and len(section_results) > idx:
        tp_result = section_results[idx]
        if isinstance(tp_result, list) and not isinstance(tp_result, Exception):
            all_items.extend(tp_result)
            logger.info("Two-pass structural extraction added %d items", len(tp_result))
        elif isinstance(tp_result, Exception):
            logger.warning("Two-pass extraction failed: %s", str(tp_result)[:200])

    # ── Step 5: Period validation ────────────────────────────
    detected_period = detect_reporting_period(text, document.period_label or "")
    if detected_period:
        all_items = validate_periods(all_items, detected_period, document.period_label or "")
        logger.info("Period validation complete (detected: %s)", detected_period)

    # ── Step 6: Post-processing ──────────────────────────────
    try:
        from services.metric_normaliser import post_process_metrics
        before = len(all_items)
        all_items = post_process_metrics(all_items)
        logger.info("Post-processing: %d → %d items", before, len(all_items))
    except Exception as e:
        logger.warning("Post-processing failed: %s", str(e)[:100])

    # ── Step 7: Validation ───────────────────────────────────
    # Lower threshold for banks/insurance — their KPIs (NIM, credit costs)
    # often score lower confidence due to non-standard table layouts
    sector_lower = (sector or "").lower()
    is_financial = any(k in sector_lower for k in ["financ", "bank", "insur", "lending"])
    conf_threshold = 0.40 if is_financial else 0.60

    try:
        from services.metric_validator import validate_extraction
        validation = await validate_extraction(
            items=all_items,
            source_text=text,
            run_cross_check=True,
            confidence_threshold=conf_threshold,
        )
        all_items = validation["validated"]
        logger.info(
            "Validation: %d passed, %d rejected",
            validation["stats"]["passed"], validation["stats"]["rejected"],
        )
    except Exception as e:
        logger.warning("Validation failed, using unvalidated: %s", str(e)[:100])

    # ── Step 8: Qualifier language analysis ─────────────────
    confidence_profile = {}
    try:
        from services.qualifier_extractor import enrich_items_with_qualifiers, build_document_confidence_profile
        all_items = enrich_items_with_qualifiers(all_items, source_text=text)
        confidence_profile = build_document_confidence_profile(all_items)
        logger.info(
            "Qualifier analysis: signal=%s, hedge_rate=%.1f%%, one_off_rate=%.1f%%",
            confidence_profile.get("overall_signal", "?"),
            confidence_profile.get("hedge_rate", 0) * 100,
            confidence_profile.get("one_off_rate", 0) * 100,
        )
    except Exception as e:
        logger.warning("Qualifier analysis failed: %s", str(e)[:100])

    # ── Step 9: Disappeared metrics / guidance detection ─────
    disappearance_flags = {}
    try:
        from services.disappeared_detector import detect_disappeared
        disappearance_flags = await detect_disappeared(
            db, document.company_id, document.period_label or detected_period or "", all_items,
        )
        if disappearance_flags.get("total_flags", 0) > 0:
            logger.info(
                "Disappearance detector: %s",
                disappearance_flags.get("summary", ""),
            )
    except Exception as e:
        logger.warning("Disappearance detection failed: %s", str(e)[:100])

    # ── Step 10: Non-GAAP bridge extraction & comparison ─────
    bridge_data = {}
    bridge_comparison = {}
    try:
        from services.non_gaap_tracker import (
            extract_non_gaap_bridge, compare_bridges_across_periods, persist_bridge_data,
        )
        # Find financial statement text for bridge extraction
        fin_text = ""
        for s in sections:
            if s.section_type == "financial_statements":
                fin_text += "\n\n" + s.text
        if not fin_text:
            fin_text = text[:30000]

        bridge_data = await extract_non_gaap_bridge(fin_text)
        bridges = bridge_data.get("bridges", [])

        if bridges:
            # Store for future comparison
            await persist_bridge_data(
                db, document.company_id, document.id,
                document.period_label or "", bridges,
            )
            # Compare against prior period
            bridge_comparison = await compare_bridges_across_periods(
                db, document.company_id,
                document.period_label or detected_period or "", bridges,
            )
            if bridge_comparison.get("total_flags", 0) > 0:
                logger.info(
                    "Non-GAAP bridge: %d flags — gap %s",
                    bridge_comparison["total_flags"],
                    bridge_comparison.get("gap_trend", "stable"),
                )
    except Exception as e:
        logger.warning("Non-GAAP bridge analysis failed: %s", str(e)[:100])

    # ── Persist ──────────────────────────────────────────────
    if doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
        await _persist_earnings_metrics(db, document, all_items)
    elif doc_type == "transcript":
        await _persist_transcript_items(db, document, all_items)

    return {
        "document_type": doc_type,
        "extraction_method": "section_aware_v2",
        "sections_found": len(sections),
        "section_types": [s.section_type for s in sections],
        "items_extracted": len(all_items),
        "raw_items": all_items,
        "mda_narrative": mda_narrative[:20000] if mda_narrative else "",
        "segment_data": segment_result if isinstance(segment_result, dict) else None,
        "detected_period": detected_period,
        # Reflex pattern outputs
        "confidence_profile": confidence_profile,
        "disappearance_flags": disappearance_flags,
        "non_gaap_bridge": bridge_data.get("bridges", []),
        "non_gaap_comparison": bridge_comparison,
    }


async def _extract_two_pass(db, document, tables_data, sector) -> list[dict]:
    """Two-pass structural extraction of financial statement tables.

    Pass 1: LLM classifies row labels only (no numbers — can't hallucinate).
    Pass 2: Match labels to parser's pre-extracted numbers (no LLM).

    Bypasses the section_splitter text-based approach for financial
    statements, which hits max_tokens on dense tables. Each table gets
    its own small LLM call (2048 tokens for label classification) so
    truncation is near-impossible.
    """
    try:
        from services.financial_statement_segmenter import segment_document, StatementType
        from services.two_pass_extractor import two_pass_extract
        from apps.api.models import Company
        import sqlalchemy

        company_q = await db.execute(
            sqlalchemy.select(Company).where(Company.id == document.company_id)
        )
        company = company_q.scalar_one_or_none()
        if not company:
            return []

        # Convert tables_data [{page, tables}] into segment_document format
        pages = []
        tables_by_page = {}
        for page_data in tables_data:
            page_num = page_data.get("page", 0)
            page_tables = page_data.get("tables", [])
            pages.append({"page_num": page_num, "text": ""})
            if page_tables:
                tables_by_page[page_num] = page_tables

        structure = segment_document(pages, tables_by_page, sector=sector)
        logger.info("Two-pass: segmented %d financial tables from %d pages",
                    len(structure.tables), len(pages))

        if not structure.tables:
            return []

        # Run two_pass_extract on each table in parallel
        tasks = [
            two_pass_extract(table, company.name, company.ticker)
            for table in structure.tables
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect and convert to ExtractedMetric-compatible format
        all_items = []
        for table, result in zip(structure.tables, results):
            if isinstance(result, Exception):
                logger.warning("Two-pass failed for %s: %s",
                               table.statement_type.value, str(result)[:100])
                continue
            if not isinstance(result, list):
                continue
            for item in result:
                if item.get("value") is None:
                    continue
                all_items.append({
                    "metric_name": item.get("metric", "unknown"),
                    "metric_value": item.get("value"),
                    "metric_text": item.get("original_label", ""),
                    "unit": f"{table.currency}_{table.unit_scale[:1].upper()}"
                            if table.unit_scale else table.currency,
                    "period": table.period,
                    "segment": table.segment,
                    "line_item_type": table.statement_type.value,
                    "source_snippet": item.get("original_label", ""),
                    "confidence": 0.95,  # high — numbers from parser, labels from Haiku
                    "source": "two_pass",
                })
        return all_items
    except Exception as e:
        logger.warning("Two-pass extraction failed: %s", str(e)[:200])
        return []


async def _extract_from_tables(db, document, tables_data) -> list[dict]:
    """Extract metrics from structured tables (runs on fast tier)."""
    try:
        from services.metric_normaliser import extract_from_tables
        from apps.api.models import Company
        import sqlalchemy

        company_q = await db.execute(
            sqlalchemy.select(Company).where(Company.id == document.company_id)
        )
        company = company_q.scalar_one_or_none()
        items = await extract_from_tables(
            tables_data,
            company_name=company.name if company else "",
            ticker=company.ticker if company else "",
            document_title=document.title or "",
        )
        return items
    except Exception as e:
        logger.warning("Table-first extraction failed: %s", str(e)[:100])
        return []


# ─────────────────────────────────────────────────────────────────
# Legacy extraction (for transcripts, broker notes, presentations)
# ─────────────────────────────────────────────────────────────────

async def _extract_legacy(
    db: AsyncSession,
    document: Document,
    text: str,
    tables_data: list = None,
    sector: str = "",
    industry: str = "",
    country: str = "",
) -> dict:
    """
    Original extraction pipeline for doc types that don't benefit from
    section splitting (transcripts, broker notes, presentations).
    Now enhanced with sector context injection.
    """
    from services.sector_kpi_config import get_sector_context

    doc_type = document.document_type or "other"
    prompt_template = DOCTYPE_PROMPTS.get(doc_type, COMBINED_EXTRACTOR)
    sector_context = get_sector_context(sector, industry, country)

    # Inject sector context into the prompt if it has the placeholder
    if "{sector_context}" in prompt_template:
        pass  # Already has placeholder
    else:
        # Add sector context before the document text
        if sector_context:
            prompt_template = prompt_template.replace(
                "--- DOCUMENT TEXT ---",
                f"SECTOR CONTEXT:\n{sector_context}\n\n--- DOCUMENT TEXT ---",
            )

    chunks = _smart_chunk(text)
    logger.info("Legacy extraction: %d chunks from %d chars (type: %s)", len(chunks), len(text), doc_type)

    if not chunks:
        return {"document_type": doc_type, "items_extracted": 0, "raw_items": []}

    # Table-first extraction
    table_items = []
    if tables_data and doc_type in SECTION_SPLIT_TYPES:
        table_items = await _extract_from_tables(db, document, tables_data)

    # Build prompts and run in parallel
    prompts = [prompt_template.format(text=chunk) for chunk in chunks]
    results = await call_llm_json_parallel(prompts, max_tokens=8192)

    all_items = []
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
        elif isinstance(result, dict):
            all_items.append(result)

    if table_items:
        all_items = table_items + all_items

    # Post-processing
    try:
        from services.metric_normaliser import post_process_metrics
        before = len(all_items)
        all_items = post_process_metrics(all_items)
        logger.info("Post-processing: %d → %d items", before, len(all_items))
    except Exception as e:
        logger.warning("Post-processing failed: %s", str(e)[:100])

    # Validation
    try:
        from services.metric_validator import validate_extraction
        validation = await validate_extraction(
            items=all_items,
            source_text=text,
            run_cross_check=True,
            confidence_threshold=0.6,
        )
        all_items = validation["validated"]
    except Exception as e:
        logger.warning("Validation failed: %s", str(e)[:100])

    # Persist
    if doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
        await _persist_earnings_metrics(db, document, all_items)
    elif doc_type == "transcript":
        await _persist_transcript_items(db, document, all_items)

    # Qualifier analysis (works on all doc types including transcripts)
    confidence_profile = {}
    try:
        from services.qualifier_extractor import enrich_items_with_qualifiers, build_document_confidence_profile
        all_items = enrich_items_with_qualifiers(all_items, source_text=text)
        confidence_profile = build_document_confidence_profile(all_items)
    except Exception as e:
        logger.warning("Qualifier analysis failed: %s", str(e)[:100])

    # Disappearance detection (works on all doc types)
    disappearance_flags = {}
    try:
        from services.disappeared_detector import detect_disappeared
        disappearance_flags = await detect_disappeared(
            db, document.company_id, document.period_label or "", all_items,
        )
    except Exception as e:
        logger.warning("Disappearance detection failed: %s", str(e)[:100])

    return {
        "document_type": doc_type,
        "extraction_method": "legacy",
        "items_extracted": len(all_items),
        "raw_items": all_items,
        "confidence_profile": confidence_profile,
        "disappearance_flags": disappearance_flags,
    }


# ─────────────────────────────────────────────────────────────────
# Main entry point — routes to section-aware or legacy
# ─────────────────────────────────────────────────────────────────

async def extract_by_document_type(
    db: AsyncSession,
    document: Document,
    text: str,
    tables_data: list = None,
) -> dict:
    """
    Main extraction entry point. Routes to section-aware or legacy pipeline
    based on document type.

    Looks up company sector/industry/country from DB and injects context.
    """
    import sqlalchemy
    from apps.api.models import Company

    doc_type = document.document_type or "other"

    # Look up company for sector context
    sector, industry, country = "", "", ""
    try:
        company_q = await db.execute(
            sqlalchemy.select(Company).where(Company.id == document.company_id)
        )
        company = company_q.scalar_one_or_none()
        if company:
            sector = company.sector or ""
            industry = company.industry or ""
            country = company.country or ""
    except Exception:
        pass

    # Route to appropriate pipeline
    if doc_type in SECTION_SPLIT_TYPES:
        logger.info("Using section-aware extraction for %s (sector: %s)", doc_type, sector or "generic")
        return await _extract_with_sections(
            db, document, text, tables_data,
            sector=sector, industry=industry, country=country,
        )
    else:
        logger.info("Using legacy extraction for %s", doc_type)
        return await _extract_legacy(
            db, document, text, tables_data,
            sector=sector, industry=industry, country=country,
        )


# ─────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────

async def _persist_earnings_metrics(db, document, raw_items):
    from services.metric_normaliser import normalise_period
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            val = item.get("metric_value")
            if val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = None

            # Qualifier enrichment (from qualifier_extractor)
            qualifiers = item.get("_qualifiers")
            is_one_off = bool(item.get("_is_one_off", False))

            raw_period = item.get("period") or document.period_label or ""
            canonical_period = normalise_period(raw_period) or document.period_label

            metric = ExtractedMetric(
                id=uuid.uuid4(),
                company_id=document.company_id,
                document_id=document.id,
                period_label=canonical_period,
                metric_name=item.get("metric_name", ""),
                metric_value=val,
                metric_text=item.get("metric_text", ""),
                unit=item.get("unit"),
                segment=item.get("segment"),
                source_snippet=str(item.get("source_snippet", ""))[:500],
                confidence=item.get("confidence", 0.8),
                needs_review=item.get("confidence", 0.8) < REVIEW_THRESHOLD,
                is_one_off=is_one_off,
                qualifier_json=qualifiers if qualifiers else None,
            )
            db.add(metric)

            if metric.needs_review:
                review = ReviewQueueItem(
                    id=uuid.uuid4(),
                    entity_type="metric",
                    entity_id=metric.id,
                    queue_reason=f"Low confidence ({item.get('confidence', 0):.2f})",
                    priority="normal",
                )
                db.add(review)
        except Exception as e:
            logger.warning("Failed to persist metric: %s", str(e)[:100])
    await db.commit()


async def _persist_transcript_items(db, document, raw_items):
    for item in raw_items:
        try:
            cat = item.get("category", "")
            if cat == "guidance":
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


async def extract_combined(db: AsyncSession, document: Document, text: str) -> dict:
    """
    Single-pass extraction: KPIs and guidance together.
    Now routes through the main entry point for sector context.
    """
    return await extract_by_document_type(db, document, text)


async def extract_metrics(db: AsyncSession, document: Document, text: str) -> list[ExtractedMetric]:
    """Combined extraction used by single-doc pipeline."""
    result = await extract_combined(db, document, text)
    return result.get("raw_items", [])


async def extract_guidance(db: AsyncSession, document: Document, text: str) -> list[dict]:
    """Returns guidance already extracted by extract_combined."""
    return []


# ─────────────────────────────────────────────────────────────────
# ESG-specific extraction — runs E, S, G extractors in parallel
# ─────────────────────────────────────────────────────────────────

async def extract_esg(db: AsyncSession, document: Document, text: str) -> dict:
    """
    Run three ESG extractors in parallel (Environmental, Social, Governance).
    Returns raw items grouped by category + auto-populates ESG data table.
    """
    chunks = _smart_chunk(text, max_chars=12000)
    full_text = "\n\n".join(chunks[:3])

    env_prompt = ESG_ENVIRONMENTAL_EXTRACTOR.format(text=full_text)
    soc_prompt = ESG_SOCIAL_EXTRACTOR.format(text=full_text)
    gov_prompt = ESG_GOVERNANCE_EXTRACTOR.format(text=full_text)

    results = await asyncio.gather(
        call_llm_json_async(env_prompt, max_tokens=4096, tier=TIER_DEFAULT),
        call_llm_json_async(soc_prompt, max_tokens=4096, tier=TIER_DEFAULT),
        call_llm_json_async(gov_prompt, max_tokens=8192, tier=TIER_DEFAULT),
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

    await db.commit()

    return {
        "environmental": env_items,
        "social": soc_items,
        "governance": gov_items,
        "total_items": len(all_items),
        "persisted": len(metrics),
    }
