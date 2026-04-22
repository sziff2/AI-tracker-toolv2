"""
Background job processor — runs the full pipeline asynchronously
and updates job status/progress in the database.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from apps.api.models import (
    Company, Document, ProcessingJob, ResearchOutput, ThesisVersion,
    ExtractionProfile,
)
from configs.settings import settings

logger = logging.getLogger(__name__)

# Store running tasks so we can check if a job is already running
_running_jobs: dict[str, asyncio.Task] = {}


async def _update_job(job_id: uuid.UUID, **kwargs):
    """Update job fields in the database."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)
        await db.commit()


async def _update_step(job_id: uuid.UUID, step: str, pct: int, completed_steps: list[str]):
    """Update current step and progress."""
    await _update_job(
        job_id,
        current_step=step,
        progress_pct=pct,
        steps_completed=json.dumps(completed_steps),
    )


async def _add_log(job_id: uuid.UUID, message: str, level: str = "info"):
    """Append a timestamped log entry to the job."""
    from datetime import datetime
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
        existing = json.loads(job.log_entries) if job.log_entries else []
        existing.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
        })
        # Keep only last 100 entries to avoid bloating
        job.log_entries = json.dumps(existing[-100:])
        await db.commit()


async def _analyse_document_with_llm(db: AsyncSession, doc, dtype: str, full_text: str):
    """Run LLM analysis on transcripts and presentations during ingestion.
    Stores output in research_outputs so the Financial Analyst agent can
    consume it as pre-built context."""
    import json as _json
    from services.llm_client import call_llm_native_async
    from prompts.loader import load_prompt
    from services.context_builder import build_context_contract

    # Load the active macro context contract so transcript/presentation
    # analysis prompts can resolve their {rates}, {usd}, {credit}, {growth},
    # {commodities}, {geopolitical}, {inflation}, {liquidity} placeholders
    # (same block the agent pipeline uses). Without this the prompt ships
    # with literal "{rates}" etc. in the text and the LLM returns malformed
    # JSON in response.
    #
    # Use a FRESH session, not the caller's — _process_one_doc runs
    # extraction and document analysis in parallel via asyncio.gather, so
    # the shared session is already mid-operation when we reach this
    # point. Reusing it triggers SQLAlchemy "concurrent operations are
    # not permitted" errors. One short-lived session just for this
    # lookup sidesteps that completely.
    try:
        async with AsyncSessionLocal() as contract_db:
            context_contract = await build_context_contract(contract_db)
    except Exception as e:
        logger.warning("Failed to load context contract for %s analysis: %s", dtype, str(e)[:200])
        context_contract = {}

    # Build minimal inputs for the prompt.
    # Only include context_contract when it's non-empty — the loader uses
    # `"context_contract" in inputs` as the signal to render the contract
    # block, so an empty dict would trip the placeholder warnings anyway.
    inputs = {
        "company_name": "",
        "ticker": "",
        "period_label": doc.period_label or "",
        "thesis": "",
        "tracked_kpis": "",
        "prior_period": "",
    }
    if context_contract and context_contract.get("macro_assumptions"):
        inputs["context_contract"] = context_contract

    # Get company info
    try:
        from sqlalchemy import select as sa_select
        cq = await db.execute(sa_select(Company).where(Company.id == doc.company_id))
        company = cq.scalar_one_or_none()
        if company:
            inputs["company_name"] = company.name or ""
            inputs["ticker"] = company.ticker or ""
    except Exception:
        pass

    # Dispatch by document type. Annual reports get a larger window
    # (60k chars) because the strategic narrative, risk factor delta,
    # and footnote disclosures we want are often deep into the document.
    # Transcripts and presentations are tighter at 30k.
    if dtype == "transcript":
        inputs["transcript_text"] = full_text[:30000]
        agent_id = "transcript_deep_dive"
        output_type = "transcript_analysis"
        analysis_model = settings.agent_fast_model  # Haiku
    elif dtype == "presentation":
        inputs["presentation_text"] = full_text[:30000]
        agent_id = "presentation_analysis"
        output_type = "presentation_analysis"
        analysis_model = settings.agent_fast_model  # Haiku
    elif dtype == "annual_report":
        inputs["annual_report_text"] = full_text[:60000]
        agent_id = "annual_report_deep_read"
        output_type = "annual_report_analysis"
        # Sonnet for annual reports — the deltas and footnote interpretation
        # are higher-stakes and the document is longer / denser.
        analysis_model = settings.agent_default_model
    else:
        logger.warning("_analyse_document_with_llm called with unsupported dtype %r", dtype)
        return

    # Build and run prompt
    prompt_result = load_prompt(agent_id, inputs)
    prompt = prompt_result[0] if isinstance(prompt_result, tuple) else prompt_result

    # Tier 2.3 — route decks to native Claude PDF when the feature
    # flag is on and the file is a reasonably-sized PDF. The narrative
    # agent reads the full slide layout (charts, waterfall shapes,
    # capital deployment pies) rather than the flattened 30k-char text.
    pdf_path: str | None = None
    if (
        getattr(settings, "use_native_pdf_for_analysis", False)
        and dtype == "presentation"
        and doc.file_path
        and str(doc.file_path).lower().endswith(".pdf")
    ):
        try:
            import fitz as _fitz
            _f = _fitz.open(doc.file_path)
            _pages = len(_f)
            _f.close()
            _cap = getattr(settings, "native_pdf_analysis_max_pages", 50)
            if _pages <= _cap:
                pdf_path = doc.file_path
                logger.info(
                    "Routing presentation analysis of doc %s to native Claude PDF (%d pages)",
                    doc.id, _pages,
                )
            else:
                logger.info(
                    "Presentation doc %s has %d pages > %d cap — staying on text path",
                    doc.id, _pages, _cap,
                )
        except Exception as _exc:
            logger.warning(
                "Native-PDF page-count probe failed for doc %s: %s — falling back to text",
                doc.id, str(_exc)[:120],
            )

    result = await call_llm_native_async(
        prompt,
        model=analysis_model,
        max_tokens=4096,
        feature=f"doc_{dtype}_analysis",
        pdf_path=pdf_path,
    )

    # Parse JSON response
    raw = result["text"].strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw[3:]

    try:
        output = _json.loads(raw)
    except Exception:
        logger.warning("Failed to parse %s analysis JSON for doc %s", dtype, doc.id)
        output = {"raw_text": raw[:5000], "parse_failed": True}

    # Persist as research_output
    ro = ResearchOutput(
        id=uuid.uuid4(),
        company_id=doc.company_id,
        period_label=doc.period_label,
        output_type=output_type,
        content_json=_json.dumps(output, default=str),
    )
    db.add(ro)
    await db.commit()
    logger.info("Persisted %s analysis for doc %s (%s %s)",
                dtype, doc.id, inputs["ticker"], doc.period_label)


async def _persist_extraction_profile(db: AsyncSession, doc, extraction: dict):
    """Persist enriched extraction metadata to extraction_profiles table
    AND to research_outputs (as extraction_context) so build_agent_context()
    can find it."""
    if not isinstance(extraction, dict):
        return
    profile = ExtractionProfile(
        id=uuid.uuid4(),
        company_id=doc.company_id,
        document_id=doc.id,
        period_label=doc.period_label,
        extraction_method=extraction.get("extraction_method"),
        sections_found=extraction.get("sections_found"),
        section_types=extraction.get("section_types"),
        items_extracted=extraction.get("items_extracted"),
        confidence_profile=extraction.get("confidence_profile"),
        segment_data=extraction.get("segment_data"),
        disappearance_flags=extraction.get("disappearance_flags"),
        non_gaap_bridges=extraction.get("non_gaap_bridge"),
        non_gaap_comparison=extraction.get("non_gaap_comparison"),
        mda_narrative=(extraction.get("mda_narrative") or "")[:20000],
        detected_period=extraction.get("detected_period"),
        reconciliation=extraction.get("reconciliation"),
    )
    db.add(profile)

    # Also persist as ResearchOutput so build_extraction_context() can
    # serve it to Phase 1 agents via build_agent_context().
    import json as _json
    ctx_row = ResearchOutput(
        id=uuid.uuid4(),
        company_id=doc.company_id,
        period_label=doc.period_label,
        output_type="extraction_context",
        content_json=_json.dumps({
            "mda_narrative":       (extraction.get("mda_narrative") or "")[:20000],
            "confidence_profile":  extraction.get("confidence_profile"),
            "disappearance_flags": extraction.get("disappearance_flags"),
            "non_gaap_bridge":     extraction.get("non_gaap_bridge", []),
            "segment_data":        extraction.get("segment_data"),
            "detected_period":     extraction.get("detected_period", ""),
            "extraction_method":   extraction.get("extraction_method", "unknown"),
        }, default=str),
    )
    db.add(ctx_row)

    await db.commit()
    logger.info("Persisted extraction profile + context for doc %s (method=%s, items=%s)",
                doc.id, extraction.get("extraction_method"), extraction.get("items_extracted"))


async def _save_research_output(company_id: uuid.UUID, period_label: str, output: dict, output_type: str = "full_analysis"):
    """Persist a research output to the database."""
    try:
        async with AsyncSessionLocal() as db:
            ro = ResearchOutput(
                id=uuid.uuid4(),
                company_id=company_id,
                period_label=period_label,
                output_type=output_type,
                content_json=json.dumps(output, default=str),
                review_status="draft",
            )
            db.add(ro)
            await db.commit()
    except Exception as e:
        logger.error("Failed to save research output: %s", e)


async def run_single_pipeline(job_id: uuid.UUID, company_id: uuid.UUID, ticker: str, doc_id: uuid.UUID, period_label: str):
    """Run the full single-document pipeline in the background."""
    from services.llm_client import set_llm_context
    set_llm_context(feature="document_analysis", ticker=ticker, period=period_label)
    completed = []
    output = {}

    try:
        await _update_job(job_id, status="processing", current_step="parse", progress_pct=5)

        async with AsyncSessionLocal() as db:
            # Load document
            doc_q = await db.execute(select(Document).where(Document.id == doc_id))
            doc = doc_q.scalar_one_or_none()
            if not doc:
                await _update_job(job_id, status="failed", error_message="Document not found")
                return

            company_q = await db.execute(select(Company).where(Company.id == company_id))
            company = company_q.scalar_one_or_none()

            output["company"] = {"ticker": ticker, "name": company.name if company else ticker}
            output["period"] = period_label
            output["pipeline_status"] = []

            parse_ok = False

            # ── Step 1: Parse ────────────────────────────────
            await _update_step(job_id, "parse", 10, completed)
            try:
                from services.document_parser import process_document
                parse_result = await process_document(db, doc, ticker=ticker)
                output["classification"] = parse_result["classification"]
                output["pipeline_status"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"]})
                completed.append("parse")
                parse_ok = True
            except Exception as e:
                logger.error("[PIPELINE] %s parse failed for doc %s: %s", ticker, doc_id, e, exc_info=True)
                output["pipeline_status"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})

            # ── Step 2: Extract (skip if parse failed) ──────
            extract_ok = False
            if parse_ok:
                await _update_step(job_id, "extract", 25, completed)
                try:
                    # Try combined extraction first
                    try:
                        from services.metric_extractor import extract_combined
                        text_path = Path(settings.storage_base_path) / "processed" / ticker / period_label / "parsed_text.json"
                        pages = json.loads(text_path.read_text())
                        full_text = "\n\n".join(p["text"] for p in pages)

                        extraction = await extract_combined(db, doc, full_text)
                        metrics = extraction["metrics"]
                        guidance = extraction.get("guidance", [])
                    except Exception:
                        # Fall back to legacy extraction
                        from services.metric_extractor import extract_metrics
                        text_path = Path(settings.storage_base_path) / "processed" / ticker / period_label / "parsed_text.json"
                        pages = json.loads(text_path.read_text())
                        full_text = "\n\n".join(p["text"] for p in pages)
                        metrics = await extract_metrics(db, doc, full_text)
                        guidance = []

                    output["metrics"] = [
                        {
                            "metric_name": m.metric_name,
                            "metric_value": float(m.metric_value) if m.metric_value else None,
                            "metric_text": m.metric_text,
                            "unit": m.unit,
                            "segment": m.segment,
                            "geography": m.geography,
                            "source_snippet": m.source_snippet,
                            "confidence": float(m.confidence) if m.confidence else None,
                        }
                        for m in metrics
                    ]
                    output["guidance"] = guidance if isinstance(guidance, list) else []
                    output["pipeline_status"].append({"step": "extract", "status": "ok", "metrics": len(metrics), "guidance": len(output["guidance"])})
                    completed.append("extract")
                    extract_ok = True
                except Exception as e:
                    logger.error("[PIPELINE] %s extraction failed for job %s: %s", ticker, job_id, e, exc_info=True)
                    output["pipeline_status"].append({"step": "extract", "status": "error", "detail": str(e)[:500]})
                    output["extraction_error"] = str(e)[:500]
            else:
                logger.warning("[PIPELINE] %s skipping extraction — parse failed", ticker)
                output["pipeline_status"].append({"step": "extract", "status": "skipped", "detail": "parse failed"})

            # Synthesis, thesis comparison, surprise detection and IR
            # question generation have all been moved to the agent pipeline
            # (FA + Bear + Bull + Debate + QC). Document processing now
            # stops once parse + extract are complete — analysts trigger the
            # agent pipeline via "Analyse Period" in the cockpit.
            await _update_step(job_id, "saving", 90, completed)
            await _save_research_output(company_id, period_label, output, "full_analysis")

        # ── Done — determine final status ────────────────────
        failed_steps = [s["step"] for s in output.get("pipeline_status", []) if s.get("status") in ("error", "skipped")]
        if failed_steps and not completed:
            final_status = "failed"
            error_msg = f"All steps failed: {', '.join(failed_steps)}"
        elif failed_steps:
            final_status = "completed"
            error_msg = None
            logger.warning("[PIPELINE] Job %s completed with partial failures: %s (completed: %s)",
                           job_id, ', '.join(failed_steps), ', '.join(completed))
        else:
            final_status = "completed"
            error_msg = None

        await _update_job(
            job_id,
            status=final_status,
            current_step="done",
            progress_pct=100,
            steps_completed=json.dumps(completed),
            result_json=json.dumps(output, default=str),
            error_message=error_msg,
        )
        logger.info("[PIPELINE] Job %s %s: %s / %s | completed_steps=%s | failed_steps=%s",
                     job_id, final_status, ticker, period_label,
                     ', '.join(completed) or 'none', ', '.join(failed_steps) or 'none')

    except Exception as e:
        logger.error("[PIPELINE] Job %s crashed: %s", job_id, str(e), exc_info=True)
        await _update_job(
            job_id,
            status="failed",
            error_message=str(e)[:500],
            result_json=json.dumps(output, default=str) if output else None,
        )


MODEL_MAP = {
    "fast": "claude-haiku-4-5-20251001",
    "standard": "claude-sonnet-4-6",
    "deep": "claude-opus-4-20250514",
}


async def run_batch_pipeline(
    job_id: uuid.UUID, company_id: uuid.UUID, ticker: str,
    doc_ids: list[uuid.UUID], doc_types: list[str], period_label: str,
    model: str = "standard",
):
    """Run the full batch pipeline in the background."""
    # Resolve model ID from user selection
    model_id = MODEL_MAP.get(model, MODEL_MAP["standard"])
    completed = []
    output = {}

    # Set the LLM usage context so every downstream call (parsing
    # classification, section extraction, two-pass, table extraction,
    # statement extraction, transcript/presentation analysis, etc.)
    # gets tagged with the ticker and period in llm_usage_log. Without
    # this the cost dashboard can't attribute extraction spend to a
    # specific (company, period). All docs in one batch share the
    # same ticker+period so setting once at the top is sufficient.
    from services.llm_client import set_llm_context
    set_llm_context(feature="batch_pipeline", ticker=ticker, period=period_label)

    try:
        await _update_job(job_id, status="processing", current_step="processing_documents", progress_pct=5)
        await _add_log(job_id, f"Starting analysis for {ticker} / {period_label}")
        await _add_log(job_id, f"Processing {len(doc_ids)} documents: {', '.join(doc_types)}")

        async with AsyncSessionLocal() as db:
            company_q = await db.execute(select(Company).where(Company.id == company_id))
            company = company_q.scalar_one_or_none()

            # Load thesis
            thesis_q = await db.execute(
                select(ThesisVersion).where(
                    ThesisVersion.company_id == company_id, ThesisVersion.active == True
                ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
            )
            thesis = thesis_q.scalar_one_or_none()
            thesis_text = thesis.core_thesis if thesis else "No thesis on file."

            output["company"] = {"ticker": ticker, "name": company.name if company else ticker}
            output["period"] = period_label
            output["documents_processed"] = []
            output["per_document_extractions"] = {}

            earnings_data, transcript_data, broker_data, presentation_data, narrative_data = [], [], [], [], []
            last_doc_id = None
            total_docs = len(doc_ids)

            # ── Process documents in parallel ────────────────────────
            await _update_step(job_id, f"processing {total_docs} documents", 10, completed)

            async def _process_one_doc(did, dtype, idx):
                """Process a single document (parse + extract) with its own DB session.
                Skips parsing and extraction if the document already has sections and metrics."""
                from services.metric_extractor import extract_by_document_type, extract_esg, ESG_DOC_TYPES
                from services.document_parser import process_document
                from apps.api.models import DocumentSection, ExtractedMetric
                from sqlalchemy import func as sa_func

                doc_result = {"document_type": dtype, "steps": []}
                items = []
                esg_data = None

                try:
                    async with AsyncSessionLocal() as doc_db:
                        doc_q = await doc_db.execute(select(Document).where(Document.id == did))
                        doc = doc_q.scalar_one_or_none()
                        if not doc:
                            return None

                        doc_result["filename"] = doc.title

                        # Check if document already has sections (parsed) and metrics (extracted)
                        sections_count_q = await doc_db.execute(
                            select(sa_func.count(DocumentSection.id)).where(DocumentSection.document_id == did)
                        )
                        sections_count = sections_count_q.scalar() or 0

                        metrics_count_q = await doc_db.execute(
                            select(sa_func.count(ExtractedMetric.id)).where(ExtractedMetric.document_id == did)
                        )
                        metrics_count = metrics_count_q.scalar() or 0

                        if sections_count > 0 and metrics_count > 0:
                            # Already parsed and extracted — load existing metrics as items
                            logger.info("Doc %s already has %d sections and %d metrics — skipping parse/extract",
                                        did, sections_count, metrics_count)
                            doc_result["steps"].append({"step": "parse", "status": "skipped", "reason": "already parsed"})
                            doc_result["steps"].append({"step": "extract", "status": "skipped", "reason": "already extracted"})

                            metrics_q = await doc_db.execute(
                                select(ExtractedMetric).where(ExtractedMetric.document_id == did)
                            )
                            existing_metrics = metrics_q.scalars().all()
                            items = [
                                {
                                    "metric_name": m.metric_name,
                                    "metric_value": float(m.metric_value) if m.metric_value is not None else None,
                                    "metric_text": m.metric_text,
                                    "unit": m.unit,
                                    "segment": m.segment,
                                    "geography": m.geography,
                                    "source_snippet": m.source_snippet,
                                    "confidence": float(m.confidence) if m.confidence is not None else 1.0,
                                }
                                for m in existing_metrics
                            ]
                            return {"doc_id": did, "result": doc_result, "items": items, "dtype": dtype, "esg": esg_data, "title": doc.title}

                        # Parse - returns full_text directly now
                        try:
                            parse_result = await process_document(doc_db, doc, ticker=ticker)
                            doc_result["steps"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"]})
                            full_text = parse_result.get("full_text", "")
                            tables_data = parse_result.get("tables_data")
                            logger.info("Parsed doc %s: %d chars", did, len(full_text))

                            # Warn if text is suspiciously short
                            if len(full_text) < 500:
                                logger.warning("Document %s has very short text (%d chars) - may be parsing issue",
                                               did, len(full_text))
                        except Exception as e:
                            logger.error("[BATCH] Parse FAILED for doc %s (%s): %s", did, doc.title, str(e)[:200], exc_info=True)
                            doc_result["steps"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
                            return {"doc_id": did, "result": doc_result, "items": [], "dtype": dtype, "esg": None}

                        # Step 2: Extract metrics + run document analysis IN PARALLEL
                        # - Metric extraction (Haiku section-aware for 10-Qs)
                        # - Transcript/presentation LLM analysis (Haiku, runs concurrently)
                        async def _run_extraction():
                            try:
                                if dtype in ESG_DOC_TYPES:
                                    return ("esg", await extract_esg(doc_db, doc, full_text))
                                else:
                                    return ("metrics", await extract_by_document_type(
                                        doc_db, doc, full_text, tables_data=tables_data
                                    ))
                            except Exception as e:
                                logger.error("Extraction failed for doc %s (%s): %s", did, dtype, str(e)[:200])
                                return ("error", str(e)[:200])

                        async def _run_doc_analysis():
                            if dtype not in ("transcript", "presentation", "annual_report"):
                                return None
                            if not full_text or len(full_text) < 500:
                                return None
                            try:
                                await _analyse_document_with_llm(doc_db, doc, dtype, full_text)
                                return "ok"
                            except Exception as e:
                                logger.warning("Document analysis failed for %s (%s): %s", did, dtype, str(e)[:200])
                                return None

                        # Run both in parallel
                        ext_result, _doc_analysis_result = await asyncio.gather(
                            _run_extraction(),
                            _run_doc_analysis(),
                            return_exceptions=False,
                        )

                        # Process extraction result
                        items = []
                        extraction = None
                        esg_data = None
                        ext_kind, ext_value = ext_result
                        if ext_kind == "esg":
                            extraction = ext_value
                            items = extraction.get("raw_items", [])
                            doc_result["steps"].append({"step": "extract_esg", "status": "ok",
                                "env": len(extraction.get("environmental", [])),
                                "soc": len(extraction.get("social", [])),
                                "gov": len(extraction.get("governance", []))})
                            esg_data = {
                                "environmental": extraction.get("environmental", [])[:10],
                                "social": extraction.get("social", [])[:10],
                                "governance": extraction.get("governance", [])[:15],
                                "fields_populated": extraction.get("esg_fields_populated", {}),
                            }
                            logger.info("Extracted ESG from %s: %d items (dtype=%s)", did, len(items), dtype)
                        elif ext_kind == "metrics":
                            extraction = ext_value
                            items = extraction.get("raw_items", [])
                            doc_result["steps"].append({"step": "extract", "status": "ok", "items": len(items)})
                            logger.info("Extracted from %s: %d items (dtype=%s)", did, len(items), dtype)
                            try:
                                await _persist_extraction_profile(doc_db, doc, extraction)
                            except Exception as ep:
                                logger.warning("Failed to persist extraction profile for %s: %s", did, str(ep)[:200])
                        else:
                            doc_result["steps"].append({"step": "extract", "status": "error", "detail": ext_value})

                        if not items and ext_kind != "error":
                            logger.warning("No items extracted from %s (%s) - text was %d chars",
                                           doc.title, dtype, len(full_text))

                        mda = extraction.get("mda_narrative", "") if isinstance(extraction, dict) else ""

                        return {"doc_id": did, "result": doc_result, "items": items, "dtype": dtype, "esg": esg_data, "title": doc.title, "mda_narrative": mda}
                except Exception as e:
                    doc_result["steps"].append({"step": "error", "detail": str(e)[:200]})
                    return {"doc_id": did, "result": doc_result, "items": [], "dtype": dtype, "esg": None, "mda_narrative": ""}

            # Run documents with limited concurrency to avoid OOM on small Railway plans
            # 2 docs × 3 LLM calls = 6 max simultaneous requests — safe without crashing
            import asyncio
            semaphore = asyncio.Semaphore(2)

            async def _limited_process(did, dtype, idx):
                async with semaphore:
                    return await _process_one_doc(did, dtype, idx)

            tasks = [_limited_process(did, dtype, i) for i, (did, dtype) in enumerate(zip(doc_ids, doc_types))]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results with per-document error tracking
            doc_failures = []
            doc_successes = 0
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    dtype = doc_types[i] if i < len(doc_types) else "unknown"
                    logger.error("[BATCH] Document %d (%s) crashed: %s", i, dtype, res, exc_info=res)
                    doc_failures.append(f"doc_{i}({dtype})")
                    await _add_log(job_id, f"Document {i} ({dtype}) failed: {str(res)[:100]}", "error")
                    continue
                if res is None:
                    dtype = doc_types[i] if i < len(doc_types) else "unknown"
                    logger.warning("[BATCH] Result %d (%s) is None - document processing failed completely", i, dtype)
                    doc_failures.append(f"doc_{i}({dtype})")
                    await _add_log(job_id, f"Document {i} ({dtype}) returned no result", "warn")
                    continue
                doc_successes += 1
                last_doc_id = res["doc_id"]
                output["documents_processed"].append(res["result"])
                completed.append(f"doc_{i+1}")

                items = res["items"]
                dtype = res["dtype"]
                title = res.get("title", f"doc_{i}")

                logger.info("Aggregating doc %d: title=%s, dtype=%s, items=%d",
                            i, title[:50] if title else "?", dtype, len(items) if items else 0)
                await _add_log(job_id, f"✓ Processed: {title[:40] if title else 'doc_'+str(i)} ({dtype}) — {len(items) if items else 0} items extracted")

                if res.get("esg"):
                    output["esg_extraction"] = res["esg"]

                # Capture MD&A narrative for synthesis
                if res.get("mda_narrative"):
                    narrative_data.append(res["mda_narrative"][:10000])
                    logger.info("  -> Captured %d chars of MD&A narrative", len(res["mda_narrative"]))

                if items:
                    items_summary = json.dumps(items[:200], indent=2, default=str)
                    if dtype in ("earnings_release", "10-Q", "10-K", "annual_report"):
                        earnings_data.append(items_summary)
                        logger.info("  -> Added %d items to earnings_data", len(items))
                    elif dtype == "transcript":
                        transcript_data.append(items_summary)
                        logger.info("  -> Added %d items to transcript_data", len(items))
                    elif dtype == "broker_note":
                        broker_data.append(items_summary)
                        logger.info("  -> Added %d items to broker_data", len(items))
                    elif dtype == "presentation":
                        presentation_data.append(items_summary)
                        logger.info("  -> Added %d items to presentation_data", len(items))
                    elif dtype not in ("sustainability_report", "proxy_statement"):
                        earnings_data.append(items_summary)
                        logger.info("  -> Added %d items to earnings_data (fallback dtype=%s)", len(items), dtype)

                    # Capture qualitative/narrative items (drivers, guidance, management commentary)
                    narrative_items = [i for i in items if i.get("type") in ("driver", "guidance") or i.get("management_explanation")]
                    if narrative_items:
                        narrative_data.append(json.dumps(narrative_items[:50], indent=2, default=str))
                        logger.info("  -> Added %d narrative items", len(narrative_items))
                    else:
                        logger.info("  -> Skipped aggregation for dtype=%s", dtype)

                    output["per_document_extractions"][title] = {
                        "type": dtype, "items_count": len(items), "sample": items[:3],
                    }
                else:
                    logger.warning("  -> No items to aggregate for doc %s (dtype=%s)", title[:30] if title else "?", dtype)

            # Log aggregation totals and document processing summary
            logger.info("[BATCH] Aggregated: earnings=%d, transcript=%d, broker=%d, presentation=%d",
                        len(earnings_data), len(transcript_data), len(broker_data), len(presentation_data))
            await _add_log(job_id, f"Extraction complete: {len(earnings_data)} earnings, {len(transcript_data)} transcript, {len(broker_data)} broker, {len(presentation_data)} presentation data blocks")

            if doc_failures:
                logger.warning("[BATCH] %s: %d/%d documents failed: %s",
                               ticker, len(doc_failures), total_docs, ', '.join(doc_failures))
                await _add_log(job_id, f"Document processing: {doc_successes}/{total_docs} succeeded, {len(doc_failures)} failed: {', '.join(doc_failures)}", "warn")

            # Include extraction stats in output for UI visibility
            output["extraction_stats"] = {
                "earnings_items": len(earnings_data),
                "transcript_items": len(transcript_data),
                "broker_items": len(broker_data),
                "presentation_items": len(presentation_data),
            }

            if not last_doc_id:
                await _update_job(job_id, status="failed", error_message="No documents processed successfully")
                return

            # Thesis comparison, surprise detection, synthesis, and IR
            # question generation are now the agent pipeline's job
            # (FA + Bear + Bull + Debate + QC). Document processing here
            # stops at parse + extract — agents are triggered separately
            # by the analyst via "Analyse Period" in the cockpit.

        # ── Summary ──────────────────────────────────────────
        step_failures = list(doc_failures) if doc_failures else []

        summary_msg = (
            f"Document processing complete: {doc_successes}/{total_docs} docs parsed and extracted"
        )
        if step_failures:
            summary_msg += f", {len(step_failures)} failures: {', '.join(step_failures)}"
            logger.warning("[BATCH] %s completed with failures: %s", ticker, ', '.join(step_failures))
            await _add_log(job_id, summary_msg, "warn")
        else:
            await _add_log(job_id, summary_msg)

        await _update_job(
            job_id, status="completed", current_step="done", progress_pct=100,
            steps_completed=json.dumps(completed), result_json=json.dumps(output, default=str),
        )
        logger.info("[BATCH] Job %s completed: %s / %s | steps=%s | failures=%s",
                     job_id, ticker, period_label,
                     ', '.join(completed) or 'none',
                     ', '.join(step_failures) or 'none')

    except Exception as e:
        logger.error("[BATCH] Job %s crashed: %s", job_id, str(e), exc_info=True)
        await _add_log(job_id, f"Failed: {str(e)[:200]}", "error")
        await _update_job(
            job_id, status="failed", error_message=str(e)[:500],
            result_json=json.dumps(output, default=str) if output else None,
        )


def start_background_job(coro):
    """Launch a background coroutine as an asyncio task."""
    task = asyncio.get_event_loop().create_task(coro)
    return task
