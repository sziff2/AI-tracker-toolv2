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
    # with literal "{rates}" etc. in the text and the LLM often returns
    # malformed JSON in response.
    try:
        context_contract = await build_context_contract(db)
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

    # Set the document text
    if dtype == "transcript":
        inputs["transcript_text"] = full_text[:30000]
        agent_id = "transcript_deep_dive"
        output_type = "transcript_analysis"
    else:
        inputs["presentation_text"] = full_text[:30000]
        agent_id = "presentation_analysis"
        output_type = "presentation_analysis"

    # Build and run prompt
    prompt_result = load_prompt(agent_id, inputs)
    prompt = prompt_result[0] if isinstance(prompt_result, tuple) else prompt_result

    result = await call_llm_native_async(
        prompt,
        model=settings.agent_fast_model,  # Haiku for cost efficiency
        max_tokens=4096,
        feature=f"doc_{dtype}_analysis",
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


async def _run_generation_steps(
    company_id: uuid.UUID, doc_id: uuid.UUID, period_label: str
) -> dict:
    """Run compare, surprises, briefing, and IR questions in parallel with separate sessions."""

    async def _compare():
        try:
            async with AsyncSessionLocal() as db2:
                from services.thesis_comparator import compare_thesis
                c = await compare_thesis(db2, company_id, doc_id, period_label)
                return ("thesis_comparison", c.model_dump())
        except ValueError:
            return ("thesis_comparison", None)
        except Exception as e:
            return ("thesis_comparison", {"error": str(e)[:200]})

    async def _surprises():
        try:
            async with AsyncSessionLocal() as db2:
                from services.surprise_detector import detect_surprises
                s = await detect_surprises(db2, company_id, doc_id, period_label)
                return ("surprises", [x.model_dump() for x in s])
        except Exception as e:
            return ("surprises", {"error": str(e)[:200]})

    async def _briefing():
        try:
            async with AsyncSessionLocal() as db2:
                from services.output_generator import generate_briefing
                b = await generate_briefing(db2, company_id, period_label)
                return ("briefing", b.model_dump())
        except Exception as e:
            return ("briefing", {"error": str(e)[:200]})

    async def _ir():
        try:
            async with AsyncSessionLocal() as db2:
                from services.output_generator import generate_ir_questions
                q = await generate_ir_questions(db2, company_id, period_label)
                return ("ir_questions", [x.model_dump() for x in q])
        except Exception as e:
            return ("ir_questions", {"error": str(e)[:200]})

    tasks = await asyncio.gather(_compare(), _surprises(), _briefing(), _ir(), return_exceptions=True)
    results = {}
    for r in tasks:
        if isinstance(r, Exception):
            logger.error("Generation step failed: %s", r)
            continue
        results[r[0]] = r[1]
    return results


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

            # ── Steps 3-6: Compare, surprises, briefing, IR (parallel with separate sessions) ──
            # Run even if extraction failed — some generation steps may still produce useful output
            await _update_step(job_id, "generating", 55, completed)
            try:
                generation_results = await _run_generation_steps(company_id, doc_id, period_label)
                for step_name, step_result in generation_results.items():
                    output[step_name] = step_result
                    completed.append(step_name)
            except Exception as e:
                logger.error("[PIPELINE] %s generation steps failed for job %s: %s", ticker, job_id, e, exc_info=True)
                output["pipeline_status"].append({"step": "generate", "status": "error", "detail": str(e)[:200]})

            # ── Save to DB ───────────────────────────────────
            await _update_step(job_id, "saving", 90, completed)
            await _save_research_output(company_id, period_label, output, "full_analysis")

            # ── KPI extraction (Haiku pass) ──────────────────
            try:
                from services.kpi_extractor import extract_kpis_from_briefing
                kpi_result = await extract_kpis_from_briefing(company_id, period_label, output, doc_id)
                if kpi_result.get("status") == "ok":
                    logger.info("[PIPELINE] KPI extraction saved %d KPIs for %s/%s",
                                kpi_result.get("kpis_saved", 0), ticker, period_label)
            except Exception as e:
                logger.warning("[PIPELINE] KPI extraction failed for %s/%s: %s", ticker, period_label, e)

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
                            if dtype not in ("transcript", "presentation"):
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

            # ── Legacy synthesis (feature-flagged) ──────────────────
            if not settings.run_legacy_synthesis:
                logger.info("[BATCH] %s skipping legacy synthesis (run_legacy_synthesis=False)", ticker)
                await _update_step(job_id, "done", 100, completed)
                await _update_job(job_id, status="completed")
                await _add_log(job_id, "Document processing complete (legacy synthesis disabled).")
                return

            # ── Thesis comparison + Surprises (parallel) ─────────
            await _update_step(job_id, "comparing thesis", 60, completed)
            await _add_log(job_id, "Running thesis comparison and surprise detection...")

            async def _do_compare():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.thesis_comparator import compare_thesis
                        c = await compare_thesis(db2, company_id, last_doc_id, period_label)
                        return ("thesis_comparison", c.model_dump())
                except ValueError:
                    return ("thesis_comparison", None)
                except Exception as e:
                    logger.error("[BATCH] %s thesis comparison failed: %s", ticker, e, exc_info=True)
                    return ("thesis_comparison", {"error": str(e)[:200]})

            async def _do_surprises():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.surprise_detector import detect_surprises
                        s = await detect_surprises(db2, company_id, last_doc_id, period_label)
                        return ("surprises", [x.model_dump() for x in s])
                except Exception as e:
                    logger.error("[BATCH] %s surprise detection failed: %s", ticker, e, exc_info=True)
                    return ("surprises", [])

            # Run thesis comparison and surprises in parallel
            try:
                results = await asyncio.gather(_do_compare(), _do_surprises(), return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception):
                        logger.error("[BATCH] %s compare/surprise step crashed: %s", ticker, res, exc_info=res)
                        continue
                    key, val = res
                    output[key] = val
                    if key == "thesis_comparison" and val:
                        completed.append("compare")
                    elif key == "surprises":
                        completed.append("surprises")
            except Exception as e:
                logger.error("[BATCH] %s thesis/surprise gather failed: %s", ticker, e, exc_info=True)
                await _add_log(job_id, f"Thesis comparison/surprises failed: {str(e)[:100]}", "error")

            # ── Synthesis + IR Questions (parallel) ─────────
            await _update_step(job_id, "synthesising", 75, completed)
            await _add_log(job_id, "Starting synthesis — generating analysis output...")

            # Compress extraction data — preserve key details and source context
            def _compress_items(items_list, label, max_items=150):
                if not items_list:
                    return f"No {label} data."
                try:
                    all_items = []
                    for block in items_list:
                        parsed = json.loads(block) if isinstance(block, str) else block
                        if isinstance(parsed, list):
                            all_items.extend(parsed)
                    # Prioritise items with numeric values over text-only items
                    all_items.sort(key=lambda x: (
                        0 if x.get("metric_value") is not None else 1,
                        -(x.get("confidence") or 0),
                    ))
                    lines = []
                    for item in all_items[:max_items]:
                        name = item.get("metric_name") or item.get("topic") or item.get("category", "")
                        val = item.get("metric_value") or item.get("metric_text") or item.get("description") or ""
                        unit = item.get("unit") or ""
                        segment = item.get("segment") or item.get("geography") or ""
                        snippet = item.get("source_snippet") or ""
                        line = f"• {name}: {val} {unit}".strip()
                        if segment:
                            line += f" [{segment}]"
                        if snippet and len(snippet) < 200:
                            line += f" — \"{snippet[:150]}\""
                        lines.append(line)
                    return "\n".join(lines)
                except Exception:
                    return "\n".join(items_list)[:5000]

            async def _do_synthesis():
                try:
                    from services.llm_client import call_llm_json_async
                    from prompts import SYNTHESIS_BRIEFING
                    from services.context_builder import build_thesis_context, build_prior_period_context
                    from services.prompt_registry import get_active_prompt

                    async with AsyncSessionLocal() as db_ctx:
                        thesis_ctx = await build_thesis_context(db_ctx, company_id)
                        synthesis_template = await get_active_prompt(db_ctx, "synthesis", SYNTHESIS_BRIEFING)

                        # ── DB FALLBACK: if in-memory accumulators are empty, pull from DB ──
                        if not any([earnings_data, transcript_data, broker_data, presentation_data]):
                            logger.warning("All extraction accumulators empty - falling back to DB metrics")
                            from sqlalchemy import select as sa_select
                            from apps.api.models import ExtractedMetric
                            db_metrics_q = await db_ctx.execute(
                                sa_select(ExtractedMetric).where(
                                    ExtractedMetric.company_id == company_id,
                                    ExtractedMetric.period_label == period_label,
                                ).order_by(ExtractedMetric.confidence.desc()).limit(60)
                            )
                            db_metrics = db_metrics_q.scalars().all()
                            if db_metrics:
                                logger.info("Found %d metrics in DB for fallback", len(db_metrics))
                                lines = []
                                for m in db_metrics:
                                    val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
                                    lines.append({"metric_name": m.metric_name, "metric_value": val, "segment": m.segment})
                                fallback_summary = json.dumps(lines)
                                earnings_data.append(fallback_summary)
                            else:
                                logger.warning("No metrics found in DB either for %s/%s", ticker, period_label)

                        # ── BROKER FALLBACK: if earnings_data is empty but broker_data has items, use broker as fallback ──
                        if not earnings_data and broker_data:
                            logger.warning("No earnings data extracted — broker data used as fallback. Tagging as estimates, not actuals.")
                            for item in broker_data:
                                if isinstance(item, dict):
                                    item["data_source"] = "broker_estimate"
                                    item["is_estimate"] = True
                            earnings_data.extend(broker_data)

                    format_args = {
                        "company": company.name if company else ticker,
                        "ticker": ticker,
                        "period": period_label,
                        "thesis": thesis_ctx,
                        "earnings_data": _compress_items(earnings_data, "earnings"),
                        "transcript_data": _compress_items(transcript_data, "transcript"),
                        "broker_data": _compress_items(broker_data, "broker"),
                        "presentation_data": _compress_items(presentation_data, "presentation"),
                        "thesis_comparison": json.dumps(output.get("thesis_comparison"), default=str)[:2500] if output.get("thesis_comparison") else "Not available.",
                        "surprises": json.dumps(output.get("surprises"), default=str)[:2000] if output.get("surprises") else "None detected.",
                        "narrative_context": _compress_items(narrative_data, "narrative/MD&A") if narrative_data else "No narrative context extracted.",
                        "text": _compress_items(earnings_data + transcript_data, "all"),
                    }
                    try:
                        # Use replace() instead of format() to avoid issues with
                        # JSON curly braces in the prompt template
                        synthesis_prompt = synthesis_template
                        for key, val in format_args.items():
                            synthesis_prompt = synthesis_prompt.replace("{" + key + "}", str(val))
                    except Exception as ke:
                        logger.warning("Synthesis prompt format failed (%s), falling back to default", ke)
                        synthesis_prompt = SYNTHESIS_BRIEFING
                        for key, val in format_args.items():
                            synthesis_prompt = synthesis_prompt.replace("{" + key + "}", str(val))
                    synthesis = await call_llm_json_async(synthesis_prompt, max_tokens=8192, model=model_id)
                    return ("synthesis", synthesis)
                except Exception as e:
                    logger.error("Synthesis failed for %s/%s: %s", ticker, period_label, str(e)[:500])
                    return ("synthesis", {"error": str(e)[:200]})

            async def _do_ir_questions():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.output_generator import generate_ir_questions
                        questions = await generate_ir_questions(db2, company_id, period_label)
                        return ("ir_questions", [q.model_dump() for q in questions])
                except Exception:
                    return ("ir_questions", [])

            # Run synthesis and IR questions in parallel
            try:
                gen_results = await asyncio.gather(_do_synthesis(), _do_ir_questions(), return_exceptions=True)
                for res in gen_results:
                    if isinstance(res, Exception):
                        logger.error("[BATCH] %s synthesis/IR step crashed: %s", ticker, res, exc_info=res)
                        await _add_log(job_id, f"Synthesis/IR step crashed: {str(res)[:100]}", "error")
                        continue
                    key, val = res
                    output[key] = val
                    completed.append(key)
                    if key == "synthesis":
                        if isinstance(val, dict) and val.get("error"):
                            await _add_log(job_id, f"Synthesis error: {val.get('error', '')[:100]}", "warn")
                        else:
                            await _add_log(job_id, "Synthesis complete — generated analysis output")
                    elif key == "ir_questions":
                        await _add_log(job_id, f"Generated {len(val) if isinstance(val, list) else 0} IR questions")
            except Exception as e:
                logger.error("[BATCH] %s synthesis/IR gather failed: %s", ticker, e, exc_info=True)
                await _add_log(job_id, f"Synthesis/IR questions failed: {str(e)[:100]}", "error")

            # ── Save ─────────────────────────────────────────
            await _update_step(job_id, "saving", 95, completed)
            await _add_log(job_id, "Saving results to database...")
            await _save_research_output(company_id, period_label, output, "batch_synthesis")

            # ── KPI extraction (Haiku pass) ──────────────────
            try:
                from services.kpi_extractor import extract_kpis_from_briefing
                kpi_result = await extract_kpis_from_briefing(company_id, period_label, output)
                if kpi_result.get("status") == "ok":
                    await _add_log(job_id, f"KPI extraction: saved {kpi_result.get('kpis_saved', 0)} KPIs")
                elif kpi_result.get("status") == "skipped":
                    await _add_log(job_id, "KPI extraction skipped — no briefing text")
            except Exception as e:
                logger.warning("[BATCH] KPI extraction failed for %s/%s: %s", ticker, period_label, e)
                await _add_log(job_id, f"KPI extraction failed: {str(e)[:100]}", "warn")

        # ── Summary ──────────────────────────────────────────
        step_failures = []
        if doc_failures:
            step_failures.extend(doc_failures)
        for step_key in ("thesis_comparison", "surprises", "synthesis", "ir_questions"):
            val = output.get(step_key)
            if isinstance(val, dict) and val.get("error"):
                step_failures.append(step_key)

        summary_msg = (
            f"Analysis complete: {doc_successes}/{total_docs} docs processed, "
            f"{len(completed)} steps completed"
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


async def run_resynthesise_pipeline(
    job_id: uuid.UUID, company_id: uuid.UUID, ticker: str,
    period_label: str, model: str = "standard",
):
    """Re-run only synthesis, thesis comparison, surprises, and IR questions.
    Skips document parsing and extraction — uses existing extracted metrics from DB."""
    model_id = MODEL_MAP.get(model, MODEL_MAP["standard"])
    completed = []
    output = {}

    try:
        await _update_job(job_id, status="processing", current_step="loading_data", progress_pct=10)
        await _add_log(job_id, f"Re-synthesising {ticker} / {period_label} with updated thesis context")

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

            output["company"] = {"ticker": ticker, "name": company.name if company else ticker}
            output["period"] = period_label

            # Load existing extracted metrics from DB
            from apps.api.models import ExtractedMetric, Document
            docs_q = await db.execute(
                select(Document).where(
                    Document.company_id == company_id,
                    Document.period_label == period_label,
                )
            )
            docs = docs_q.scalars().all()
            if not docs:
                await _update_job(job_id, status="failed", error_message="No documents found for this period")
                return

            last_doc_id = docs[-1].id
            doc_types = [d.document_type or "other" for d in docs]

            metrics_q = await db.execute(
                select(ExtractedMetric).where(
                    ExtractedMetric.company_id == company_id,
                    ExtractedMetric.period_label == period_label,
                ).order_by(ExtractedMetric.confidence.desc()).limit(60)
            )
            db_metrics = metrics_q.scalars().all()

            # Build extraction data from existing metrics
            earnings_data, transcript_data, broker_data, presentation_data, narrative_data = [], [], [], [], []
            if db_metrics:
                await _add_log(job_id, f"Loaded {len(db_metrics)} existing metrics from database")
                lines = []
                for m in db_metrics:
                    val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
                    lines.append({"metric_name": m.metric_name, "metric_value": val,
                                  "segment": m.segment, "source_snippet": m.source_snippet})
                earnings_data.append(json.dumps(lines))
            else:
                await _add_log(job_id, "⚠ No existing metrics found — synthesis may be limited", "warn")

            output["documents_processed"] = [{"document_type": d.document_type, "filename": d.title} for d in docs]

            # ── Thesis comparison + Surprises (parallel) ─────────
            await _update_step(job_id, "comparing thesis", 30, completed)
            await _add_log(job_id, "Running thesis comparison and surprise detection with updated thesis...")

            async def _do_compare():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.thesis_comparator import compare_thesis
                        c = await compare_thesis(db2, company_id, last_doc_id, period_label)
                        return ("thesis_comparison", c.model_dump())
                except ValueError:
                    return ("thesis_comparison", None)
                except Exception as e:
                    logger.error("[RESYNTH] %s thesis comparison failed: %s", ticker, e, exc_info=True)
                    return ("thesis_comparison", {"error": str(e)[:200]})

            async def _do_surprises():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.surprise_detector import detect_surprises
                        s = await detect_surprises(db2, company_id, last_doc_id, period_label)
                        return ("surprises", [x.model_dump() for x in s])
                except Exception as e:
                    logger.error("[RESYNTH] %s surprise detection failed: %s", ticker, e, exc_info=True)
                    return ("surprises", [])

            try:
                results = await asyncio.gather(_do_compare(), _do_surprises(), return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception):
                        logger.error("[RESYNTH] %s compare/surprise step crashed: %s", ticker, res, exc_info=res)
                        continue
                    key, val = res
                    output[key] = val
                    if key == "thesis_comparison" and val:
                        completed.append("compare")
                    elif key == "surprises":
                        completed.append("surprises")
            except Exception as e:
                logger.error("[RESYNTH] %s thesis/surprise gather failed: %s", ticker, e, exc_info=True)
                await _add_log(job_id, f"Thesis comparison/surprises failed: {str(e)[:100]}", "error")

            # ── Synthesis + IR Questions (parallel) ─────────
            await _update_step(job_id, "synthesising", 60, completed)
            await _add_log(job_id, "Generating synthesis with updated thesis context...")

            def _compress_items(items_list, label, max_items=150):
                if not items_list:
                    return f"No {label} data."
                try:
                    all_items = []
                    for block in items_list:
                        parsed = json.loads(block) if isinstance(block, str) else block
                        if isinstance(parsed, list):
                            all_items.extend(parsed)
                    # Prioritise items with numeric values over text-only items
                    all_items.sort(key=lambda x: (
                        0 if x.get("metric_value") is not None else 1,
                        -(x.get("confidence") or 0),
                    ))
                    lines = []
                    for item in all_items[:max_items]:
                        name = item.get("metric_name") or item.get("topic") or item.get("category", "")
                        val = item.get("metric_value") or item.get("metric_text") or item.get("description") or ""
                        unit = item.get("unit") or ""
                        segment = item.get("segment") or item.get("geography") or ""
                        snippet = item.get("source_snippet") or ""
                        line = f"• {name}: {val} {unit}".strip()
                        if segment:
                            line += f" [{segment}]"
                        if snippet and len(snippet) < 200:
                            line += f" — \"{snippet[:150]}\""
                        lines.append(line)
                    return "\n".join(lines)
                except Exception:
                    return "\n".join(items_list)[:5000]

            async def _do_synthesis():
                try:
                    from services.llm_client import call_llm_json_async
                    from prompts import SYNTHESIS_BRIEFING
                    from services.context_builder import build_thesis_context
                    from services.prompt_registry import get_active_prompt

                    async with AsyncSessionLocal() as db_ctx:
                        thesis_ctx = await build_thesis_context(db_ctx, company_id)
                        synthesis_template = await get_active_prompt(db_ctx, "synthesis", SYNTHESIS_BRIEFING)

                    format_args = {
                        "company": company.name if company else ticker,
                        "ticker": ticker,
                        "period": period_label,
                        "thesis": thesis_ctx,
                        "earnings_data": _compress_items(earnings_data, "earnings"),
                        "transcript_data": _compress_items(transcript_data, "transcript"),
                        "broker_data": _compress_items(broker_data, "broker"),
                        "presentation_data": _compress_items(presentation_data, "presentation"),
                        "thesis_comparison": json.dumps(output.get("thesis_comparison"), default=str)[:2500] if output.get("thesis_comparison") else "Not available.",
                        "surprises": json.dumps(output.get("surprises"), default=str)[:2000] if output.get("surprises") else "None detected.",
                        "narrative_context": _compress_items(narrative_data, "narrative/MD&A") if narrative_data else "No narrative context extracted.",
                        "text": _compress_items(earnings_data + transcript_data, "all"),
                    }
                    try:
                        # Use replace() instead of format() to avoid issues with
                        # JSON curly braces in the prompt template
                        synthesis_prompt = synthesis_template
                        for key, val in format_args.items():
                            synthesis_prompt = synthesis_prompt.replace("{" + key + "}", str(val))
                    except Exception as ke:
                        logger.warning("Synthesis prompt format failed (%s), falling back to default", ke)
                        synthesis_prompt = SYNTHESIS_BRIEFING
                        for key, val in format_args.items():
                            synthesis_prompt = synthesis_prompt.replace("{" + key + "}", str(val))
                    synthesis = await call_llm_json_async(synthesis_prompt, max_tokens=8192, model=model_id)
                    return ("synthesis", synthesis)
                except Exception as e:
                    logger.error("[RESYNTH] %s/%s synthesis failed: %s", ticker, period_label, str(e)[:500])
                    return ("synthesis", {"error": str(e)[:200]})

            async def _do_ir_questions():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.output_generator import generate_ir_questions
                        questions = await generate_ir_questions(db2, company_id, period_label)
                        return ("ir_questions", [q.model_dump() for q in questions])
                except Exception as e:
                    logger.error("[RESYNTH] %s IR questions failed: %s", ticker, e, exc_info=True)
                    return ("ir_questions", [])

            try:
                gen_results = await asyncio.gather(_do_synthesis(), _do_ir_questions(), return_exceptions=True)
                for res in gen_results:
                    if isinstance(res, Exception):
                        logger.error("[RESYNTH] %s synthesis/IR step crashed: %s", ticker, res, exc_info=res)
                        await _add_log(job_id, f"Synthesis/IR step crashed: {str(res)[:100]}", "error")
                        continue
                    key, val = res
                    output[key] = val
                    completed.append(key)
                    if key == "synthesis":
                        if isinstance(val, dict) and val.get("error"):
                            await _add_log(job_id, f"Synthesis error: {val.get('error', '')[:100]}", "warn")
                        else:
                            await _add_log(job_id, "Synthesis complete — generated with updated thesis context")
                    elif key == "ir_questions":
                        await _add_log(job_id, f"Generated {len(val) if isinstance(val, list) else 0} IR questions")
            except Exception as e:
                logger.error("[RESYNTH] %s synthesis/IR gather failed: %s", ticker, e, exc_info=True)
                await _add_log(job_id, f"Synthesis/IR questions failed: {str(e)[:100]}", "error")

            # ── Save ─────────────────────────────────────────
            await _update_step(job_id, "saving", 95, completed)
            await _add_log(job_id, "Saving results to database...")
            await _save_research_output(company_id, period_label, output, "batch_synthesis")

            # ── KPI extraction (Haiku pass) ──────────────────
            try:
                from services.kpi_extractor import extract_kpis_from_briefing
                kpi_result = await extract_kpis_from_briefing(company_id, period_label, output)
                if kpi_result.get("status") == "ok":
                    await _add_log(job_id, f"KPI extraction: saved {kpi_result.get('kpis_saved', 0)} KPIs")
                elif kpi_result.get("status") == "skipped":
                    await _add_log(job_id, "KPI extraction skipped — no briefing text")
            except Exception as e:
                logger.warning("[RESYNTH] KPI extraction failed for %s/%s: %s", ticker, period_label, e)
                await _add_log(job_id, f"KPI extraction failed: {str(e)[:100]}", "warn")

        # ── Summary ──────────────────────────────────────────
        step_failures = []
        for step_key in ("thesis_comparison", "surprises", "synthesis", "ir_questions"):
            val = output.get(step_key)
            if isinstance(val, dict) and val.get("error"):
                step_failures.append(step_key)

        summary_msg = f"Re-synthesis complete: {len(completed)} steps completed"
        if step_failures:
            summary_msg += f", {len(step_failures)} failures: {', '.join(step_failures)}"
            logger.warning("[RESYNTH] %s completed with failures: %s", ticker, ', '.join(step_failures))
            await _add_log(job_id, summary_msg, "warn")
        else:
            await _add_log(job_id, summary_msg)

        await _update_job(
            job_id, status="completed", current_step="done", progress_pct=100,
            steps_completed=json.dumps(completed), result_json=json.dumps(output, default=str),
        )
        logger.info("[RESYNTH] Job %s completed: %s / %s | steps=%s | failures=%s",
                     job_id, ticker, period_label,
                     ', '.join(completed) or 'none',
                     ', '.join(step_failures) or 'none')

    except Exception as e:
        logger.error("[RESYNTH] Job %s crashed: %s", job_id, str(e), exc_info=True)
        await _add_log(job_id, f"Failed: {str(e)[:200]}", "error")
        await _update_job(
            job_id, status="failed", error_message=str(e)[:500],
            result_json=json.dumps(output, default=str) if output else None,
        )


def start_background_job(coro):
    """Launch a background coroutine as an asyncio task."""
    task = asyncio.get_event_loop().create_task(coro)
    return task
