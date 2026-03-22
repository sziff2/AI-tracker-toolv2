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

            # ── Step 1: Parse ────────────────────────────────
            await _update_step(job_id, "parse", 10, completed)
            try:
                from services.document_parser import process_document
                parse_result = await process_document(db, doc, ticker=ticker)
                output["classification"] = parse_result["classification"]
                output["pipeline_status"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"]})
                completed.append("parse")
            except Exception as e:
                output["pipeline_status"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
                await _update_job(job_id, status="failed", error_message=f"Parse failed: {str(e)[:200]}", result_json=json.dumps(output, default=str))
                return

            # ── Step 2: Extract ──────────────────────────────
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
            except Exception as e:
                logger.error("Extraction failed for job %s: %s", job_id, str(e))
                output["pipeline_status"].append({"step": "extract", "status": "error", "detail": str(e)[:500]})
                output["extraction_error"] = str(e)[:500]

            # ── Steps 3-6: Compare, surprises, briefing, IR (parallel with separate sessions) ──
            await _update_step(job_id, "generating", 55, completed)
            generation_results = await _run_generation_steps(company_id, doc_id, period_label)
            for step_name, step_result in generation_results.items():
                output[step_name] = step_result
                completed.append(step_name)

            # ── Save to DB ───────────────────────────────────
            await _update_step(job_id, "saving", 90, completed)
            await _save_research_output(company_id, period_label, output, "full_analysis")

        # ── Done ─────────────────────────────────────────────
        await _update_job(
            job_id,
            status="completed",
            current_step="done",
            progress_pct=100,
            steps_completed=json.dumps(completed),
            result_json=json.dumps(output, default=str),
        )
        logger.info("Job %s completed: %s / %s", job_id, ticker, period_label)

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, str(e))
        await _update_job(
            job_id,
            status="failed",
            error_message=str(e)[:500],
            result_json=json.dumps(output, default=str) if output else None,
        )


async def run_batch_pipeline(
    job_id: uuid.UUID, company_id: uuid.UUID, ticker: str,
    doc_ids: list[uuid.UUID], doc_types: list[str], period_label: str,
):
    """Run the full batch pipeline in the background."""
    completed = []
    output = {}

    try:
        await _update_job(job_id, status="processing", current_step="processing_documents", progress_pct=5)

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

            earnings_data, transcript_data, broker_data, presentation_data = [], [], [], []
            last_doc_id = None
            total_docs = len(doc_ids)

            # ── Process documents in parallel ────────────────────────
            await _update_step(job_id, f"processing {total_docs} documents", 10, completed)

            async def _process_one_doc(did, dtype, idx):
                """Process a single document (parse + extract) with its own DB session."""
                from services.metric_extractor import extract_by_document_type, extract_esg, ESG_DOC_TYPES
                from services.document_parser import process_document

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
                            doc_result["steps"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
                            return {"doc_id": did, "result": doc_result, "items": [], "dtype": dtype, "esg": None}

                        # Extract - use returned text directly
                        try:
                            if dtype in ESG_DOC_TYPES:
                                extraction = await extract_esg(doc_db, doc, full_text)
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
                            else:
                                extraction = await extract_by_document_type(doc_db, doc, full_text, tables_data=tables_data)
                                items = extraction.get("raw_items", [])
                                doc_result["steps"].append({"step": "extract", "status": "ok", "items": len(items)})
                                logger.info("Extracted from %s: %d items (dtype=%s)", did, len(items), dtype)

                            # Warn if extraction returned no items
                            if not items:
                                logger.warning("No items extracted from %s (%s) - text was %d chars",
                                               doc.title, dtype, len(full_text))
                        except Exception as e:
                            doc_result["steps"].append({"step": "extract", "status": "error", "detail": str(e)[:200]})
                            logger.error("Extraction failed for doc %s (%s): %s", did, dtype, str(e)[:200])

                        return {"doc_id": did, "result": doc_result, "items": items, "dtype": dtype, "esg": esg_data, "title": doc.title}
                except Exception as e:
                    doc_result["steps"].append({"step": "error", "detail": str(e)[:200]})
                    return {"doc_id": did, "result": doc_result, "items": [], "dtype": dtype, "esg": None}

            # Run all documents in parallel (max 4 concurrent to respect API limits)
            import asyncio
            semaphore = asyncio.Semaphore(4)

            async def _limited_process(did, dtype, idx):
                async with semaphore:
                    return await _process_one_doc(did, dtype, idx)

            tasks = [_limited_process(did, dtype, i) for i, (did, dtype) in enumerate(zip(doc_ids, doc_types))]
            results = await asyncio.gather(*tasks)

            # Aggregate results
            for i, res in enumerate(results):
                if res is None:
                    logger.warning("Result %d is None - document processing failed completely", i)
                    continue
                last_doc_id = res["doc_id"]
                output["documents_processed"].append(res["result"])
                completed.append(f"doc_{i+1}")

                items = res["items"]
                dtype = res["dtype"]
                title = res.get("title", f"doc_{i}")

                logger.info("Aggregating doc %d: title=%s, dtype=%s, items=%d",
                            i, title[:50] if title else "?", dtype, len(items) if items else 0)

                if res.get("esg"):
                    output["esg_extraction"] = res["esg"]

                if items:
                    items_summary = json.dumps(items[:30], indent=2, default=str)
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
                    else:
                        logger.info("  -> Skipped aggregation for dtype=%s", dtype)

                    output["per_document_extractions"][title] = {
                        "type": dtype, "items_count": len(items), "sample": items[:3],
                    }
                else:
                    logger.warning("  -> No items to aggregate for doc %s (dtype=%s)", title[:30] if title else "?", dtype)

            # Log aggregation totals for debugging
            logger.info("Aggregated: earnings=%d, transcript=%d, broker=%d, presentation=%d",
                        len(earnings_data), len(transcript_data), len(broker_data), len(presentation_data))

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

            # ── Thesis comparison + Surprises (parallel) ─────────
            await _update_step(job_id, "comparing thesis", 60, completed)

            async def _do_compare():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.thesis_comparator import compare_thesis
                        c = await compare_thesis(db2, company_id, last_doc_id, period_label)
                        return ("thesis_comparison", c.model_dump())
                except ValueError:
                    return ("thesis_comparison", None)
                except Exception as e:
                    return ("thesis_comparison", {"error": str(e)[:200]})

            async def _do_surprises():
                try:
                    async with AsyncSessionLocal() as db2:
                        from services.surprise_detector import detect_surprises
                        s = await detect_surprises(db2, company_id, last_doc_id, period_label)
                        return ("surprises", [x.model_dump() for x in s])
                except Exception:
                    return ("surprises", [])

            # Run thesis comparison and surprises in parallel
            results = await asyncio.gather(_do_compare(), _do_surprises())
            for key, val in results:
                output[key] = val
                if key == "thesis_comparison" and val:
                    completed.append("compare")
                elif key == "surprises":
                    completed.append("surprises")

            # ── Synthesis + IR Questions (parallel) ─────────
            await _update_step(job_id, "synthesising", 75, completed)

            # Compress extraction data — preserve key details and source context
            def _compress_items(items_list, label, max_items=40):
                if not items_list:
                    return f"No {label} data."
                try:
                    all_items = []
                    for block in items_list:
                        parsed = json.loads(block) if isinstance(block, str) else block
                        if isinstance(parsed, list):
                            all_items.extend(parsed)
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
                        "text": _compress_items(earnings_data + transcript_data, "all"),
                    }
                    try:
                        synthesis_prompt = synthesis_template.format(**format_args)
                    except KeyError as ke:
                        logger.warning("Synthesis prompt format failed (bad placeholder %s), falling back to default", ke)
                        synthesis_prompt = SYNTHESIS_BRIEFING.format(**format_args)
                    synthesis = await call_llm_json_async(synthesis_prompt, max_tokens=8192)
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
            gen_results = await asyncio.gather(_do_synthesis(), _do_ir_questions())
            for key, val in gen_results:
                output[key] = val
                completed.append(key)

            # ── Save ─────────────────────────────────────────
            await _update_step(job_id, "saving", 95, completed)
            await _save_research_output(company_id, period_label, output, "batch_synthesis")

        await _update_job(
            job_id, status="completed", current_step="done", progress_pct=100,
            steps_completed=json.dumps(completed), result_json=json.dumps(output, default=str),
        )
        logger.info("Batch job %s completed: %s / %s", job_id, ticker, period_label)

    except Exception as e:
        logger.error("Batch job %s failed: %s", job_id, str(e))
        await _update_job(
            job_id, status="failed", error_message=str(e)[:500],
            result_json=json.dumps(output, default=str) if output else None,
        )


def start_background_job(coro):
    """Launch a background coroutine as an asyncio task."""
    task = asyncio.get_event_loop().create_task(coro)
    return task
