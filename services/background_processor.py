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
                    from services.metric_extractor import extract_metrics, extract_guidance
                    text_path = Path(settings.storage_base_path) / "processed" / ticker / period_label / "parsed_text.json"
                    pages = json.loads(text_path.read_text())
                    full_text = "\n\n".join(p["text"] for p in pages)
                    metrics = await extract_metrics(db, doc, full_text)
                    guidance_items = await extract_guidance(db, doc, full_text)
                    guidance = guidance_items

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

            # ── Steps 3-6: Compare, surprises, briefing, IR (parallel) ──
            await _update_step(job_id, "generating", 55, completed)
            import asyncio as _aio

            async def _compare():
                try:
                    from services.thesis_comparator import compare_thesis
                    c = await compare_thesis(db, company_id, doc_id, period_label)
                    return ("compare", c.model_dump())
                except ValueError:
                    return ("compare", None)
                except Exception as e:
                    return ("compare", {"error": str(e)[:200]})

            async def _surprises():
                try:
                    from services.surprise_detector import detect_surprises
                    s = await detect_surprises(db, company_id, doc_id, period_label)
                    return ("surprises", [x.model_dump() for x in s])
                except Exception:
                    return ("surprises", [])

            async def _briefing():
                try:
                    from services.output_generator import generate_briefing
                    b = await generate_briefing(db, company_id, period_label)
                    return ("briefing", b.model_dump())
                except Exception as e:
                    return ("briefing", {"error": str(e)[:200]})

            async def _ir():
                try:
                    from services.output_generator import generate_ir_questions
                    q = await generate_ir_questions(db, company_id, period_label)
                    return ("ir_questions", [x.model_dump() for x in q])
                except Exception:
                    return ("ir_questions", [])

            results = await _aio.gather(_compare(), _surprises(), _briefing(), _ir(), return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    continue
                if r[0] == "compare":
                    output["thesis_comparison"] = r[1]
                    completed.append("compare")
                elif r[0] == "surprises":
                    output["surprises"] = r[1]
                    completed.append("surprises")
                elif r[0] == "briefing":
                    output["briefing"] = r[1]
                    completed.append("briefing")
                elif r[0] == "ir_questions":
                    output["ir_questions"] = r[1]
                    completed.append("ir_questions")

            # ── Save to DB ───────────────────────────────────
            await _update_step(job_id, "saving", 90, completed)
            try:
                ro = ResearchOutput(
                    id=uuid.uuid4(),
                    company_id=company_id,
                    period_label=period_label,
                    output_type="full_analysis",
                    content_json=json.dumps(output, default=str),
                    review_status="draft",
                )
                db.add(ro)
                await db.commit()
            except Exception:
                pass

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

            # ── Process each document ────────────────────────
            for i, (did, dtype) in enumerate(zip(doc_ids, doc_types)):
                pct = 5 + int((i / total_docs) * 50)
                await _update_step(job_id, f"processing doc {i+1}/{total_docs}", pct, completed)

                doc_q = await db.execute(select(Document).where(Document.id == did))
                doc = doc_q.scalar_one_or_none()
                if not doc:
                    continue

                doc_result = {"filename": doc.title, "document_type": dtype, "steps": []}
                last_doc_id = did

                # Parse
                try:
                    from services.document_parser import process_document
                    parse_result = await process_document(db, doc, ticker=ticker)
                    doc_result["steps"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"]})
                except Exception as e:
                    doc_result["steps"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
                    output["documents_processed"].append(doc_result)
                    continue

                # Extract
                try:
                    from services.metric_extractor import extract_by_document_type
                    text_path = Path(settings.storage_base_path) / "processed" / ticker / period_label / "parsed_text.json"
                    pages = json.loads(text_path.read_text())
                    full_text = "\n\n".join(p["text"] for p in pages)

                    extraction = await extract_by_document_type(db, doc, full_text)
                    items = extraction.get("raw_items", [])
                    doc_result["steps"].append({"step": "extract", "status": "ok", "items": len(items)})

                    items_summary = json.dumps(items[:30], indent=2, default=str)
                    if dtype in ("earnings_release", "10-Q", "10-K", "annual_report"):
                        earnings_data.append(items_summary)
                    elif dtype == "transcript":
                        transcript_data.append(items_summary)
                    elif dtype == "broker_note":
                        broker_data.append(items_summary)
                    elif dtype == "presentation":
                        presentation_data.append(items_summary)
                    else:
                        earnings_data.append(items_summary)

                    output["per_document_extractions"][doc.title or f"doc_{i}"] = {
                        "type": dtype, "items_count": len(items), "sample": items[:3],
                    }
                except Exception as e:
                    doc_result["steps"].append({"step": "extract", "status": "error", "detail": str(e)[:200]})

                output["documents_processed"].append(doc_result)
                completed.append(f"doc_{i+1}")

            if not last_doc_id:
                await _update_job(job_id, status="failed", error_message="No documents processed successfully")
                return

            # ── Thesis comparison ────────────────────────────
            await _update_step(job_id, "comparing thesis", 60, completed)
            try:
                from services.thesis_comparator import compare_thesis
                comparison = await compare_thesis(db, company_id, last_doc_id, period_label)
                output["thesis_comparison"] = comparison.model_dump()
                completed.append("compare")
            except ValueError:
                output["thesis_comparison"] = None
            except Exception as e:
                output["thesis_comparison"] = {"error": str(e)[:200]}

            # ── Surprises ────────────────────────────────────
            try:
                from services.surprise_detector import detect_surprises
                surprises = await detect_surprises(db, company_id, last_doc_id, period_label)
                output["surprises"] = [s.model_dump() for s in surprises]
                completed.append("surprises")
            except Exception:
                output["surprises"] = []

            # ── Synthesis ────────────────────────────────────
            await _update_step(job_id, "synthesising", 75, completed)
            try:
                from services.llm_client import call_llm_json
                from prompts import SYNTHESIS_BRIEFING
                synthesis_prompt = SYNTHESIS_BRIEFING.format(
                    company=company.name if company else ticker,
                    ticker=ticker,
                    period=period_label,
                    thesis=thesis_text,
                    earnings_data="\n".join(earnings_data) if earnings_data else "No earnings data.",
                    transcript_data="\n".join(transcript_data) if transcript_data else "No transcript data.",
                    broker_data="\n".join(broker_data) if broker_data else "No broker notes.",
                    presentation_data="\n".join(presentation_data) if presentation_data else "No presentations.",
                    thesis_comparison=json.dumps(output.get("thesis_comparison"), indent=2, default=str) if output.get("thesis_comparison") else "Not available.",
                    surprises=json.dumps(output.get("surprises"), indent=2, default=str) if output.get("surprises") else "None detected.",
                )
                synthesis = call_llm_json(synthesis_prompt, max_tokens=8192)
                output["synthesis"] = synthesis
                completed.append("synthesis")
            except Exception as e:
                output["synthesis"] = {"error": str(e)[:200]}

            # ── IR Questions ─────────────────────────────────
            await _update_step(job_id, "generating questions", 85, completed)
            try:
                from services.output_generator import generate_ir_questions
                questions = await generate_ir_questions(db, company_id, period_label)
                output["ir_questions"] = [q.model_dump() for q in questions]
                completed.append("ir_questions")
            except Exception:
                output["ir_questions"] = []

            # ── Save ─────────────────────────────────────────
            await _update_step(job_id, "saving", 95, completed)
            try:
                ro = ResearchOutput(
                    id=uuid.uuid4(),
                    company_id=company_id,
                    period_label=period_label,
                    output_type="batch_synthesis",
                    content_json=json.dumps(output, default=str),
                    review_status="draft",
                )
                db.add(ro)
                await db.commit()
            except Exception:
                pass

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
