"""
Document endpoints (§8): upload, list, process, extract, compare.
"""

import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, Document, DocumentSection, ExtractedMetric, EventAssessment, ResearchOutput, ReviewQueueItem
from schemas import DocumentCreate, DocumentOut
from services.document_ingestion import ingest_document
from services.document_parser import process_document
from services.metric_extractor import extract_metrics, extract_guidance
from services.thesis_comparator import compare_thesis

router = APIRouter(tags=["documents"])


# ─────────────────────────────────────────────────────────────────
# List documents for a company
# ─────────────────────────────────────────────────────────────────
@router.get("/companies/{ticker}/documents", response_model=list[DocumentOut])
async def list_documents(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document)
        .join(Company)
        .where(Company.ticker == ticker.upper())
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────
# Upload a document
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(
    ticker: str,
    file: UploadFile = File(...),
    document_type: str = Form(...),
    period_label: str = Form(...),
    title: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Save upload to temp file
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db,
            company_id=company.id,
            ticker=company.ticker,
            file_path=tmp_path,
            filename=file.filename or f"upload{suffix}",
            document_type=document_type,
            period_label=period_label,
            title=title,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    return doc


# ─────────────────────────────────────────────────────────────────
# Upload + full auto pipeline
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/upload-and-process", status_code=200)
async def upload_and_process(
    ticker: str,
    file: UploadFile = File(...),
    document_type: str = Form("earnings_release"),
    period_label: str = Form(...),
    title: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    One-click pipeline: upload → parse → extract KPIs → compare thesis → generate all outputs.
    Returns full content of every step.
    """
    import json as _json
    from services.output_generator import generate_briefing, generate_ir_questions, generate_thesis_drift_report

    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    output = {
        "company": {"ticker": company.ticker, "name": company.name},
        "period": period_label,
        "pipeline_status": [],
        "classification": None,
        "metrics": [],
        "guidance": [],
        "thesis_comparison": None,
        "surprises": [],
        "briefing": None,
        "ir_questions": [],
        "thesis_drift": None,
    }

    # ── Step 1: Ingest ───────────────────────────────────────────
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db,
            company_id=company.id,
            ticker=company.ticker,
            file_path=tmp_path,
            filename=file.filename or f"upload{suffix}",
            document_type=document_type,
            period_label=period_label,
            title=title,
        )
        output["pipeline_status"].append({"step": "upload", "status": "ok", "document_id": str(doc.id)})
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    # ── Step 2: Parse ────────────────────────────────────────────
    try:
        parse_result = await process_document(db, doc, ticker=company.ticker)
        output["classification"] = parse_result["classification"]
        output["pipeline_status"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"], "tables": parse_result["tables_found"]})
    except Exception as e:
        output["pipeline_status"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
        return output

    # ── Step 3: Extract KPIs ─────────────────────────────────────
    try:
        from configs.settings import settings as _settings
        text_path = Path(_settings.storage_base_path) / "processed" / company.ticker / period_label / "parsed_text.json"
        pages = _json.loads(text_path.read_text())
        full_text = "\n\n".join(p["text"] for p in pages)

        metrics = await extract_metrics(db, doc, full_text)
        guidance = await extract_guidance(db, doc, full_text)

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
        output["guidance"] = guidance
        output["pipeline_status"].append({"step": "extract", "status": "ok", "metrics": len(metrics), "guidance": len(guidance)})
    except Exception as e:
        output["pipeline_status"].append({"step": "extract", "status": "error", "detail": str(e)[:200]})

    # ── Steps 4-6: Run thesis comparison, surprises, briefing, IR questions IN PARALLEL ──
    import asyncio as _aio

    async def _compare():
        try:
            c = await compare_thesis(db, company.id, doc.id, period_label)
            return ("compare", c.model_dump(), c.thesis_direction)
        except ValueError:
            return ("compare", None, "skipped")
        except Exception as e:
            return ("compare", None, str(e)[:200])

    async def _surprises():
        try:
            from services.surprise_detector import detect_surprises as _ds
            s = await _ds(db, company.id, doc.id, period_label)
            return ("surprises", [x.model_dump() for x in s])
        except Exception as e:
            return ("surprises", [], str(e)[:200])

    async def _briefing():
        try:
            b = await generate_briefing(db, company.id, period_label)
            return ("briefing", b.model_dump())
        except Exception as e:
            return ("briefing", None, str(e)[:200])

    async def _ir():
        try:
            q = await generate_ir_questions(db, company.id, period_label)
            return ("ir_questions", [x.model_dump() for x in q])
        except Exception as e:
            return ("ir_questions", [], str(e)[:200])

    # Run all four in parallel
    results = await _aio.gather(_compare(), _surprises(), _briefing(), _ir(), return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            continue
        if r[0] == "compare":
            output["thesis_comparison"] = r[1]
            status = "ok" if r[1] else ("skipped" if r[2] == "skipped" else "error")
            output["pipeline_status"].append({"step": "compare", "status": status})
        elif r[0] == "surprises":
            output["surprises"] = r[1]
            output["pipeline_status"].append({"step": "surprises", "status": "ok", "count": len(r[1])})
        elif r[0] == "briefing":
            output["briefing"] = r[1]
            output["pipeline_status"].append({"step": "briefing", "status": "ok" if r[1] else "error"})
        elif r[0] == "ir_questions":
            output["ir_questions"] = r[1]
            output["pipeline_status"].append({"step": "ir_questions", "status": "ok", "count": len(r[1])})

    # Save full analysis to DB for history
    try:
        import json as _save_json
        ro = ResearchOutput(
            id=uuid.uuid4(),
            company_id=company.id,
            period_label=period_label,
            output_type="full_analysis",
            content_json=_save_json.dumps(output, default=str),
            review_status="draft",
        )
        db.add(ro)
        await db.commit()
    except Exception:
        pass

    return output


# ─────────────────────────────────────────────────────────────────
# Batch upload: type-specific extraction per doc, then synthesis
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/batch-upload", status_code=200)
async def batch_upload_and_process(
    ticker: str,
    files: list[UploadFile] = File(...),
    period_label: str = Form(...),
    document_types: str = Form(...),
    titles: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload multiple documents for the same period. Each document is
    processed with a TYPE-SPECIFIC prompt (earnings get number extraction,
    transcripts get tone/guidance analysis, broker notes get consensus
    comparison). Then everything is synthesised into one opinionated briefing.

    document_types: comma-separated list matching each file, e.g. "earnings_release,transcript,broker_note"
    titles: comma-separated list matching each file (optional)
    """
    import json as _json
    from services.metric_extractor import extract_by_document_type
    from services.surprise_detector import detect_surprises
    from services.output_generator import generate_ir_questions
    from services.llm_client import call_llm_json
    from prompts import SYNTHESIS_BRIEFING

    # Parse comma-separated strings into lists
    doc_types_list = [t.strip() for t in document_types.split(",")]
    titles_list = [t.strip() for t in titles.split(",")] if titles else []

    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Load thesis
    from apps.api.models import ThesisVersion
    thesis_q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company.id, ThesisVersion.active == True
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()
    thesis_text = thesis.core_thesis if thesis else "No thesis on file."

    if not titles_list or titles_list == ['']:
        titles_list = [None] * len(files)
    while len(doc_types_list) < len(files):
        doc_types_list.append("other")
    while len(titles_list) < len(files):
        titles_list.append(None)

    # Buckets for type-specific extraction results
    earnings_data = []
    transcript_data = []
    broker_data = []
    presentation_data = []

    output = {
        "company": {"ticker": company.ticker, "name": company.name},
        "period": period_label,
        "documents_processed": [],
        "per_document_extractions": {},
        "thesis_comparison": None,
        "surprises": [],
        "synthesis": None,
        "ir_questions": [],
    }

    last_doc_id = None

    # ── Process each file with type-specific prompts ─────────────
    for i, file in enumerate(files):
        doc_type = doc_types_list[i]
        doc_result = {"filename": file.filename, "document_type": doc_type, "steps": []}

        # Ingest
        suffix = Path(file.filename).suffix if file.filename else ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc = await ingest_document(
                db=db,
                company_id=company.id,
                ticker=company.ticker,
                file_path=tmp_path,
                filename=file.filename or f"upload_{i}{suffix}",
                document_type=doc_type,
                period_label=period_label,
                title=titles_list[i] or file.filename,
            )
            doc_result["document_id"] = str(doc.id)
            doc_result["steps"].append({"step": "upload", "status": "ok"})
            last_doc_id = doc.id
        except ValueError as exc:
            doc_result["steps"].append({"step": "upload", "status": "skipped", "detail": str(exc)})
            output["documents_processed"].append(doc_result)
            continue

        # Parse
        try:
            parse_result = await process_document(db, doc, ticker=company.ticker)
            doc_result["steps"].append({"step": "parse", "status": "ok", "pages": parse_result["pages"]})
        except Exception as e:
            doc_result["steps"].append({"step": "parse", "status": "error", "detail": str(e)[:200]})
            output["documents_processed"].append(doc_result)
            continue

        # Type-specific extraction
        try:
            from configs.settings import settings as _settings
            text_path = Path(_settings.storage_base_path) / "processed" / company.ticker / period_label / "parsed_text.json"
            pages = _json.loads(text_path.read_text())
            full_text = "\n\n".join(p["text"] for p in pages)

            extraction = await extract_by_document_type(db, doc, full_text)
            items = extraction.get("raw_items", [])
            doc_result["steps"].append({
                "step": "extract",
                "status": "ok",
                "prompt_type": doc_type,
                "items_extracted": len(items),
            })

            # Route to the right bucket
            items_summary = _json.dumps(items[:30], indent=2, default=str)  # cap for prompt size
            if doc_type in ("earnings_release", "10-Q", "10-K", "annual_report"):
                earnings_data.append(items_summary)
            elif doc_type == "transcript":
                transcript_data.append(items_summary)
            elif doc_type == "broker_note":
                broker_data.append(items_summary)
            elif doc_type == "presentation":
                presentation_data.append(items_summary)
            else:
                earnings_data.append(items_summary)  # fallback

            output["per_document_extractions"][file.filename] = {
                "type": doc_type,
                "items_count": len(items),
                "sample": items[:5],
            }

        except Exception as e:
            doc_result["steps"].append({"step": "extract", "status": "error", "detail": str(e)[:200]})

        output["documents_processed"].append(doc_result)

    if last_doc_id is None:
        return output

    # ── Thesis comparison (uses all metrics in DB for this period) ─
    try:
        comparison = await compare_thesis(db, company.id, last_doc_id, period_label)
        output["thesis_comparison"] = comparison.model_dump()
    except ValueError:
        output["thesis_comparison"] = None
    except Exception as e:
        output["thesis_comparison"] = {"error": str(e)[:200]}

    # ── Surprises ─────────────────────────────────────────────────
    try:
        surprises = await detect_surprises(db, company.id, last_doc_id, period_label)
        output["surprises"] = [s.model_dump() for s in surprises]
    except Exception:
        pass

    # ── SYNTHESIS: combine all sources into one briefing ──────────
    try:
        synthesis_prompt = SYNTHESIS_BRIEFING.format(
            company=company.name,
            ticker=company.ticker,
            period=period_label,
            thesis=thesis_text,
            earnings_data="\n".join(earnings_data) if earnings_data else "No earnings data available.",
            transcript_data="\n".join(transcript_data) if transcript_data else "No transcript data available.",
            broker_data="\n".join(broker_data) if broker_data else "No broker notes available.",
            presentation_data="\n".join(presentation_data) if presentation_data else "No presentation data available.",
            thesis_comparison=_json.dumps(output.get("thesis_comparison"), indent=2, default=str) if output.get("thesis_comparison") else "Not available.",
            surprises=_json.dumps(output.get("surprises"), indent=2, default=str) if output.get("surprises") else "None detected.",
        )
        synthesis = call_llm_json(synthesis_prompt, max_tokens=8192)
        output["synthesis"] = synthesis
    except Exception as e:
        output["synthesis"] = {"error": str(e)[:200]}

    # ── IR Questions ──────────────────────────────────────────────
    try:
        questions = await generate_ir_questions(db, company.id, period_label)
        output["ir_questions"] = [q.model_dump() for q in questions]
    except Exception:
        pass

    # Save full analysis to DB for history
    try:
        ro = ResearchOutput(
            id=uuid.uuid4(),
            company_id=company.id,
            period_label=period_label,
            output_type="batch_synthesis",
            content_json=_json.dumps(output, default=str),
            review_status="draft",
        )
        db.add(ro)
        await db.commit()
    except Exception:
        pass

    return output


# ─────────────────────────────────────────────────────────────────
# Get single document
# ─────────────────────────────────────────────────────────────────
@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


# ─────────────────────────────────────────────────────────────────
# Process (parse) a document
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/process")
async def process_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    # Load company to get ticker (avoid lazy load in async context)
    company_result = await db.execute(select(Company).where(Company.id == doc.company_id))
    company = company_result.scalar_one_or_none()
    ticker = company.ticker if company else "UNKNOWN"

    # Restore file from DB if missing on disk (Railway redeploys wipe filesystem)
    file_path = Path(doc.file_path)
    if not file_path.exists() and doc.file_content:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(doc.file_content)

    if not file_path.exists():
        raise HTTPException(400, "File not found on disk and no content stored in DB. Please re-upload.")

    summary = await process_document(db, doc, ticker=ticker)
    return summary


# ─────────────────────────────────────────────────────────────────
# Extract metrics from a document
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/extract")
async def extract_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Load company to get ticker (avoid lazy load in async context)
    company_result = await db.execute(select(Company).where(Company.id == doc.company_id))
    company = company_result.scalar_one_or_none()
    ticker = company.ticker if company else "UNKNOWN"

    # Read parsed text
    from configs.settings import settings
    proc_dir = Path(settings.storage_base_path) / "processed"
    text_path = proc_dir / ticker / (doc.period_label or "misc") / "parsed_text.json"

    if not text_path.exists():
        raise HTTPException(400, "Document has not been processed yet. Call /process first.")

    import json
    pages = json.loads(text_path.read_text())
    full_text = "\n\n".join(p["text"] for p in pages)

    metrics = await extract_metrics(db, doc, full_text)
    guidance = await extract_guidance(db, doc, full_text)

    return {
        "metrics_extracted": len(metrics),
        "guidance_items": len(guidance),
    }


# ─────────────────────────────────────────────────────────────────
# Compare document against thesis
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/{document_id}/compare")
async def compare_doc(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        comparison = await compare_thesis(db, doc.company_id, doc.id, doc.period_label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return comparison.model_dump()


# ─────────────────────────────────────────────────────────────────
# Delete a single document
# ─────────────────────────────────────────────────────────────────
@router.delete("/documents/{document_id}")
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete related records first
    await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id == document_id))
    await db.execute(delete(EventAssessment).where(EventAssessment.document_id == document_id))
    await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id == document_id))
    await db.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))
    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "document_id": str(document_id)}


# ─────────────────────────────────────────────────────────────────
# Delete ALL documents for a company
# ─────────────────────────────────────────────────────────────────
@router.delete("/companies/{ticker}/documents")
async def delete_all_documents(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    try:
        docs = await db.execute(select(Document).where(Document.company_id == company.id))
        doc_ids = [d.id for d in docs.scalars().all()]

        if doc_ids:
            # Get metric and assessment IDs for review queue cleanup
            metrics = await db.execute(select(ExtractedMetric.id).where(ExtractedMetric.document_id.in_(doc_ids)))
            metric_ids = [m[0] for m in metrics.all()]

            assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
            assessment_ids = [a[0] for a in assessments.all()]

            # Delete review queue items (could reference metrics, assessments, or outputs)
            all_entity_ids = doc_ids + metric_ids + assessment_ids
            if all_entity_ids:
                await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(all_entity_ids)))

            # Delete in dependency order
            await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
            await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id.in_(doc_ids)))
            await db.execute(delete(DocumentSection).where(DocumentSection.document_id.in_(doc_ids)))
            await db.execute(delete(Document).where(Document.company_id == company.id))

        # Also clean up research outputs and their review items
        outputs = await db.execute(select(ResearchOutput.id).where(ResearchOutput.company_id == company.id))
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(output_ids)))
            await db.execute(delete(ResearchOutput).where(ResearchOutput.company_id == company.id))

        await db.commit()
        return {"status": "deleted", "documents_removed": len(doc_ids)}

    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Delete failed: {str(e)}")
