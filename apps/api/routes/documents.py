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
# Upload + process (background — returns job ID immediately)
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/documents/upload-and-process", status_code=202)
async def upload_and_process(
    ticker: str,
    file: UploadFile = File(...),
    document_type: str = Form("earnings_release"),
    period_label: str = Form(...),
    title: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document and start background processing.
    Returns a job_id immediately — poll /jobs/{job_id} for progress.
    """
    from apps.api.models import ProcessingJob
    from services.background_processor import run_single_pipeline, start_background_job

    # Look up company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Ingest
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db, company_id=company.id, ticker=company.ticker,
            file_path=tmp_path, filename=file.filename or f"upload{suffix}",
            document_type=document_type, period_label=period_label, title=title,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    # Create job record
    job = ProcessingJob(
        id=uuid.uuid4(),
        company_id=company.id,
        period_label=period_label,
        job_type="single",
        status="queued",
        current_step="queued",
        progress_pct=0,
    )
    db.add(job)
    await db.commit()

    # Launch background processing
    start_background_job(
        run_single_pipeline(job.id, company.id, company.ticker, doc.id, period_label)
    )

    return {
        "job_id": str(job.id),
        "status": "queued",
        "message": "Processing started. Poll /api/v1/jobs/" + str(job.id) + " for progress.",
    }


# ─────────────────────────────────────────────────────────────────
# Batch upload (background)
# ─────────────────────────────────────────────────────────────────

@router.post("/companies/{ticker}/documents/batch-upload", status_code=202)
async def batch_upload_and_process(
    ticker: str,
    files: list[UploadFile] = File(...),
    period_label: str = Form(...),
    document_types: str = Form(...),
    titles: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload multiple documents and start background processing.
    Returns a job_id immediately — poll /jobs/{job_id} for progress.
    """
    from apps.api.models import ProcessingJob
    from services.background_processor import run_batch_pipeline, start_background_job

    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    doc_types_list = [t.strip() for t in document_types.split(",")]
    titles_list = [t.strip() for t in titles.split(",")] if titles else []
    while len(doc_types_list) < len(files):
        doc_types_list.append("other")
    while len(titles_list) < len(files):
        titles_list.append(None)

    # Ingest all files first (fast)
    doc_ids = []
    for i, file in enumerate(files):
        suffix = Path(file.filename).suffix if file.filename else ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            doc = await ingest_document(
                db=db, company_id=company.id, ticker=company.ticker,
                file_path=tmp_path, filename=file.filename or f"upload_{i}{suffix}",
                document_type=doc_types_list[i], period_label=period_label,
                title=titles_list[i] or (file.filename if file.filename else None),
            )
            doc_ids.append(doc.id)
        except ValueError:
            continue  # skip duplicates

    if not doc_ids:
        raise HTTPException(409, "All documents are duplicates")

    # Create job
    job = ProcessingJob(
        id=uuid.uuid4(),
        company_id=company.id,
        period_label=period_label,
        job_type="batch",
        status="queued",
        current_step="queued",
        progress_pct=0,
    )
    db.add(job)
    await db.commit()

    # Launch background processing
    start_background_job(
        run_batch_pipeline(job.id, company.id, company.ticker, doc_ids, doc_types_list[:len(doc_ids)], period_label)
    )

    return {
        "job_id": str(job.id),
        "documents_uploaded": len(doc_ids),
        "status": "queued",
        "message": "Processing started. Poll /api/v1/jobs/" + str(job.id) + " for progress.",
    }


# ─────────────────────────────────────────────────────────────────
# Job status polling
# ─────────────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}")
async def get_job_status(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Poll this endpoint for processing progress and results."""
    import json as _json
    from apps.api.models import ProcessingJob

    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")

    out = {
        "job_id": str(job.id),
        "status": job.status,
        "current_step": job.current_step,
        "progress_pct": job.progress_pct,
        "steps_completed": _json.loads(job.steps_completed) if job.steps_completed else [],
        "error_message": job.error_message,
    }

    # Include full result when completed
    if job.status == "completed" and job.result_json:
        try:
            out["result"] = _json.loads(job.result_json)
        except Exception:
            out["result"] = None

    return out


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
# Download / view a document file
# ─────────────────────────────────────────────────────────────────
@router.get("/documents/{document_id}/file")
async def download_document_file(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    from fastapi.responses import Response

    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Try file_content from DB first
    if doc.file_content:
        filename = doc.title or "document.pdf"
        suffix = Path(filename).suffix.lower()
        content_type = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
        }.get(suffix, "application/octet-stream")
        return Response(
            content=doc.file_content,
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    # Fall back to filesystem
    file_path = Path(doc.file_path)
    if file_path.exists():
        filename = doc.title or file_path.name
        suffix = file_path.suffix.lower()
        content_type = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
        }.get(suffix, "application/octet-stream")
        return Response(
            content=file_path.read_bytes(),
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    raise HTTPException(404, "File content not available")


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
