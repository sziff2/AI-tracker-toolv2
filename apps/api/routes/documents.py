"""
Document endpoints (§8): upload, list, process, extract, compare.
"""

import logging
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db, get_company_or_404
from apps.api.models import Company, Document, DocumentSection, ExtractedMetric, EventAssessment, ResearchOutput, ReviewQueueItem
from configs.settings import settings
from schemas import DocumentCreate, DocumentOut
from services.document_ingestion import ingest_document
from services.document_parser import process_document
from services.metric_extractor import extract_metrics
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
    company = await get_company_or_404(db, ticker)

    # Save upload to temp file with size limit
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large. Maximum size is {settings.max_upload_size_mb} MB.")
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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

    company = await get_company_or_404(db, ticker)

    # Ingest with size limit
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large. Maximum size is {settings.max_upload_size_mb} MB.")
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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
    model: str = Form("standard"),  # fast | standard | deep
    db: AsyncSession = Depends(get_db),
):
    """
    Upload multiple documents and start background processing.
    Returns a job_id immediately — poll /jobs/{job_id} for progress.
    """
    from apps.api.models import ProcessingJob
    from services.background_processor import run_batch_pipeline, start_background_job

    company = await get_company_or_404(db, ticker)

    doc_types_list = [t.strip() for t in document_types.split(",")]
    titles_list = [t.strip() for t in titles.split(",")] if titles else []
    while len(doc_types_list) < len(files):
        doc_types_list.append("other")
    while len(titles_list) < len(files):
        titles_list.append(None)

    # Ingest all files first (fast) with size limit
    doc_ids = []
    for i, file in enumerate(files):
        try:
            suffix = Path(file.filename).suffix if file.filename else ".pdf"
            content = await file.read()
            logger.info("Processing file %s (%d bytes)", file.filename, len(content))
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(413, f"File '{file.filename}' too large. Maximum size is {settings.max_upload_size_mb} MB.")
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            doc = await ingest_document(
                db=db, company_id=company.id, ticker=company.ticker,
                file_path=tmp_path, filename=file.filename or f"upload_{i}{suffix}",
                document_type=doc_types_list[i], period_label=period_label,
                title=titles_list[i] or (file.filename if file.filename else None),
            )
            doc_ids.append(doc.id)
            logger.info("Ingested document %s as %s", file.filename, doc.id)
        except ValueError as e:
            logger.warning("Skipping duplicate document %s: %s", file.filename, str(e))
            continue  # skip duplicates
        except Exception as e:
            logger.error("Error ingesting file %s: %s", file.filename, str(e))
            raise HTTPException(500, f"Error processing file '{file.filename}': {str(e)}")

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
        model=model,  # fast | standard | deep
    )
    db.add(job)
    await db.commit()

    # Launch background processing
    start_background_job(
        run_batch_pipeline(job.id, company.id, company.ticker, doc_ids, doc_types_list[:len(doc_ids)], period_label, model=model)
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
        "log_entries": _json.loads(job.log_entries) if job.log_entries else [],
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

    return {
        "metrics_extracted": len(metrics),
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
    company = await get_company_or_404(db, ticker)

    try:
        docs = await db.execute(select(Document).where(Document.company_id == company.id))
        doc_ids = [d.id for d in docs.scalars().all()]

        if doc_ids:
            metrics = await db.execute(select(ExtractedMetric.id).where(ExtractedMetric.document_id.in_(doc_ids)))
            metric_ids = [m[0] for m in metrics.all()]
            assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
            assessment_ids = [a[0] for a in assessments.all()]
            all_entity_ids = doc_ids + metric_ids + assessment_ids
            if all_entity_ids:
                await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(all_entity_ids)))
            await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
            await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id.in_(doc_ids)))
            await db.execute(delete(DocumentSection).where(DocumentSection.document_id.in_(doc_ids)))
            await db.execute(delete(Document).where(Document.company_id == company.id))

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


# ─────────────────────────────────────────────────────────────────
# Delete a specific period (documents, metrics, outputs for that period)
# ─────────────────────────────────────────────────────────────────
@router.delete("/companies/{ticker}/periods/{period_label}")
async def delete_period(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    company = await get_company_or_404(db, ticker)

    try:
        # Find documents for this period
        docs = await db.execute(
            select(Document).where(Document.company_id == company.id, Document.period_label == period_label)
        )
        doc_ids = [d.id for d in docs.scalars().all()]

        if doc_ids:
            metrics = await db.execute(select(ExtractedMetric.id).where(ExtractedMetric.document_id.in_(doc_ids)))
            metric_ids = [m[0] for m in metrics.all()]
            assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
            assessment_ids = [a[0] for a in assessments.all()]
            all_entity_ids = doc_ids + metric_ids + assessment_ids
            if all_entity_ids:
                await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(all_entity_ids)))
            await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
            await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id.in_(doc_ids)))
            await db.execute(delete(DocumentSection).where(DocumentSection.document_id.in_(doc_ids)))
            await db.execute(delete(Document).where(Document.id.in_(doc_ids)))

        # Also delete metrics not linked to documents for this period
        await db.execute(delete(ExtractedMetric).where(
            ExtractedMetric.company_id == company.id, ExtractedMetric.period_label == period_label
        ))

        # Delete research outputs for this period
        outputs = await db.execute(
            select(ResearchOutput.id).where(ResearchOutput.company_id == company.id, ResearchOutput.period_label == period_label)
        )
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(output_ids)))
            await db.execute(delete(ResearchOutput).where(ResearchOutput.id.in_(output_ids)))

        # Delete KPI scores for this period
        from apps.api.models import KPIScore
        await db.execute(delete(KPIScore).where(
            KPIScore.company_id == company.id, KPIScore.period_label == period_label
        ))

        await db.commit()
        return {"status": "deleted", "period": period_label, "documents_removed": len(doc_ids)}

    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Delete failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────
# Re-run analysis on existing documents (no re-upload needed)
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/periods/{period_label}/reprocess")
async def reprocess_period(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    """
    Clear existing metrics/outputs for a period and re-run the full analysis
    pipeline on the already-stored documents. No need to re-upload files.
    """
    from apps.api.models import ProcessingJob
    from services.background_processor import run_batch_pipeline, start_background_job

    company = await get_company_or_404(db, ticker)

    # Find existing documents for this period
    docs_q = await db.execute(
        select(Document).where(
            Document.company_id == company.id,
            Document.period_label == period_label,
        )
    )
    docs = docs_q.scalars().all()
    if not docs:
        raise HTTPException(404, f"No documents found for {ticker} / {period_label}")

    doc_ids = [d.id for d in docs]
    doc_types = [d.document_type or "other" for d in docs]

    # Clear existing metrics, assessments, outputs, sections for this period
    try:
        metrics = await db.execute(select(ExtractedMetric.id).where(ExtractedMetric.document_id.in_(doc_ids)))
        metric_ids = [m[0] for m in metrics.all()]
        assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
        assessment_ids = [a[0] for a in assessments.all()]
        all_entity_ids = metric_ids + assessment_ids
        if all_entity_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(all_entity_ids)))
        await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
        await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id.in_(doc_ids)))
        # Clear document sections so parser can recreate them
        await db.execute(delete(DocumentSection).where(DocumentSection.document_id.in_(doc_ids)))
        # Also clear unlinked metrics for this period
        await db.execute(delete(ExtractedMetric).where(
            ExtractedMetric.company_id == company.id, ExtractedMetric.period_label == period_label
        ))
        # Clear research outputs
        outputs = await db.execute(
            select(ResearchOutput.id).where(
                ResearchOutput.company_id == company.id, ResearchOutput.period_label == period_label
            )
        )
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(output_ids)))
            await db.execute(delete(ResearchOutput).where(ResearchOutput.id.in_(output_ids)))
        # Clear KPI scores
        from apps.api.models import KPIScore
        await db.execute(delete(KPIScore).where(
            KPIScore.company_id == company.id, KPIScore.period_label == period_label
        ))
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning("Cleanup before reprocess failed: %s", str(e)[:200])

    # Create new processing job
    job = ProcessingJob(
        id=uuid.uuid4(),
        company_id=company.id,
        period_label=period_label,
        job_type="batch",
        status="queued",
        current_step="reprocessing",
        progress_pct=0,
    )
    db.add(job)
    await db.commit()

    # Launch background processing with existing doc_ids
    start_background_job(
        run_batch_pipeline(job.id, company.id, company.ticker, doc_ids, doc_types, period_label)
    )

    return {
        "job_id": str(job.id),
        "documents": len(doc_ids),
        "status": "queued",
        "message": f"Reprocessing {len(doc_ids)} documents for {ticker} / {period_label}",
    }


@router.post("/companies/{ticker}/periods/{period_label}/resynthesise")
async def resynthesise_period(ticker: str, period_label: str, db: AsyncSession = Depends(get_db)):
    """
    Re-run only synthesis, thesis comparison, surprises, and IR questions
    using existing extracted metrics. Skips document parsing/extraction.
    Use this when you've updated the thesis and want to regenerate the output.
    """
    from apps.api.models import ProcessingJob
    from services.background_processor import run_resynthesise_pipeline, start_background_job

    company = await get_company_or_404(db, ticker)

    # Verify documents exist for this period
    docs_q = await db.execute(
        select(Document).where(
            Document.company_id == company.id,
            Document.period_label == period_label,
        )
    )
    docs = docs_q.scalars().all()
    if not docs:
        raise HTTPException(404, f"No documents found for {ticker} / {period_label}")

    # Clear existing research outputs only (keep metrics and sections)
    try:
        outputs = await db.execute(
            select(ResearchOutput.id).where(
                ResearchOutput.company_id == company.id, ResearchOutput.period_label == period_label
            )
        )
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ResearchOutput).where(ResearchOutput.id.in_(output_ids)))
        # Clear event assessments so they can be regenerated
        doc_ids = [d.id for d in docs]
        await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning("Cleanup before resynthesise failed: %s", str(e)[:200])

    # Create processing job
    job = ProcessingJob(
        id=uuid.uuid4(),
        company_id=company.id,
        period_label=period_label,
        job_type="batch",
        status="queued",
        current_step="resynthesising",
        progress_pct=0,
    )
    db.add(job)
    await db.commit()

    start_background_job(
        run_resynthesise_pipeline(job.id, company.id, company.ticker, period_label)
    )

    return {
        "job_id": str(job.id),
        "documents": len(docs),
        "status": "queued",
        "message": f"Re-synthesising {ticker} / {period_label} with updated thesis context",
    }


# ═══════════════════════════════════════════════════════════════
# EDGAR FILING BROWSER — browse and selectively ingest SEC filings
# ═══════════════════════════════════════════════════════════════

@router.get("/edgar/lookup/{ticker}")
async def lookup_cik(ticker: str, db: AsyncSession = Depends(get_db)):
    """Look up SEC CIK by ticker. Tries company_tickers.json from EDGAR."""
    import httpx

    # Check if already stored
    company = await get_company_or_404(db, ticker)
    if company.cik:
        return {"ticker": ticker, "cik": company.cik, "source": "cached"}

    # Strip exchange suffix (e.g. "LKQ US" → "LKQ", "ALLY US" → "ALLY")
    clean_ticker = ticker.split()[0].upper()

    headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise HTTPException(502, f"EDGAR lookup failed: {str(exc)[:200]}")

    # Search by ticker
    for entry in data.values():
        if entry.get("ticker", "").upper() == clean_ticker:
            cik = str(entry["cik_str"]).zfill(10)
            company.cik = cik
            await db.commit()
            return {"ticker": ticker, "cik": cik, "company_name": entry.get("title"), "source": "edgar_lookup"}

    return {"ticker": ticker, "cik": None, "source": "not_found"}


@router.post("/companies/{ticker}/set-cik")
async def set_cik(ticker: str, cik: str = Form(...), db: AsyncSession = Depends(get_db)):
    """Manually set the CIK for a company."""
    company = await get_company_or_404(db, ticker)
    company.cik = cik.strip().zfill(10)
    await db.commit()
    return {"ticker": ticker, "cik": company.cik}


@router.get("/edgar/browse/{cik}")
async def browse_edgar(cik: str, form_types: str = "10-K,10-Q,8-K,ARS,DEF 14A,20-F,40-F,6-K"):
    """Browse available SEC EDGAR filings for a given CIK. Returns a list for human review."""
    import httpx
    from datetime import datetime, timezone

    cik_padded = cik.lstrip("0").zfill(10)
    target_forms = set(f.strip() for f in form_types.split(","))
    headers = {
        "User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise HTTPException(502, f"EDGAR fetch failed: {str(exc)[:200]}")

    company_name = data.get("name", "")
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    periods = filings.get("periodOfReport", [])
    primary_docs = filings.get("primaryDocument", [])
    descriptions = filings.get("primaryDocDescription", [])

    cik_int = int(cik.lstrip("0"))
    cutoff_year = datetime.now(timezone.utc).year - 3
    results = []

    for i, form in enumerate(forms):
        if form not in target_forms:
            continue
        filing_date = dates[i] if i < len(dates) else ""
        try:
            if int(filing_date[:4]) < cutoff_year:
                continue
        except (ValueError, IndexError):
            pass

        accession = accessions[i] if i < len(accessions) else ""
        period = periods[i] if i < len(periods) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        desc = descriptions[i] if i < len(descriptions) else form

        # Build document URL
        accession_nodash = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}" if primary_doc else None
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.htm"

        results.append({
            "form_type": form,
            "filing_date": filing_date,
            "period_of_report": period,
            "description": desc,
            "accession": accession,
            "doc_url": doc_url,
            "index_url": index_url,
        })

    return {
        "company_name": company_name,
        "cik": cik,
        "total_filings": len(results),
        "filings": results,
    }


@router.post("/companies/{ticker}/ingest-edgar")
async def ingest_edgar_filing(
    ticker: str,
    doc_url: str = Form(...),
    form_type: str = Form(...),
    filing_date: str = Form(""),
    period_label: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Download a specific EDGAR filing and ingest it."""
    import httpx
    from datetime import datetime, timezone

    company = await get_company_or_404(db, ticker)

    # Map form type to document type
    doc_type_map = {
        "10-K": "annual_report", "10-Q": "earnings_release", "8-K": "earnings_release",
        "20-F": "annual_report", "40-F": "annual_report", "6-K": "earnings_release",
        "ARS": "annual_report", "DEF 14A": "proxy_statement",
    }
    document_type = doc_type_map.get(form_type, "other")

    # Download the filing
    headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(doc_url, headers=headers)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            raise HTTPException(502, f"Failed to download filing: {str(exc)[:200]}")

    # Save to temp file and ingest
    filename = doc_url.split("/")[-1] or f"{form_type}_{filing_date}.htm"
    suffix = Path(filename).suffix or ".htm"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db, company_id=company.id, ticker=ticker,
            file_path=tmp_path, filename=filename,
            document_type=document_type, period_label=period_label,
            title=f"{form_type} {filing_date}", source="edgar", source_url=doc_url,
        )
        return {"status": "ingested", "document_id": str(doc.id), "title": doc.title}
    except ValueError as e:
        return {"status": "duplicate", "message": str(e)}
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)[:200]}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
