"""
Document endpoints (§8): upload, list, process, extract, compare.
"""

import logging
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from apps.api.rate_limit import limiter

logger = logging.getLogger(__name__)
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db, get_company_or_404
from apps.api.models import Company, Document, DocumentSection, ExtractedMetric, EventAssessment, ExtractionProfile, ResearchOutput, ReviewQueueItem
from configs.settings import settings
from schemas import DocumentCreate, DocumentOut
from services.document_ingestion import ingest_document
from services.document_parser import process_document
from services.metric_extractor import extract_metrics

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
@limiter.limit("10/minute")
async def upload_and_process(
    request: Request,
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

    # Dispatch: Celery worker (Sprint J / Tier 5.5) or in-process fallback.
    # Celery path survives web-container restarts; in-process dies on deploy.
    _dispatch_single_pipeline(
        job_id=job.id, company_id=company.id, ticker=company.ticker,
        doc_id=doc.id, period_label=period_label,
    )

    return {
        "job_id": str(job.id),
        "status": "queued",
        "message": "Processing started. Poll /api/v1/jobs/" + str(job.id) + " for progress.",
    }


def _dispatch_single_pipeline(
    job_id, company_id, ticker: str, doc_id, period_label: str,
) -> None:
    """Route a single-document parse+extract job to Celery when the
    feature flag is on, otherwise spawn it in-process (legacy path)."""
    if settings.use_celery_for_document_processing:
        try:
            from apps.worker.tasks import parse_and_extract_single_task
            parse_and_extract_single_task.delay(
                str(job_id), str(company_id), ticker, str(doc_id), period_label,
            )
            return
        except Exception as exc:
            logger.warning(
                "Celery dispatch failed (%s) — falling back to in-process",
                str(exc)[:200],
            )
    from services.background_processor import run_single_pipeline, start_background_job
    start_background_job(
        run_single_pipeline(job_id, company_id, ticker, doc_id, period_label)
    )


def _dispatch_batch_pipeline(
    job_id, company_id, ticker: str, doc_ids: list, doc_types: list,
    period_label: str, model: str = "standard",
) -> None:
    """Route a batch parse+extract job. Same flag as the single path."""
    if settings.use_celery_for_document_processing:
        try:
            from apps.worker.tasks import parse_and_extract_batch_task
            parse_and_extract_batch_task.delay(
                str(job_id), str(company_id), ticker,
                [str(d) for d in doc_ids], list(doc_types),
                period_label, model,
            )
            return
        except Exception as exc:
            logger.warning(
                "Celery dispatch failed (%s) — falling back to in-process",
                str(exc)[:200],
            )
    from services.background_processor import run_batch_pipeline, start_background_job
    start_background_job(
        run_batch_pipeline(job_id, company_id, ticker, doc_ids, doc_types, period_label, model=model)
    )


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

    # Dispatch to Celery (Sprint J) or in-process fallback
    _dispatch_batch_pipeline(
        job_id=job.id, company_id=company.id, ticker=company.ticker,
        doc_ids=doc_ids, doc_types=doc_types_list[:len(doc_ids)],
        period_label=period_label, model=model,
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


@router.post("/extraction/reconcile-backfill")
async def reconcile_backfill(
    limit: int = 500,
    only_missing: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Re-run the extraction reconciler over historical ExtractionProfile rows
    and persist the report.

    Historical ExtractedMetric rows don't store statement_type, so this uses
    the reconciler's alias-based lookup (Revenue, Net Income, Total Assets,
    etc.) — every period's metrics are placed under each statement bucket,
    and alias matching still finds the relevant values for each check.

    Args:
        limit: max profiles to process in one call (default 500)
        only_missing: if True, skip profiles that already have reconciliation
    """
    from services.extraction_reconciler import reconcile_extractions

    q = select(ExtractionProfile).order_by(ExtractionProfile.created_at.desc())
    if only_missing:
        q = q.where(ExtractionProfile.reconciliation.is_(None))
    q = q.limit(limit)
    profiles_result = await db.execute(q)
    profiles = profiles_result.scalars().all()

    processed = 0
    flagged = 0
    errors = 0
    for prof in profiles:
        try:
            metrics_q = await db.execute(
                select(ExtractedMetric).where(ExtractedMetric.document_id == prof.document_id)
            )
            metrics = metrics_q.scalars().all()
            if not metrics:
                continue

            # Group by (period, segment). One entry per group, placed under
            # every statement-type bucket so alias matching can find it.
            by_key: dict[tuple, dict] = {}
            for m in metrics:
                if m.metric_value is None or not m.metric_name:
                    continue
                key = (m.period_label or prof.period_label or "", m.segment)
                entry = by_key.setdefault(key, {
                    "period": key[0], "segment": key[1],
                    "data": {}, "items": [],
                })
                try:
                    entry["data"].setdefault(m.metric_name, float(m.metric_value))
                except (ValueError, TypeError):
                    continue
                entry["items"].append({
                    "line_item": m.metric_name,
                    "metric_name": m.metric_name,
                    "value": float(m.metric_value) if m.metric_value is not None else None,
                })

            recon_input: dict[str, list[dict]] = {
                "income_statements": [], "balance_sheets": [],
                "cash_flows": [], "segments": [],
            }
            for entry in by_key.values():
                if entry["segment"]:
                    recon_input["segments"].append(entry)
                else:
                    recon_input["income_statements"].append(entry)
                    recon_input["balance_sheets"].append(entry)
                    recon_input["cash_flows"].append(entry)

            report = reconcile_extractions(recon_input)
            prof.reconciliation = report
            if not report.get("passed", True):
                flagged += 1
            processed += 1
        except Exception as e:
            logger.warning("Backfill failed for profile %s: %s", prof.id, str(e)[:200])
            errors += 1

    await db.commit()
    return {
        "processed": processed,
        "flagged": flagged,
        "errors": errors,
        "scanned": len(profiles),
        "only_missing": only_missing,
    }


@router.get("/documents/{document_id}/extraction-profile")
async def get_extraction_profile(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Latest ExtractionProfile row for a document, including the
    reconciliation report (Q-sum vs FY, segment vs consolidated,
    BS equation, P&L vs CF cross-checks)."""
    result = await db.execute(
        select(ExtractionProfile)
        .where(ExtractionProfile.document_id == document_id)
        .order_by(ExtractionProfile.created_at.desc())
        .limit(1)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        return {"document_id": str(document_id), "profile": None}
    return {
        "document_id": str(document_id),
        "profile": {
            "period_label":        profile.period_label,
            "extraction_method":   profile.extraction_method,
            "sections_found":      profile.sections_found,
            "items_extracted":     profile.items_extracted,
            "detected_period":     profile.detected_period,
            "confidence_profile":  profile.confidence_profile,
            "segment_data":        profile.segment_data,
            "disappearance_flags": profile.disappearance_flags,
            "non_gaap_comparison": profile.non_gaap_comparison,
            "reconciliation":      profile.reconciliation,
            "created_at":          profile.created_at.isoformat() if profile.created_at else None,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Download / view a document file
# ─────────────────────────────────────────────────────────────────
@router.get("/documents/{document_id}/file")
async def download_document_file(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Stream the original uploaded file inline (browser PDF viewer renders it).

    The raw file on disk is treated as a cache, not the source of truth.
    If it's missing (e.g. evicted under disk pressure, or never landed
    on the persistent volume), services.doc_fetch.ensure_local_file()
    lazy-downloads from Document.source_url before serving. The user
    never sees a 404 as long as source_url is set.
    """
    from fastapi.responses import Response
    from services.doc_fetch import ensure_local_file

    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        file_path = await ensure_local_file(doc)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

    filename = doc.title or file_path.name
    suffix = file_path.suffix.lower()
    content_type = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt":  "text/plain",
        ".html": "text/html",
        ".htm":  "text/html",
    }.get(suffix, "application/octet-stream")

    # Starlette encodes response headers as latin-1. Document titles
    # contain em-dashes (e.g. "Arrow Electronics 10-K — 2025-12-31"),
    # which crash that encode. RFC 5987 handles non-ASCII filenames:
    # provide an ASCII fallback plus a percent-encoded UTF-8 version.
    import re
    from urllib.parse import quote
    ascii_fallback = re.sub(r'[^\x20-\x7e]+', '-', filename).strip('-') or "file"
    content_disposition = (
        f'inline; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )
    return Response(
        content=file_path.read_bytes(),
        media_type=content_type,
        headers={"Content-Disposition": content_disposition},
    )


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

    # Railway redeploys wipe the ephemeral filesystem. If the file is gone
    # there's no recovery path (raw PDFs are no longer stored in the DB per
    # CLAUDE.md) — the analyst has to re-upload.
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(400, "File not found on disk (likely wiped by a redeploy). Please re-upload this document.")

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


# Per-document /compare endpoint was part of the legacy pipeline.
# Thesis comparison is now the Financial Analyst agent's job — trigger
# it via POST /companies/{ticker}/run-pipeline/{period}.


# ─────────────────────────────────────────────────────────────────
# Update a document's period or type
# ─────────────────────────────────────────────────────────────────
from pydantic import BaseModel as _DocBaseModel

class _DocUpdate(_DocBaseModel):
    period_label: str | None = None
    document_type: str | None = None
    title: str | None = None

@router.patch("/documents/{document_id}")
async def update_document_metadata(document_id: uuid.UUID, body: _DocUpdate, db: AsyncSession = Depends(get_db)):
    """Update document metadata (period_label, document_type, title)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    if body.period_label is not None:
        doc.period_label = body.period_label
    if body.document_type is not None:
        doc.document_type = body.document_type
    if body.title is not None:
        doc.title = body.title
    await db.commit()
    return {"status": "updated", "id": str(doc.id), "period_label": doc.period_label, "document_type": doc.document_type, "title": doc.title}


# ─────────────────────────────────────────────────────────────────
# Fix EDGAR period labels — use reportDate from EDGAR API
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/fix-edgar-periods")
async def fix_edgar_periods(db: AsyncSession = Depends(get_db)):
    """One-time fix: correct period labels for all EDGAR documents using reportDate."""
    import httpx
    from sqlalchemy import text

    # Get all EDGAR source companies
    from services.harvester.sources.sec_edgar import EDGAR_SOURCES, _period_from_form

    headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    fixed = 0
    total = 0

    # For ALL EDGAR documents, derive correct period from the filing's
    # actual report date (fetched from EDGAR API per company)
    all_docs = await db.execute(text("""
        SELECT d.id, d.period_label, d.published_at, d.document_type, c.ticker
        FROM documents d JOIN companies c ON d.company_id = c.id
        WHERE d.source = 'sec_edgar'
    """))
    docs = all_docs.all()
    total = len(docs)

    for ticker_key, config in EDGAR_SOURCES.items():
        cik = config["cik"].lstrip("0").zfill(10)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    headers=headers,
                )
                data = resp.json()

            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            reports = filings.get("reportDate", [])

            # Build mapping: filing_date → correct period
            date_to_period = {}
            for i, form in enumerate(forms):
                filing_date = dates[i] if i < len(dates) else ""
                report_date = reports[i] if i < len(reports) else ""
                if not report_date or not filing_date:
                    continue
                correct_period = _period_from_form(form, filing_date, report_date)
                if correct_period:
                    date_to_period[filing_date] = correct_period

            # Update documents for this ticker
            for doc in docs:
                if doc.ticker != ticker_key:
                    continue
                pub_date = str(doc.published_at)[:10] if doc.published_at else ""
                if pub_date in date_to_period:
                    new_period = date_to_period[pub_date]
                    if new_period != doc.period_label:
                        await db.execute(text(
                            "UPDATE documents SET period_label = :p WHERE id = :id"
                        ), {"p": new_period, "id": str(doc.id)})
                        fixed += 1

        except Exception as e:
            logger.warning("Fix periods failed for %s: %s", ticker_key, str(e)[:100])

    await db.commit()
    return {"fixed": fixed, "total_docs": total}


# ─────────────────────────────────────────────────────────────────
# Fix EDGAR index page URLs — one-time migration
# ─────────────────────────────────────────────────────────────────
@router.post("/documents/fix-edgar-urls")
async def fix_edgar_urls(db: AsyncSession = Depends(get_db)):
    """One-time fix: resolve -index.htm URLs to actual primary document URLs."""
    import re
    import httpx
    from sqlalchemy import text

    result = await db.execute(
        text("SELECT id, source_url FROM documents WHERE source_url LIKE '%-index.htm'")
    )
    rows = result.all()
    if not rows:
        return {"fixed": 0, "message": "No index-page URLs found"}

    headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    fixed = 0
    failed = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        for doc_id, source_url in rows:
            try:
                resp = await client.get(source_url, headers=headers)
                # Find primary doc link (not exhibit, not index)
                links = re.findall(r'href="(/Archives/edgar/[^"]+\.htm)"', resp.text)
                primary = [l for l in links if 'exhibit' not in l.lower() and '-index' not in l]
                if not primary:
                    ix = re.findall(r'/ix\?doc=(/Archives/edgar/[^"]+\.htm)', resp.text)
                    primary = ix
                if primary:
                    new_url = f"https://www.sec.gov{primary[0]}"
                    await db.execute(
                        text("UPDATE documents SET source_url = :url WHERE id = :id"),
                        {"url": new_url, "id": str(doc_id)},
                    )
                    fixed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    await db.commit()
    return {"fixed": fixed, "failed": failed, "total": len(rows)}


# ─────────────────────────────────────────────────────────────────
# Delete a single document
# ─────────────────────────────────────────────────────────────────
@router.delete("/documents/{document_id}")
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete related records first (all tables with document_id FK)
    from apps.api.models import (
        HarvestedDocument, ManagementStatement,
        IngestionTriage, AgentOutput, ExtractionFeedback,
    )
    await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id == document_id))
    await db.execute(delete(EventAssessment).where(EventAssessment.document_id == document_id))
    await db.execute(delete(ExtractedMetric).where(ExtractedMetric.document_id == document_id))
    await db.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))
    await db.execute(delete(ManagementStatement).where(ManagementStatement.document_id == document_id))
    # ExtractionProfile has nullable=False FK — must delete, not unlink
    await db.execute(delete(ExtractionProfile).where(ExtractionProfile.document_id == document_id))
    # Nullable FKs — unlink to preserve audit/history rows
    await db.execute(update(HarvestedDocument).where(HarvestedDocument.document_id == document_id).values(document_id=None, ingested=False))
    await db.execute(update(IngestionTriage).where(IngestionTriage.document_id == document_id).values(document_id=None))
    await db.execute(update(AgentOutput).where(AgentOutput.document_id == document_id).values(document_id=None))
    await db.execute(update(ExtractionFeedback).where(ExtractionFeedback.document_id == document_id).values(document_id=None))
    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "document_id": str(document_id)}


# ─────────────────────────────────────────────────────────────────
# Delete ALL documents for a company
# ─────────────────────────────────────────────────────────────────
@router.delete("/companies/{ticker}/documents")
async def delete_all_documents(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await get_company_or_404(db, ticker)

    from apps.api.models import (
        HarvestedDocument, IngestionTriage, AgentOutput, ExtractionFeedback,
    )
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
            await db.execute(delete(ExtractionProfile).where(ExtractionProfile.document_id.in_(doc_ids)))
            # Null out nullable FKs so the documents delete isn't blocked
            await db.execute(
                update(HarvestedDocument)
                .where(HarvestedDocument.document_id.in_(doc_ids))
                .values(document_id=None, ingested=False)
            )
            await db.execute(
                update(IngestionTriage)
                .where(IngestionTriage.document_id.in_(doc_ids))
                .values(document_id=None)
            )
            await db.execute(
                update(AgentOutput)
                .where(AgentOutput.document_id.in_(doc_ids))
                .values(document_id=None)
            )
            await db.execute(
                update(ExtractionFeedback)
                .where(ExtractionFeedback.document_id.in_(doc_ids))
                .values(document_id=None)
            )
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

    from apps.api.models import (
        HarvestedDocument, IngestionTriage, AgentOutput, ExtractionFeedback,
    )
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
            await db.execute(delete(ExtractionProfile).where(ExtractionProfile.document_id.in_(doc_ids)))
            # Null out nullable FKs so the documents delete isn't blocked
            await db.execute(
                update(HarvestedDocument)
                .where(HarvestedDocument.document_id.in_(doc_ids))
                .values(document_id=None, ingested=False)
            )
            await db.execute(
                update(IngestionTriage)
                .where(IngestionTriage.document_id.in_(doc_ids))
                .values(document_id=None)
            )
            await db.execute(
                update(AgentOutput)
                .where(AgentOutput.document_id.in_(doc_ids))
                .values(document_id=None)
            )
            await db.execute(
                update(ExtractionFeedback)
                .where(ExtractionFeedback.document_id.in_(doc_ids))
                .values(document_id=None)
            )
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

    # Clear research outputs and synthesis data for this period,
    # but PRESERVE per-document extraction data (sections, metrics).
    # The batch pipeline will skip already-extracted documents automatically
    # (checks sections_count + metrics_count), enabling incremental extraction.
    try:
        # Clear event assessments and their review queue items
        assessments = await db.execute(select(EventAssessment.id).where(EventAssessment.document_id.in_(doc_ids)))
        assessment_ids = [a[0] for a in assessments.all()]
        if assessment_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(assessment_ids)))
        await db.execute(delete(EventAssessment).where(EventAssessment.document_id.in_(doc_ids)))
        # Clear research outputs and their review queue items
        outputs = await db.execute(
            select(ResearchOutput.id).where(
                ResearchOutput.company_id == company.id, ResearchOutput.period_label == period_label
            )
        )
        output_ids = [o[0] for o in outputs.all()]
        if output_ids:
            await db.execute(delete(ReviewQueueItem).where(ReviewQueueItem.entity_id.in_(output_ids)))
            await db.execute(delete(ResearchOutput).where(ResearchOutput.id.in_(output_ids)))
        # Clear KPI scores (synthesis-level, not extraction)
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

    # Dispatch to Celery (Sprint J) or in-process fallback
    _dispatch_batch_pipeline(
        job_id=job.id, company_id=company.id, ticker=company.ticker,
        doc_ids=doc_ids, doc_types=doc_types, period_label=period_label,
    )

    return {
        "job_id": str(job.id),
        "documents": len(doc_ids),
        "status": "queued",
        "message": f"Reprocessing {len(doc_ids)} documents for {ticker} / {period_label}",
    }


# /resynthesise was removed when the legacy pipeline was deleted. To
# re-run analysis against an updated thesis, call
# POST /companies/{ticker}/run-pipeline/{period}?force_rerun=true which
# forces a fresh agent pipeline run bypassing the cache.


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


# ─────────────────────────────────────────────────────────────────
# Peer tickers — Tier 5.1 analyst-curated peer set
# ─────────────────────────────────────────────────────────────────
class _PeerTickersBody(_DocBaseModel):
    peer_tickers: list[str]


@router.get("/companies/{ticker}/peers")
async def get_peers(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await get_company_or_404(db, ticker)
    return {"ticker": ticker, "peer_tickers": list(company.peer_tickers or [])}


@router.put("/companies/{ticker}/peers")
async def set_peers(
    ticker: str,
    body: _PeerTickersBody,
    db: AsyncSession = Depends(get_db),
):
    """Replace the peer set. Peers are stored as Bloomberg-format tickers
    (e.g. "ACE US"). Self-ticker is silently stripped."""
    company = await get_company_or_404(db, ticker)
    self_tkr = (company.ticker or "").strip().upper()
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in body.peer_tickers or []:
        t = (raw or "").strip().upper()
        if not t or t == self_tkr or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    company.peer_tickers = cleaned
    await db.commit()
    return {"ticker": ticker, "peer_tickers": cleaned}


@router.get("/edgar/proxy")
async def edgar_proxy(url: str):
    """Proxy download from SEC.gov to avoid CORS restrictions."""
    import httpx
    from fastapi.responses import Response
    if not url.startswith("https://www.sec.gov/") and not url.startswith("https://data.sec.gov/"):
        raise HTTPException(400, "Only SEC.gov URLs are allowed")
    headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return Response(content=resp.content, media_type=resp.headers.get("content-type", "text/html"))
        except Exception as exc:
            raise HTTPException(502, f"Download failed: {str(exc)[:200]}")


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
    items_list = filings.get("items", [])

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
        items = items_list[i] if i < len(items_list) else ""

        # Build document URL
        accession_nodash = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}" if primary_doc else None
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.htm"

        # Convert period_of_report to standard label
        period_label = ""
        if period:
            try:
                pdt = datetime.strptime(period, "%Y-%m-%d")
                py, pm = pdt.year, pdt.month
                if form in ("10-K", "20-F", "40-F", "ARS"):
                    period_label = f"{py}_Q4"
                elif form in ("10-Q", "6-K"):
                    period_label = f"{py}_Q{((pm - 1) // 3) + 1}"
                elif form == "8-K":
                    period_label = f"{py}_Q{((pm - 1) // 3) + 1}"
                else:
                    period_label = f"{py}_Q{((pm - 1) // 3) + 1}"
            except ValueError:
                period_label = period

        # Classify 8-K based on items (2.02 = earnings, others = material event)
        doc_type_map = {
            "10-K": "annual_report", "10-Q": "10-Q", "20-F": "annual_report",
            "40-F": "annual_report", "ARS": "annual_report", "DEF 14A": "proxy_statement",
            "6-K": "other",
        }
        _6k_note = ""
        if form == "8-K":
            inferred_type = "earnings_release" if "2.02" in items else "other"
        elif form == "6-K":
            # 6-K has no items — infer from filename and description
            _6k_ctx = f"{primary_doc} {desc}".lower()
            if any(k in _6k_ctx for k in ["earnings", "results", "current report"]):
                inferred_type = "earnings_release"
                _6k_note = "Earnings/Results"
            elif any(k in _6k_ctx for k in ["annual", "20-f"]):
                inferred_type = "annual_report"
                _6k_note = "Annual Report"
            else:
                # Check if filename has a date pattern matching quarter end (e.g. mt-20250630)
                import re as _re
                _qend = _re.search(r'(\d{4})(0[369]|12)(30|31)', primary_doc)
                if _qend:
                    inferred_type = "earnings_release"
                    _6k_note = "Quarterly Report"
                else:
                    inferred_type = "other"
                    _6k_note = ""
        else:
            inferred_type = doc_type_map.get(form, "other")

        # Add items description for 8-K / context for 6-K
        items_desc = ""
        if form == "6-K" and _6k_note:
            items_desc = " — " + _6k_note
        elif items:
            item_labels = {
                "1.01": "Material Agreement",
                "1.02": "Termination of Agreement",
                "1.03": "Bankruptcy",
                "2.01": "Acquisition/Disposition of Assets",
                "2.02": "Earnings Release",
                "2.03": "Credit Enhancement/Direct Financial Obligation",
                "2.04": "Triggering Events (Acceleration of Obligations)",
                "2.05": "Costs for Exit/Disposal Activities",
                "2.06": "Material Impairments",
                "3.01": "Delisting Notice",
                "3.02": "Unregistered Sale of Equity",
                "3.03": "Material Modification to Rights",
                "4.01": "Change in Auditor",
                "4.02": "Non-Reliance on Financial Statements",
                "5.01": "Change in Control",
                "5.02": "Director/Officer Change",
                "5.03": "Amendments to Articles/Bylaws",
                "5.04": "Temporary Suspension of Trading",
                "5.05": "Amendments to Code of Ethics",
                "5.06": "Change in Shell Company Status",
                "5.07": "Shareholder Vote Results",
                "5.08": "Shareholder Nominations",
                "7.01": "Reg FD Disclosure",
                "8.01": "Other Events",
                "9.01": "Financial Statements/Exhibits",
            }
            parts = [item_labels.get(x.strip(), x.strip()) for x in items.split(",")]
            items_desc = " — " + ", ".join(parts)

        results.append({
            "form_type": form,
            "filing_date": filing_date,
            "period_of_report": period,
            "period_label": period_label,
            "description": (desc or form) + items_desc,
            "inferred_type": inferred_type,
            "items": items,
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


async def _download_and_ingest_one(
    db: AsyncSession,
    company: Company,
    doc_url: str,
    form_type: str = "other",
    filing_date: str = "",
    period_label: str = "",
    title: str = "",
) -> dict:
    """Download a single URL and ingest it. Does not raise on download/ingest
    errors — returns a dict with status in {ingested, duplicate, failed}.
    On success the dict carries document_id, document_type and period_label,
    which the caller needs to kick off a batch pipeline.
    """
    import httpx
    import re
    from services.harvester.sources.ir_scraper import _infer_period, _classify_doc_type
    from services.doc_utils import clean_title

    context = f"{title} {doc_url} {form_type} {filing_date}"

    if not period_label:
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})\.\w+$', doc_url)
        if date_match:
            py, pm = int(date_match.group(1)), int(date_match.group(2))
            if form_type in ("10-K", "20-F", "40-F", "ARS", "annual_report"):
                period_label = f"{py}_Q4"
            else:
                period_label = f"{py}_Q{((pm - 1) // 3) + 1}"
        else:
            period_label = _infer_period(context) or ""

    if not form_type or form_type == "other":
        inferred_type = _classify_doc_type(context)
        if inferred_type != "other":
            form_type = inferred_type

    document_type = form_type if form_type else "other"
    source = "edgar" if "sec.gov" in doc_url else "ir_scrape"

    # SEC EDGAR requires a descriptive UA; most IR CDNs (e.g. Heineken) block
    # non-browser UAs with 403, so match the scraper's headers.
    if "sec.gov" in doc_url:
        headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
    else:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(doc_url, headers=headers)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            return {"status": "failed", "doc_url": doc_url,
                    "error": f"Failed to download: {str(exc)[:200]}"}

    filename = doc_url.split("/")[-1].split("?")[0] or "document.pdf"
    suffix = Path(filename).suffix or ".pdf"
    if not title:
        slug = filename.replace("-", " ").replace("_", " ")
        slug = slug.rsplit(".", 1)[0] if "." in slug else slug
        title = f"{form_type} {filing_date}".strip() if filing_date else slug[:80]
    title = clean_title(title)

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = await ingest_document(
            db=db, company_id=company.id, ticker=company.ticker,
            file_path=tmp_path, filename=filename,
            document_type=document_type, period_label=period_label,
            title=title, source=source, source_url=doc_url,
        )
        return {
            "status": "ingested",
            "doc_url": doc_url,
            "document_id": str(doc.id),
            "document_type": document_type,
            "period_label": period_label,
            "title": doc.title,
        }
    except ValueError as e:
        return {"status": "duplicate", "doc_url": doc_url, "message": str(e)}
    except Exception as e:
        return {"status": "failed", "doc_url": doc_url,
                "error": f"Ingestion failed: {str(e)[:200]}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/companies/{ticker}/ingest-url")
async def ingest_from_url(
    ticker: str,
    doc_url: str = Form(...),
    form_type: str = Form("other"),
    filing_date: str = Form(""),
    period_label: str = Form(""),
    title: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Download a document from a URL (EDGAR or IR page) and ingest it."""
    company = await get_company_or_404(db, ticker)
    result = await _download_and_ingest_one(
        db, company, doc_url, form_type, filing_date, period_label, title,
    )
    if result["status"] == "ingested":
        return {"status": "ingested", "document_id": result["document_id"], "title": result["title"]}
    if result["status"] == "duplicate":
        return {"status": "duplicate", "message": result.get("message", "")}
    err = result.get("error", "Ingestion failed")
    if err.startswith("Failed to download"):
        raise HTTPException(502, err)
    raise HTTPException(500, err)


@router.post("/companies/{ticker}/ingest-urls")
async def ingest_urls_batch(
    ticker: str,
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    """Ingest a batch of URLs and kick off one batch pipeline per period.

    Body: {"items": [{doc_url, form_type, period_label, title, filing_date}, ...]}

    After all items are ingested, successfully-ingested docs are grouped by
    period_label and a single run_batch_pipeline job is started per period —
    matching the behaviour of upload-and-process / batch-upload so that
    Results populates without a manual re-run step.
    """
    from apps.api.models import ProcessingJob

    company = await get_company_or_404(db, ticker)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "items must be a non-empty list")

    results: list[dict] = []
    # period_label -> list of (document_id: UUID, document_type: str)
    period_map: dict[str, list[tuple]] = {}

    for it in items:
        if not isinstance(it, dict):
            results.append({"status": "failed", "error": "invalid item"})
            continue
        doc_url = (it.get("doc_url") or "").strip()
        if not doc_url:
            results.append({"status": "failed", "error": "missing doc_url"})
            continue
        r = await _download_and_ingest_one(
            db, company,
            doc_url=doc_url,
            form_type=(it.get("form_type") or "other"),
            filing_date=(it.get("filing_date") or ""),
            period_label=(it.get("period_label") or ""),
            title=(it.get("title") or ""),
        )
        results.append(r)
        if r["status"] == "ingested" and r.get("period_label"):
            period_map.setdefault(r["period_label"], []).append(
                (uuid.UUID(r["document_id"]), r["document_type"] or "other")
            )

    # Kick off one batch pipeline per period of newly-ingested docs.
    jobs: list[dict] = []
    for period_label, entries in period_map.items():
        doc_ids = [e[0] for e in entries]
        doc_types = [e[1] for e in entries]
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
        # Dispatch to Celery (Sprint J) or in-process fallback
        _dispatch_batch_pipeline(
            job_id=job.id, company_id=company.id, ticker=company.ticker,
            doc_ids=doc_ids, doc_types=doc_types, period_label=period_label,
        )
        jobs.append({
            "job_id": str(job.id),
            "period": period_label,
            "doc_count": len(doc_ids),
        })

    ingested_count = sum(1 for r in results if r.get("status") == "ingested")
    dup_count = sum(1 for r in results if r.get("status") == "duplicate")
    fail_count = sum(1 for r in results if r.get("status") == "failed")
    return {
        "ingested": ingested_count,
        "duplicate": dup_count,
        "failed": fail_count,
        "results": results,
        "jobs": jobs,
    }


# Backward compatibility alias
@router.post("/companies/{ticker}/ingest-edgar")
async def ingest_edgar_filing(
    ticker: str,
    doc_url: str = Form(...),
    form_type: str = Form("other"),
    filing_date: str = Form(""),
    period_label: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Backward-compatible alias for ingest-url."""
    return await ingest_from_url(ticker, doc_url, form_type, filing_date, period_label, "", db)
