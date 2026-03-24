"""
Celery worker — handles background tasks (§10 Trigger System).

Start with:
    celery -A apps.worker.tasks worker --loglevel=info
    celery -A apps.worker.tasks beat   --loglevel=info   (scheduler)
"""

import logging
import uuid

from celery import Celery
from celery.schedules import crontab
from sqlalchemy import select

from configs.settings import settings

logger = logging.getLogger(__name__)

# ── Celery app ───────────────────────────────────────────────────
celery_app = Celery(
    "research_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


# ─────────────────────────────────────────────────────────────────
# Scheduled tasks
# ─────────────────────────────────────────────────────────────────
celery_app.conf.beat_schedule = {
    # Legacy daily scan (kept for backward compat)
    "daily-source-scan": {
        "task": "apps.worker.tasks.scan_sources",
        "schedule": crontab(hour=6, minute=0),
    },
    # Harvesting agent — polls IR feeds & EDGAR every 30 minutes
    "harvest-documents": {
        "task": "apps.worker.tasks.harvest_new_documents",
        "schedule": crontab(minute="*/30"),
    },
}


@celery_app.task(name="apps.worker.tasks.scan_sources")
def scan_sources():
    """
    Legacy daily scan placeholder.
    Real harvesting is now handled by harvest_new_documents().
    """
    logger.info("Starting daily source scan …")
    logger.info("Daily source scan complete.")
    return {"status": "completed"}


# ─────────────────────────────────────────────────────────────────
# Harvesting Agent tasks
# ─────────────────────────────────────────────────────────────────

@celery_app.task(name="apps.worker.tasks.harvest_new_documents")
def harvest_new_documents(tickers: list = None):
    """
    Poll IR RSS feeds and SEC EDGAR for new documents.
    Runs every 30 minutes via Celery beat.

    Auto-ingests new docs and triggers the full extraction pipeline.
    Posts Slack notification for each new document found.

    Args:
        tickers: Optional list of tickers to restrict the run.
                 Defaults to all active companies with configured sources.
    """
    import asyncio
    result = asyncio.run(_async_harvest(tickers))
    logger.info("[HARVEST] Run complete: %s", result)
    return result


async def _async_harvest(tickers=None):
    from services.harvester import run_harvest
    return await run_harvest(tickers=tickers)


# ─────────────────────────────────────────────────────────────────
# Event-driven tasks
# ─────────────────────────────────────────────────────────────────

@celery_app.task(name="apps.worker.tasks.process_document_task")
def process_document_task(document_id: str):
    """
    Full pipeline for a single document:
      1. Parse
      2. Extract metrics
      3. Compare thesis
      4. Detect surprises
    """
    import asyncio
    asyncio.run(_async_process(document_id))


async def _async_process(document_id: str):
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import Company, Document
    from services.document_parser import process_document
    from services.metric_extractor import extract_metrics, extract_guidance
    from services.thesis_comparator import compare_thesis
    from services.surprise_detector import detect_surprises
    import json
    from pathlib import Path

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.id == uuid.UUID(document_id))
        )
        doc = result.scalar_one_or_none()
        if not doc:
            logger.error("Document %s not found", document_id)
            return

        company_result = await db.execute(select(Company).where(Company.id == doc.company_id))
        company = company_result.scalar_one_or_none()
        ticker = company.ticker if company else "UNKNOWN"

        # 1. Parse
        logger.info("[%s] Step 1: Parsing …", document_id)
        summary = await process_document(db, doc, ticker=ticker)

        # 2. Extract
        logger.info("[%s] Step 2: Extracting metrics …", document_id)
        text_path = (
            Path(settings.storage_base_path) / "processed" / ticker / (doc.period_label or "misc") / "parsed_text.json"
        )
        if text_path.exists():
            pages = json.loads(text_path.read_text())
            full_text = "\n\n".join(p["text"] for p in pages)
            await extract_metrics(db, doc, full_text)
            await extract_guidance(db, doc, full_text)

        # 3. Compare thesis
        logger.info("[%s] Step 3: Thesis comparison …", document_id)
        try:
            await compare_thesis(db, doc.company_id, doc.id, doc.period_label)
        except ValueError:
            logger.warning("No active thesis — skipping comparison")

        # 4. Detect surprises
        logger.info("[%s] Step 4: Surprise detection …", document_id)
        await detect_surprises(db, doc.company_id, doc.id, doc.period_label)

        logger.info("[%s] Pipeline complete.", document_id)


@celery_app.task(name="apps.worker.tasks.generate_outputs_task")
def generate_outputs_task(company_id: str, period_label: str):
    """Generate all outputs for a company/period."""
    import asyncio
    asyncio.run(_async_generate(company_id, period_label))


async def _async_generate(company_id: str, period_label: str):
    from apps.api.database import AsyncSessionLocal
    from services.output_generator import generate_briefing, generate_ir_questions

    async with AsyncSessionLocal() as db:
        cid = uuid.UUID(company_id)
        await generate_briefing(db, cid, period_label)
        await generate_ir_questions(db, cid, period_label)
        logger.info("[%s/%s] Output generation complete.", company_id, period_label)
