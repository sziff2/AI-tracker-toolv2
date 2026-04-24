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


# ─────────────────────────────────────────────────────────────────
# Celery async task runner
# ─────────────────────────────────────────────────────────────────
#
# Celery's prefork worker keeps the Python process alive across tasks.
# Each task that calls `asyncio.run(...)` creates a fresh event loop,
# runs the coroutine, then closes that loop. But services/api/database
# exports a module-level async_engine whose connection pool caches
# `AsyncConnection` objects bound to whichever event loop first used
# them. When the NEXT task starts a new event loop and SQLAlchemy
# reaches into the pool, the cached connection's cleanup paths call
# `self._loop.create_task(...)` on the OLD (now closed) loop and
# crash with `RuntimeError: Event loop is closed`.
#
# Observed in prod: a daily price refresh task populated the pool at
# 18:00:05, then a pipeline task at 18:05:43 immediately died with:
#   "Exception terminating connection <AdaptedConnection>"
#   "RuntimeError: Event loop is closed"
#   "Phase A check failed: Task <Task pending...>"
# and returned phase_a_incomplete in 0.18s without running anything.
#
# Fix: dispose the async engine INSIDE the same event loop that owns
# the connections. The finally block runs while the loop is still
# live, so pool cleanup schedules work on the correct loop. After
# dispose returns, asyncio.run closes the loop. The next task gets a
# fresh pool and is immune to cross-task contamination.
def _run_async_task(coro):
    import asyncio
    from apps.api.database import async_engine

    async def _with_engine_cleanup():
        try:
            return await coro
        finally:
            try:
                await async_engine.dispose()
            except Exception as e:
                logger.warning("async_engine.dispose() failed during task cleanup: %s", str(e)[:200])

    return asyncio.run(_with_engine_cleanup())

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
    # Weekly harvest — Monday 00:00 UTC (1 AM BST)
    "weekly-harvest": {
        "task": "apps.worker.tasks.weekly_harvest_and_report",
        "schedule": crontab(hour=0, minute=0, day_of_week=1),
    },
    # Daily price refresh — 18:00 UTC (after US market close, 7pm BST)
    "daily-prices": {
        "task": "apps.worker.tasks.refresh_prices_task",
        "schedule": crontab(hour=18, minute=0),
    },
    # Daily DB backup summary + download link — 06:00 UTC (7 AM BST)
    "daily-backup": {
        "task": "apps.worker.tasks.daily_backup_report",
        "schedule": crontab(hour=6, minute=0),
    },
    # Coverage Monitor — 14:00 UTC daily (between European close and US open).
    # Detects overdue documents and auto-triggers targeted rescans for
    # eligible gaps. Per-gap 24h rate limit + 3-attempt cap avoids
    # hammering broken sources.
    "daily-coverage-monitor": {
        "task": "apps.worker.tasks.coverage_monitor_task",
        "schedule": crontab(hour=14, minute=0),
    },
    # Monthly factor-beta refresh — 1st of each month at 22:00 UTC, after
    # the daily price feed has run for the prior month-end.
    "monthly-factor-betas": {
        "task": "apps.worker.tasks.refresh_factor_betas_task",
        "schedule": crontab(hour=22, minute=0, day_of_month=1),
    },
}


@celery_app.task(name="apps.worker.tasks.weekly_harvest_and_report")
def weekly_harvest_and_report():
    """
    Weekly auto-harvest: runs EDGAR + Investegate + IR regex scraper
    (no LLM to contain costs), saves a report, and posts to Teams.
    """
    result = _run_async_task(_async_weekly_harvest())
    logger.info("[HARVEST] Weekly run complete: %s", {k: v for k, v in result.items() if k != "details"})
    return {k: v for k, v in result.items() if k != "details"}


async def _async_weekly_harvest():
    from services.harvester.scheduler import run_and_report
    return await run_and_report(trigger="auto_weekly")


@celery_app.task(name="apps.worker.tasks.refresh_prices_task")
def refresh_prices_task():
    """Daily price refresh for all active companies."""
    result = _run_async_task(_async_refresh_prices())
    logger.info("[PRICES] Daily refresh: %s", result)
    return result


async def _async_refresh_prices():
    from services.price_feed import refresh_prices
    return await refresh_prices()


@celery_app.task(name="apps.worker.tasks.refresh_factor_betas_task")
def refresh_factor_betas_task(window_months: int = 36):
    """Monthly factor-beta refresh — recomputes per-holding β to each
    factor over the trailing window using OLS on monthly USD log returns."""
    result = _run_async_task(_async_refresh_factor_betas(window_months))
    logger.info("[FACTOR] Beta refresh: %s", {k: v for k, v in result.items() if k != "skipped"})
    return result


async def _async_refresh_factor_betas(window_months: int):
    from apps.api.database import AsyncSessionLocal
    from services.factor_analytics import refresh_all_holding_betas
    async with AsyncSessionLocal() as db:
        return await refresh_all_holding_betas(db, window_months=window_months)


@celery_app.task(name="apps.worker.tasks.harvest_new_documents")
def harvest_new_documents(tickers: list = None, skip_llm: bool = False):
    """
    Manual harvest trigger (called via API).

    Args:
        tickers: Optional list of tickers to restrict the run.
        skip_llm: If True, skip the LLM scraper.
    """
    result = _run_async_task(_async_harvest(tickers, skip_llm))
    logger.info("[HARVEST] Manual run complete: %s", {k: v for k, v in result.items() if k != "details"})
    return result


async def _async_harvest(tickers=None, skip_llm=False):
    from services.harvester import run_harvest
    return await run_harvest(tickers=tickers, skip_llm=skip_llm)


@celery_app.task(name="apps.worker.tasks.coverage_monitor_task")
def coverage_monitor_task():
    """Daily Coverage Monitor — detects overdue documents and auto-triggers
    targeted rescans for eligible gaps. Runs at 14:00 UTC via Celery Beat."""
    result = _run_async_task(_async_coverage_monitor())
    logger.info("[COVERAGE] Daily monitor complete: %s", result)
    return result


async def _async_coverage_monitor():
    from agents.ingestion.coverage_monitor import CoverageMonitor
    res = await CoverageMonitor().run_daily_check(auto_trigger=True)
    return {
        "gaps_found":                res.gaps_found,
        "rescans_triggered":         res.rescans_triggered,
        "rescans_skipped_recent":    res.rescans_skipped_recent,
        "rescans_skipped_exhausted": res.rescans_skipped_exhausted,
        "rescan_successes":          res.rescan_successes,
        "rescan_no_new":             res.rescan_no_new,
        "rescan_errors":             res.rescan_errors,
        "triggered_tickers":         res.triggered_tickers,
    }


# ─────────────────────────────────────────────────────────────────
# Phase B — Agent pipeline tasks
# ─────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="run_document_pipeline", max_retries=0)
def run_document_pipeline_task(
    self,
    company_id: str,
    period_label: str,
    agent_ids: list[str] | None = None,
    force_rerun: bool = False,
) -> dict:
    """Phase B: agent pipeline after document extraction."""
    from agents.orchestrator import AgentOrchestrator
    logger.info("run_document_pipeline: %s %s (force_rerun=%s)", company_id, period_label, force_rerun)
    try:
        result = _run_async_task(
            AgentOrchestrator().run_document_pipeline(
                company_id, period_label, agent_ids, force_rerun=force_rerun,
            )
        )
        return {
            "pipeline_run_id":  result.pipeline_run_id,
            "status":           result.status,
            "total_cost_usd":   result.total_cost_usd,
            "agents_completed": result.agents_completed,
            "agents_failed":    result.agents_failed,
            "duration_ms":      result.duration_ms,
            "error_message":    result.error_message,
        }
    except Exception as e:
        logger.exception("run_document_pipeline failed: %s", e)
        raise


@celery_app.task(bind=True, name="run_macro_refresh", max_retries=0)
def run_macro_refresh_task(self) -> dict:
    """Monthly macro agent refresh."""
    from agents.orchestrator import AgentOrchestrator
    logger.info("run_macro_refresh task started")
    try:
        result = _run_async_task(AgentOrchestrator().run_macro_refresh())
        return {"status": result.status, "agents_completed": result.agents_completed}
    except Exception as e:
        logger.exception("run_macro_refresh failed: %s", e)
        raise


@celery_app.task(bind=True, name="run_agent_on_demand", max_retries=0)
def run_agent_on_demand_task(
    self,
    agent_id: str,
    company_id: str | None = None,
    period_label: str | None = None,
) -> dict:
    """On-demand single agent with auto dependency resolution."""
    from agents.orchestrator import AgentOrchestrator
    logger.info("run_agent_on_demand: %s for %s %s", agent_id, company_id, period_label)
    try:
        result = _run_async_task(
            AgentOrchestrator().run_agent_on_demand(agent_id, company_id, period_label)
        )
        return {
            "pipeline_run_id":  result.pipeline_run_id,
            "status":           result.status,
            "agents_completed": result.agents_completed,
            "total_cost_usd":   result.total_cost_usd,
        }
    except Exception as e:
        logger.exception("run_agent_on_demand failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────
# Phase A — Document parse + extract tasks (Sprint J / Tier 5.5)
# ─────────────────────────────────────────────────────────────────
#
# These wrap services/background_processor entry points. Moving ad-hoc
# parse/extract from in-process asyncio.create_task on the web service
# to Celery on the worker service means a Railway deploy that restarts
# the web container no longer kills mid-flight document processing.

@celery_app.task(bind=True, name="parse_and_extract_single", max_retries=0)
def parse_and_extract_single_task(
    self,
    job_id: str,
    company_id: str,
    ticker: str,
    doc_id: str,
    period_label: str,
) -> dict:
    """Single-document parse + extract. Caller records job_id before dispatch;
    this task updates the ProcessingJob row as it progresses."""
    import uuid as _uuid
    from services.background_processor import run_single_pipeline
    logger.info("parse_and_extract_single: %s doc=%s period=%s", ticker, doc_id, period_label)
    try:
        _run_async_task(
            run_single_pipeline(
                _uuid.UUID(job_id), _uuid.UUID(company_id), ticker,
                _uuid.UUID(doc_id), period_label,
            )
        )
        return {"status": "ok", "job_id": job_id}
    except Exception as e:
        logger.exception("parse_and_extract_single failed: %s", e)
        raise


@celery_app.task(bind=True, name="backfill_embeddings", max_retries=0, time_limit=1800)
def backfill_embeddings_task(
    self,
    ticker: str | None = None,
    chunk_size: int = 64,
    limit: int | None = None,
) -> dict:
    """One-shot Tier 3.4 backfill. Lives on the worker because that's
    where sentence-transformers is installed (the user's laptop and
    the web container's lifespan don't load it). 30-min hard cap is
    plenty for ~3000 sections; bump if our footprint grows."""
    from apps.api.database import AsyncSessionLocal
    from scripts.backfill_embeddings import run_backfill

    async def _run():
        async with AsyncSessionLocal() as db:
            return await run_backfill(
                db, ticker=ticker, chunk_size=chunk_size,
                limit=limit, apply=True,
            )
    try:
        result = _run_async_task(_run())
        logger.info("[BACKFILL_EMBEDDINGS] %s", result)
        return result
    except Exception as e:
        logger.exception("backfill_embeddings failed: %s", e)
        raise


@celery_app.task(bind=True, name="parse_and_extract_batch", max_retries=0)
def parse_and_extract_batch_task(
    self,
    job_id: str,
    company_id: str,
    ticker: str,
    doc_ids: list[str],
    doc_types: list[str],
    period_label: str,
    model: str = "standard",
) -> dict:
    """Multi-document parse + extract. Same job_id + ProcessingJob pattern."""
    import uuid as _uuid
    from services.background_processor import run_batch_pipeline
    logger.info("parse_and_extract_batch: %s period=%s n_docs=%d",
                ticker, period_label, len(doc_ids))
    try:
        _run_async_task(
            run_batch_pipeline(
                _uuid.UUID(job_id), _uuid.UUID(company_id), ticker,
                [_uuid.UUID(d) for d in doc_ids], list(doc_types),
                period_label, model=model,
            )
        )
        return {"status": "ok", "job_id": job_id, "n_docs": len(doc_ids)}
    except Exception as e:
        logger.exception("parse_and_extract_batch failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────
# Daily DB backup — posts summary + download link to Teams
# ─────────────────────────────────────────────────────────────────

@celery_app.task(name="apps.worker.tasks.daily_backup_report")
def daily_backup_report():
    """Count rows in key tables and post a summary to Teams with a download link."""
    result = _run_async_task(_async_daily_backup())
    logger.info("[BACKUP] Daily report posted: %s", result.get("status"))
    return result


async def _async_daily_backup():
    from datetime import date
    from sqlalchemy import text
    import httpx

    from apps.api.database import AsyncSessionLocal

    TABLES = [
        "companies", "documents", "document_sections", "extracted_metrics",
        "extraction_profiles", "harvested_documents", "research_outputs",
        "valuation_scenarios", "scenario_snapshots", "price_records",
        "portfolio_holdings", "portfolios",
    ]

    row_counts = {}
    async with AsyncSessionLocal() as db:
        for table in TABLES:
            try:
                result = await db.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                row_counts[table] = result.scalar()
            except Exception:
                row_counts[table] = "error"

    today = date.today().isoformat()
    total_rows = sum(v for v in row_counts.values() if isinstance(v, int))

    # Build row count summary lines
    summary_lines = [f"**{table}**: {count:,}" if isinstance(count, int) else f"**{table}**: {count}"
                     for table, count in row_counts.items()]

    download_url = f"{settings.app_base_url}/api/v1/admin/backup"

    # Adaptive Card for Teams
    payload = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": f"💾 Daily DB Backup — {today}",
                "weight": "Bolder",
                "size": "Medium",
            },
            {
                "type": "TextBlock",
                "text": f"**Total rows**: {total_rows:,} across {len(TABLES)} tables",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": "   \n".join(summary_lines),
                "wrap": True,
                "size": "Small",
            },
        ],
        "actions": [
            {
                "type": "Action.OpenUrl",
                "title": "Download Backup JSON",
                "url": download_url,
            }
        ],
    }

    webhook_url = settings.teams_webhook_url
    if not webhook_url:
        logger.warning("[BACKUP] No TEAMS_WEBHOOK_URL configured, skipping notification")
        return {"status": "skipped", "reason": "no webhook", "row_counts": row_counts}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        logger.info("[BACKUP] Teams backup notification sent")
        return {"status": "sent", "total_rows": total_rows, "row_counts": row_counts}
    except Exception as exc:
        logger.error("[BACKUP] Teams notification failed: %s", exc)
        return {"status": "failed", "error": str(exc)[:200]}
