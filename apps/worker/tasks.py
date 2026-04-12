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
