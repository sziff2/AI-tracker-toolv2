"""
AutoRun API Routes — two-pipeline autonomous prompt optimisation.

POST /autorun/start   {"hours": 8, "prompt_types": [...], "pipeline": "extraction"|"output"|"both"}
POST /autorun/stop    {"pipeline": "extraction"|"output"|"both"}
GET  /autorun/status  → full state for both pipelines + combined log
"""
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from apps.api.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["autorun"])

try:
    from scripts.autorun import (
        run_autorun_loop, get_state, request_stop,
        EXTRACTION_PROMPT_TYPES, OUTPUT_PROMPT_TYPES,
    )
    _available = True
except ImportError as e:
    logger.warning("AutoRun module not available: %s", e)
    _available = False
    EXTRACTION_PROMPT_TYPES = []
    OUTPUT_PROMPT_TYPES = []


class AutoRunRequest(BaseModel):
    hours: float = 8.0
    prompt_types: Optional[list[str]] = None
    pipeline: str = "extraction"   # "extraction" | "output" | "both"
    dry_run: bool = False


class StopRequest(BaseModel):
    pipeline: str = "both"


@router.post("/autorun/start")
@limiter.limit("10/minute")
async def start_autorun(request: Request, body: AutoRunRequest, background_tasks: BackgroundTasks):
    if not _available:
        raise HTTPException(503, "AutoRun module not loaded. Check scripts/autorun.py exists.")
    if body.hours <= 0 or body.hours > 24:
        raise HTTPException(400, "hours must be between 0 and 24")

    state = get_state()
    pipelines = ["extraction", "output"] if body.pipeline == "both" else [body.pipeline]
    for p in pipelines:
        if state.get(p, {}).get("running"):
            raise HTTPException(409, f"{p} pipeline already running (job {state[p].get('job_id')}). Stop it first.")

    background_tasks.add_task(
        run_autorun_loop,
        hours=body.hours,
        prompt_types=body.prompt_types,
        pipeline=body.pipeline,
        dry_run=body.dry_run,
    )
    return {
        "status": "started",
        "pipeline": body.pipeline,
        "hours": body.hours,
        "prompt_types": body.prompt_types,
    }


@router.post("/autorun/stop")
async def stop_autorun(body: StopRequest):
    if not _available:
        raise HTTPException(503, "AutoRun module not loaded.")
    request_stop(body.pipeline)
    return {"status": "stop_requested", "pipeline": body.pipeline}


@router.get("/autorun/status")
async def autorun_status():
    if not _available:
        return {"available": False, "extraction": {}, "output": {}, "log": []}
    state = get_state()
    state["available"] = True
    state["extraction_prompt_types"] = EXTRACTION_PROMPT_TYPES
    state["output_prompt_types"] = OUTPUT_PROMPT_TYPES
    return state


# ═══════════════════════════════════════════════════════════════
# LLM Usage Analytics
# ═══════════════════════════════════════════════════════════════

@router.get("/usage")
async def get_usage(days: int = 7):
    """Get LLM usage stats for the last N days."""
    from sqlalchemy import select, func, text
    from apps.api.database import AsyncSessionLocal
    from apps.api.models import LLMUsageLog
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        # Total summary
        totals_q = await db.execute(
            select(
                func.count(LLMUsageLog.id),
                func.sum(LLMUsageLog.input_tokens),
                func.sum(LLMUsageLog.output_tokens),
                func.sum(LLMUsageLog.cost_usd),
            ).where(LLMUsageLog.timestamp >= cutoff)
        )
        row = totals_q.one()
        total_calls = row[0] or 0
        total_input = row[1] or 0
        total_output = row[2] or 0
        total_cost = float(row[3] or 0)

        # By feature
        by_feature_q = await db.execute(
            select(
                LLMUsageLog.feature,
                func.count(LLMUsageLog.id),
                func.sum(LLMUsageLog.input_tokens),
                func.sum(LLMUsageLog.output_tokens),
                func.sum(LLMUsageLog.cost_usd),
            ).where(LLMUsageLog.timestamp >= cutoff)
            .group_by(LLMUsageLog.feature)
            .order_by(func.sum(LLMUsageLog.cost_usd).desc())
        )
        by_feature = [{
            "feature": r[0], "calls": r[1],
            "input_tokens": r[2] or 0, "output_tokens": r[3] or 0,
            "cost_usd": round(float(r[4] or 0), 4),
        } for r in by_feature_q.all()]

        # By model
        by_model_q = await db.execute(
            select(
                LLMUsageLog.model,
                func.count(LLMUsageLog.id),
                func.sum(LLMUsageLog.input_tokens),
                func.sum(LLMUsageLog.output_tokens),
                func.sum(LLMUsageLog.cost_usd),
            ).where(LLMUsageLog.timestamp >= cutoff)
            .group_by(LLMUsageLog.model)
            .order_by(func.sum(LLMUsageLog.cost_usd).desc())
        )
        by_model = [{
            "model": r[0], "calls": r[1],
            "input_tokens": r[2] or 0, "output_tokens": r[3] or 0,
            "cost_usd": round(float(r[4] or 0), 4),
        } for r in by_model_q.all()]

        # By day
        by_day_q = await db.execute(
            select(
                func.date_trunc('day', LLMUsageLog.timestamp).label('day'),
                func.count(LLMUsageLog.id),
                func.sum(LLMUsageLog.cost_usd),
            ).where(LLMUsageLog.timestamp >= cutoff)
            .group_by(text("1"))
            .order_by(text("1"))
        )
        by_day = [{
            "date": r[0].isoformat()[:10] if r[0] else None,
            "calls": r[1],
            "cost_usd": round(float(r[2] or 0), 4),
        } for r in by_day_q.all()]

        # By ticker (top 10)
        by_ticker_q = await db.execute(
            select(
                LLMUsageLog.ticker,
                func.count(LLMUsageLog.id),
                func.sum(LLMUsageLog.cost_usd),
            ).where(LLMUsageLog.timestamp >= cutoff, LLMUsageLog.ticker.isnot(None))
            .group_by(LLMUsageLog.ticker)
            .order_by(func.sum(LLMUsageLog.cost_usd).desc())
            .limit(10)
        )
        by_ticker = [{
            "ticker": r[0], "calls": r[1],
            "cost_usd": round(float(r[2] or 0), 4),
        } for r in by_ticker_q.all()]

    return {
        "period_days": days,
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 4),
        "by_feature": by_feature,
        "by_model": by_model,
        "by_day": by_day,
        "by_ticker": by_ticker,
    }
