"""
Pipeline API routes — trigger and monitor Phase B agent runs.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from apps.api.database import get_db
from apps.api.models import PipelineRun, Company, ResearchOutput

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pipeline"])


@router.get("/companies/{ticker}/phase-a-status/{period}")
async def get_phase_a_status(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Check whether Phase A (extraction) is complete for a company + period.
    UI calls this to decide whether to enable the Run Analysis button.
    """
    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    ctx_q = await db.execute(
        select(ResearchOutput)
        .where(ResearchOutput.company_id == company.id)
        .where(ResearchOutput.period_label == period)
        .where(ResearchOutput.output_type == "extraction_context")
        .order_by(desc(ResearchOutput.created_at))
        .limit(1)
    )
    ctx = ctx_q.scalar_one_or_none()
    if not ctx:
        return {"phase_a_complete": False, "extraction_method": None}

    import json
    method = None
    try:
        data = json.loads(ctx.content_json or "{}")
        method = data.get("extraction_method")
    except Exception:
        pass

    return {"phase_a_complete": True, "extraction_method": method}


@router.post("/companies/{ticker}/run-pipeline/{period}")
async def trigger_pipeline(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger Phase B agent pipeline. Phase A must be complete.
    Returns immediately — UI polls for status.
    """
    from apps.worker.tasks import run_document_pipeline_task
    from agents.orchestrator import AgentOrchestrator

    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    orch = AgentOrchestrator()
    if not await orch._is_phase_a_complete(db, str(company.id), period):
        raise HTTPException(
            status_code=400,
            detail=f"Extraction not complete for {ticker} {period}. Process document first.",
        )

    task = run_document_pipeline_task.delay(str(company.id), period)
    return {
        "status": "queued",
        "task_id": task.id,
        "ticker": ticker,
        "period": period,
    }


@router.post("/companies/{ticker}/run-agent/{agent_id}")
async def trigger_agent_on_demand(
    ticker: str,
    agent_id: str,
    period: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a specific agent on demand. Powers [Run Deep Dive] buttons.
    Auto-resolves upstream dependencies.
    """
    from apps.worker.tasks import run_agent_on_demand_task

    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    task = run_agent_on_demand_task.delay(agent_id, str(company.id), period)
    return {
        "status": "queued",
        "task_id": task.id,
        "agent_id": agent_id,
        "ticker": ticker,
        "period": period,
    }


@router.get("/pipeline-runs/latest/{ticker}/{period}")
async def get_latest_pipeline_run(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """Poll endpoint — UI calls every 3s to update agent timeline."""
    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    run_q = await db.execute(
        select(PipelineRun)
        .where(PipelineRun.company_id == company.id)
        .where(PipelineRun.period_label == period)
        .order_by(desc(PipelineRun.started_at))
        .limit(1)
    )
    run = run_q.scalar_one_or_none()
    if not run:
        return {"status": "not_started"}

    return {
        "pipeline_run_id":     str(run.id),
        "status":              run.status,
        "started_at":          run.started_at.isoformat() if run.started_at else None,
        "completed_at":        run.completed_at.isoformat() if run.completed_at else None,
        "duration_ms":         run.duration_ms,
        "total_cost_usd":      float(run.total_cost_usd or 0),
        "agents_planned":      run.agents_planned,
        "agents_completed":    run.agents_completed,
        "agents_failed":       run.agents_failed,
        "overall_qc_score":    run.overall_qc_score,
        "error_message":       run.error_message,
        "agent_execution_log": run.agent_execution_log or [],
    }


@router.get("/pipeline-runs/{pipeline_run_id}")
async def get_pipeline_run(
    pipeline_run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific pipeline run by ID."""
    import uuid as uuid_mod
    try:
        run_id = uuid_mod.UUID(pipeline_run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline_run_id")

    q = await db.execute(select(PipelineRun).where(PipelineRun.id == run_id))
    run = q.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    return {
        "pipeline_run_id":      str(run.id),
        "status":               run.status,
        "period_label":         run.period_label,
        "started_at":           run.started_at.isoformat() if run.started_at else None,
        "completed_at":         run.completed_at.isoformat() if run.completed_at else None,
        "duration_ms":          run.duration_ms,
        "total_cost_usd":       float(run.total_cost_usd or 0),
        "total_input_tokens":   run.total_input_tokens,
        "total_output_tokens":  run.total_output_tokens,
        "total_llm_calls":      run.total_llm_calls,
        "agents_completed":     run.agents_completed,
        "agents_failed":        run.agents_failed,
        "overall_qc_score":     run.overall_qc_score,
        "error_message":        run.error_message,
        "agent_execution_log":  run.agent_execution_log or [],
    }
