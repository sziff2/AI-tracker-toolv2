"""
Pipeline API routes — trigger and monitor Phase B agent runs.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
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


@router.get("/companies/{ticker}/agent-outputs/{period}")
async def get_agent_outputs(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all agent outputs for a company/period. Powers the Results tab."""
    from apps.api.models import AgentOutput
    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    outputs_q = await db.execute(
        select(AgentOutput)
        .where(AgentOutput.company_id == company.id)
        .where(AgentOutput.period_label == period)
        .where(AgentOutput.status.in_(["completed", "degraded"]))
        .order_by(desc(AgentOutput.created_at))
    )
    rows = outputs_q.scalars().all()

    # Deduplicate — keep latest per agent_id
    seen = set()
    outputs = {}
    for row in rows:
        if row.agent_id in seen:
            continue
        seen.add(row.agent_id)
        outputs[row.agent_id] = {
            "agent_id": row.agent_id,
            "status": row.status,
            "output": row.output_json,
            "confidence": float(row.confidence) if row.confidence else None,
            "qc_score": float(row.qc_score) if row.qc_score else None,
            "cost_usd": float(row.cost_usd) if row.cost_usd else None,
            "duration_ms": row.duration_ms,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    return {"ticker": ticker, "period": period, "agents": outputs}


# ─────────────────────────────────────────────────────────────────
# Context Contract — shared macro assumptions
# ─────────────────────────────────────────────────────────────────

@router.get("/context-contract")
async def get_context_contract(db: AsyncSession = Depends(get_db)):
    """Get the active context contract."""
    from apps.api.models import ContextContract
    q = await db.execute(
        select(ContextContract)
        .where(ContextContract.is_active == True)
        .order_by(desc(ContextContract.version))
        .limit(1)
    )
    contract = q.scalar_one_or_none()
    if not contract:
        return {"active": False}
    return {
        "active": True,
        "version": contract.version,
        "macro_assumptions": contract.macro_assumptions or {},
        "analyst_overrides": contract.analyst_overrides or {},
        "authored_by": contract.authored_by,
        "created_at": contract.created_at.isoformat() if contract.created_at else None,
    }


@router.put("/context-contract")
async def save_context_contract(request: Request, db: AsyncSession = Depends(get_db)):
    """Save a new context contract version. Deactivates previous."""
    import uuid
    from apps.api.models import ContextContract
    from sqlalchemy import update

    body = await request.json()
    macro = body.get("macro_assumptions", {})
    notes = body.get("analyst_notes", "")
    author = body.get("authored_by", "analyst")

    # Deactivate all existing
    await db.execute(
        update(ContextContract).values(is_active=False)
    )

    # Get next version number
    q = await db.execute(
        select(ContextContract.version)
        .order_by(desc(ContextContract.version))
        .limit(1)
    )
    last = q.scalar_one_or_none()
    next_version = (last or 0) + 1

    contract = ContextContract(
        id=uuid.uuid4(),
        version=next_version,
        is_active=True,
        macro_assumptions=macro,
        analyst_overrides={"notes": notes} if notes else None,
        authored_by=author,
    )
    db.add(contract)
    await db.commit()

    return {"status": "saved", "version": next_version}


@router.get("/context-contract/history")
async def get_contract_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Get recent context contract versions."""
    from apps.api.models import ContextContract
    q = await db.execute(
        select(ContextContract)
        .order_by(desc(ContextContract.version))
        .limit(limit)
    )
    contracts = q.scalars().all()
    return [{
        "version": c.version,
        "is_active": c.is_active,
        "macro_assumptions": c.macro_assumptions or {},
        "analyst_overrides": c.analyst_overrides or {},
        "authored_by": c.authored_by,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    } for c in contracts]
