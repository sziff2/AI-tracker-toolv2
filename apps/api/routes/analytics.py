"""
Analytics routes — cost tracking and spend visibility.

Surfaces cumulative Anthropic API spend per company and month so the
analyst can see where the LLM budget is going without digging through
logs or the per-run pipeline panels.
"""
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from apps.api.database import get_db
from apps.api.models import Company, PipelineRun

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analytics"])


@router.get("/analytics/costs")
async def get_cost_breakdown(db: AsyncSession = Depends(get_db)):
    """
    Aggregate pipeline_runs cost by company + month.

    Returns a list of rows:
      {ticker, company_name, month, run_count, total_cost_usd,
       total_input_tokens, total_output_tokens, avg_cost_per_run}

    Sorted by month DESC, then total cost DESC within each month.
    Rows with no company_id (macro-only runs) are grouped under
    ticker="(macro)".
    """
    month_col = func.date_trunc("month", PipelineRun.started_at).label("month")

    q = await db.execute(
        select(
            Company.ticker,
            Company.name,
            month_col,
            func.count(PipelineRun.id).label("run_count"),
            func.coalesce(func.sum(PipelineRun.total_cost_usd), 0).label("total_cost_usd"),
            func.coalesce(func.sum(PipelineRun.total_input_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(PipelineRun.total_output_tokens), 0).label("total_output_tokens"),
            func.coalesce(func.sum(PipelineRun.total_llm_calls), 0).label("total_llm_calls"),
        )
        .outerjoin(Company, PipelineRun.company_id == Company.id)
        .where(PipelineRun.started_at.is_not(None))
        .group_by(Company.ticker, Company.name, month_col)
        .order_by(month_col.desc(), func.sum(PipelineRun.total_cost_usd).desc().nulls_last())
    )

    rows = []
    total_spend = 0.0
    total_runs = 0
    for r in q.all():
        cost = float(r.total_cost_usd or 0)
        runs = int(r.run_count or 0)
        total_spend += cost
        total_runs += runs
        rows.append({
            "ticker": r.ticker or "(macro)",
            "company_name": r.name or "Macro / system",
            "month": r.month.isoformat() if r.month else None,
            "run_count": runs,
            "total_cost_usd": round(cost, 4),
            "total_input_tokens": int(r.total_input_tokens or 0),
            "total_output_tokens": int(r.total_output_tokens or 0),
            "total_llm_calls": int(r.total_llm_calls or 0),
            "avg_cost_per_run": round(cost / runs, 4) if runs > 0 else 0.0,
        })

    return {
        "rows": rows,
        "summary": {
            "total_spend_usd": round(total_spend, 4),
            "total_runs": total_runs,
            "avg_cost_per_run": round(total_spend / total_runs, 4) if total_runs > 0 else 0.0,
        },
    }


@router.get("/analytics/costs/recent")
async def get_recent_runs(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """
    Most recent pipeline runs with per-run cost, for a "recent activity"
    list at the top of the cost dashboard.
    """
    q = await db.execute(
        select(
            PipelineRun.id,
            PipelineRun.started_at,
            PipelineRun.completed_at,
            PipelineRun.duration_ms,
            PipelineRun.status,
            PipelineRun.period_label,
            PipelineRun.total_cost_usd,
            PipelineRun.total_llm_calls,
            PipelineRun.trigger,
            Company.ticker,
            Company.name.label("company_name"),
        )
        .outerjoin(Company, PipelineRun.company_id == Company.id)
        .where(PipelineRun.started_at.is_not(None))
        .order_by(PipelineRun.started_at.desc())
        .limit(limit)
    )

    out = []
    for r in q.all():
        out.append({
            "pipeline_run_id": str(r.id),
            "ticker": r.ticker or "(macro)",
            "company_name": r.company_name or "Macro / system",
            "period_label": r.period_label,
            "status": r.status,
            "trigger": r.trigger,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "duration_ms": r.duration_ms,
            "total_cost_usd": round(float(r.total_cost_usd or 0), 4),
            "total_llm_calls": int(r.total_llm_calls or 0),
        })
    return {"runs": out}
