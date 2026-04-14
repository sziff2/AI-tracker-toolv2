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
    Phase A is complete only when EVERY document in the period has been
    parsed AND has had extraction run against it (ExtractionProfile row
    present). The second condition is critical: the batch pipeline sets
    parsing_status='completed' before extraction finishes, so a docs-only
    check leaves a multi-minute window where an analyst can fire agents
    against an empty dataset on a large 10-Q.
    """
    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    from apps.api.models import Document, ExtractionProfile
    from sqlalchemy import func

    # Per-status document count for this period
    rows = await db.execute(
        select(Document.parsing_status, func.count(Document.id))
        .where(Document.company_id == company.id)
        .where(Document.period_label == period)
        .group_by(Document.parsing_status)
    )
    counts = {status or "unknown": int(n) for status, n in rows.all()}
    total = sum(counts.values())
    done = counts.get("completed", 0)
    pending = counts.get("pending", 0)
    processing = counts.get("processing", 0)
    failed = counts.get("failed", 0)

    # Distinct documents with at least one ExtractionProfile row
    prof_q = await db.execute(
        select(func.count(func.distinct(ExtractionProfile.document_id)))
        .where(ExtractionProfile.company_id == company.id)
        .where(ExtractionProfile.period_label == period)
    )
    extracted = int(prof_q.scalar() or 0)

    all_parsed = total > 0 and done == total
    all_extracted = all_parsed and extracted == total
    extraction_in_progress = all_parsed and extracted < total

    # Pending/extracting doc metadata (for UI to show what's blocking)
    blocking_docs: list[dict] = []
    if (not all_extracted) and total > 0:
        if all_parsed:
            # Extraction is the bottleneck — show docs still missing a profile
            extracted_ids_q = await db.execute(
                select(func.distinct(ExtractionProfile.document_id))
                .where(ExtractionProfile.company_id == company.id)
                .where(ExtractionProfile.period_label == period)
            )
            extracted_ids = {row[0] for row in extracted_ids_q.all()}
            pd_q = await db.execute(
                select(Document.id, Document.document_type, Document.source_url)
                .where(Document.company_id == company.id)
                .where(Document.period_label == period)
            )
            for did, dtype, url in pd_q.all():
                if did in extracted_ids:
                    continue
                blocking_docs.append({
                    "id": str(did),
                    "document_type": dtype,
                    "parsing_status": "extracting",
                    "source_url": url,
                })
        else:
            # Parsing is the bottleneck
            pd_q = await db.execute(
                select(Document.id, Document.document_type, Document.parsing_status, Document.source_url)
                .where(Document.company_id == company.id)
                .where(Document.period_label == period)
                .where(Document.parsing_status != "completed")
            )
            for did, dtype, pstatus, url in pd_q.all():
                blocking_docs.append({
                    "id": str(did),
                    "document_type": dtype,
                    "parsing_status": pstatus,
                    "source_url": url,
                })

    # Extraction method (from most recent extraction_context row)
    method = None
    if extracted > 0:
        ctx_q = await db.execute(
            select(ResearchOutput)
            .where(ResearchOutput.company_id == company.id)
            .where(ResearchOutput.period_label == period)
            .where(ResearchOutput.output_type == "extraction_context")
            .order_by(desc(ResearchOutput.created_at))
            .limit(1)
        )
        ctx = ctx_q.scalar_one_or_none()
        if ctx:
            import json
            try:
                method = json.loads(ctx.content_json or "{}").get("extraction_method")
            except Exception:
                pass

    return {
        "phase_a_complete": all_extracted,
        "extraction_method": method or ("legacy" if extracted > 0 else None),
        "total_documents": total,
        "parsed_documents": done,
        "extracted_documents": extracted,
        "pending_documents": pending,
        "processing_documents": processing,
        "failed_documents": failed,
        "processing": processing > 0 or pending > 0 or extraction_in_progress,
        "extraction_in_progress": extraction_in_progress,
        "pending_doc_details": blocking_docs,
    }


@router.post("/companies/{ticker}/run-pipeline/{period}")
async def trigger_pipeline(
    ticker: str,
    period: str,
    force_rerun: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger Phase B agent pipeline. Phase A must be complete.
    Returns immediately — UI polls for status.

    If a completed pipeline run already exists for this (company, period)
    the orchestrator returns its id without re-running any agents. Pass
    `?force_rerun=true` to bypass this cache and run the pipeline fresh.
    """
    from apps.worker.tasks import run_document_pipeline_task
    from agents.orchestrator import AgentOrchestrator

    q = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = q.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    orch = AgentOrchestrator()
    if not await orch._is_phase_a_complete(db, str(company.id), period):
        # Report exactly what's missing so the UI can prompt the analyst
        from apps.api.models import Document
        from sqlalchemy import func
        rows = await db.execute(
            select(Document.parsing_status, func.count(Document.id))
            .where(Document.company_id == company.id)
            .where(Document.period_label == period)
            .group_by(Document.parsing_status)
        )
        counts = {s or "unknown": int(n) for s, n in rows.all()}
        total = sum(counts.values())
        done = counts.get("completed", 0)
        missing = total - done
        if total == 0:
            msg = f"No documents in {ticker} {period}. Upload or ingest documents first."
        else:
            msg = (
                f"{missing} of {total} documents still need to be parsed "
                f"for {ticker} {period}. Run document processing first."
            )
        raise HTTPException(status_code=400, detail=msg)

    # Cache short-circuit: if a completed run already exists and the
    # caller didn't force a re-run, return its id immediately without
    # dispatching a Celery task. UI treats this as instant success.
    if not force_rerun:
        cached = await orch._find_cached_run(db, str(company.id), period)
        if cached is not None:
            return {
                "status": "cached",
                "pipeline_run_id": str(cached.id),
                "ticker": ticker,
                "period": period,
                "cached_run_completed_at": (
                    cached.completed_at.isoformat() if cached.completed_at else None
                ),
                "cached_run_cost_usd": float(cached.total_cost_usd or 0),
            }

    task = run_document_pipeline_task.delay(str(company.id), period, None, force_rerun)
    return {
        "status": "queued",
        "task_id": task.id,
        "ticker": ticker,
        "period": period,
        "force_rerun": force_rerun,
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
    """Get the latest agent outputs for a company/period (one row per agent).
    Uses a SQL-level DISTINCT ON to avoid loading every historical run."""
    from apps.api.models import AgentOutput
    from sqlalchemy import text
    q = await db.execute(select(Company.id).where(Company.ticker == ticker.upper()))
    co_row = q.first()
    if not co_row:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    company_id = co_row[0]

    # Use raw SQL DISTINCT ON — one row per agent_id, latest only.
    # Much faster than loading all rows and dedup'ing in Python.
    sql = text("""
        SELECT DISTINCT ON (agent_id)
            agent_id, status, output_json, confidence, qc_score,
            cost_usd, duration_ms, created_at
        FROM agent_outputs
        WHERE company_id = :cid
          AND period_label = :period
          AND status IN ('completed', 'degraded')
        ORDER BY agent_id, created_at DESC
    """)
    result = await db.execute(sql, {"cid": str(company_id), "period": period})

    outputs = {}
    for row in result.mappings():
        outputs[row["agent_id"]] = {
            "agent_id": row["agent_id"],
            "status": row["status"],
            "output": row["output_json"],
            "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            "qc_score": float(row["qc_score"]) if row["qc_score"] is not None else None,
            "cost_usd": float(row["cost_usd"]) if row["cost_usd"] is not None else None,
            "duration_ms": row["duration_ms"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
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


@router.post("/admin/normalise-period-labels")
async def normalise_period_labels(
    ticker: str | None = None,
    dry_run: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """One-shot backfill: rewrite extracted_metrics.period_label to canonical YYYY_QN.

    Query params:
      - ticker: optional, limit to one company (e.g. 'ALLY US'). Omit for all.
      - dry_run: if true, report what would change without writing.
    """
    from sqlalchemy import text
    from services.metric_normaliser import normalise_period

    where = ""
    params: dict = {}
    if ticker:
        q = await db.execute(select(Company.id).where(Company.ticker == ticker.upper()))
        row = q.first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
        where = "WHERE company_id = :cid"
        params["cid"] = str(row[0])

    rows = await db.execute(
        text(f"SELECT DISTINCT period_label FROM extracted_metrics {where}"), params
    )
    labels = [r[0] for r in rows.fetchall() if r[0]]

    changes: list[dict] = []
    total_updated = 0
    for lbl in labels:
        canon = normalise_period(lbl)
        if canon == lbl:
            continue
        if dry_run:
            count_row = await db.execute(
                text(
                    f"SELECT COUNT(*) FROM extracted_metrics {where}"
                    + (" AND " if where else " WHERE ")
                    + "period_label = :old"
                ),
                {**params, "old": lbl},
            )
            n = count_row.scalar() or 0
        else:
            upd = await db.execute(
                text(
                    f"UPDATE extracted_metrics SET period_label = :new {where}"
                    + (" AND " if where else " WHERE ")
                    + "period_label = :old"
                ),
                {**params, "old": lbl, "new": canon},
            )
            n = upd.rowcount or 0
        total_updated += n
        changes.append({"from": lbl, "to": canon, "rows": n})

    if not dry_run:
        await db.commit()

    return {
        "ticker": ticker,
        "dry_run": dry_run,
        "distinct_labels_seen": len(labels),
        "labels_rewritten": len(changes),
        "rows_updated": total_updated,
        "changes": changes,
    }


@router.post("/admin/sync-profile-periods")
async def sync_profile_periods(
    ticker: str | None = None,
    dry_run: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Backfill extraction_profiles.period_label to match the period_label
    on the associated Document row.

    Fixes cases where ExtractionProfile rows were written before canonical
    period labels were standardised (e.g. "Q2 2025" or "FY 2025" instead of
    "2025_Q2" / "2025_Q4"), leaving them out of sync with the Document they
    belong to. This matters because the Phase A completion gate joins
    Document.period_label against ExtractionProfile.period_label — orphaned
    profiles make docs look "not yet extracted" and hide the Re-run button.

    Query params:
      ticker   — limit to one company (Bloomberg format, e.g. "ALLY US"),
                 defaults to all companies
      dry_run  — if true, return the mismatches without touching the DB
    """
    from sqlalchemy import text as sa_text
    from apps.api.models import ExtractionProfile, Document

    where_ticker = ""
    params: dict = {}
    if ticker:
        co_q = await db.execute(select(Company.id).where(Company.ticker == ticker.upper()))
        row = co_q.first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
        where_ticker = "AND ep.company_id = :cid"
        params["cid"] = str(row[0])

    # Find all mismatched rows: profile's period_label doesn't match the
    # document it points at. IS DISTINCT FROM handles NULLs safely.
    preview_sql = sa_text(f"""
        SELECT
            ep.id            AS profile_id,
            ep.document_id   AS document_id,
            ep.period_label  AS profile_period,
            d.period_label   AS document_period,
            c.ticker         AS ticker
        FROM extraction_profiles ep
        JOIN documents d ON ep.document_id = d.id
        LEFT JOIN companies c ON ep.company_id = c.id
        WHERE ep.period_label IS DISTINCT FROM d.period_label
          {where_ticker}
        ORDER BY c.ticker, d.period_label
    """)
    preview = await db.execute(preview_sql, params)
    mismatches = [
        {
            "profile_id":      str(r.profile_id),
            "document_id":     str(r.document_id),
            "ticker":          r.ticker,
            "profile_period":  r.profile_period,
            "document_period": r.document_period,
        }
        for r in preview.all()
    ]

    if dry_run:
        # Group by (from, to) for a concise summary
        from collections import Counter
        counts = Counter(
            (m["profile_period"], m["document_period"]) for m in mismatches
        )
        return {
            "ticker": ticker,
            "dry_run": True,
            "mismatches_found": len(mismatches),
            "summary": [
                {"from": f, "to": t, "rows": n}
                for (f, t), n in sorted(counts.items(), key=lambda x: -x[1])
            ],
            "examples": mismatches[:20],
        }

    # Real run: UPDATE ... FROM ... WHERE IS DISTINCT FROM
    update_sql = sa_text(f"""
        UPDATE extraction_profiles ep
        SET period_label = d.period_label
        FROM documents d
        WHERE ep.document_id = d.id
          AND ep.period_label IS DISTINCT FROM d.period_label
          {where_ticker}
    """)
    result = await db.execute(update_sql, params)
    await db.commit()
    return {
        "ticker": ticker,
        "dry_run": False,
        "rows_updated": result.rowcount or 0,
        "mismatches_before": len(mismatches),
    }


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
