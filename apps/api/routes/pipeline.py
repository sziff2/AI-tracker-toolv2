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
    create_missing: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Reconcile extraction_profiles with documents for Phase A gating.

    Phase A joins Document.period_label against
    ExtractionProfile.period_label to decide when a period is "extracted".
    Two classes of bug break this join:

    1. Mismatched period labels — profile says "Q2 2025", document says
       "2025_Q2". Rewrites profile to match.
    2. Missing profiles — profile row doesn't exist at all, even though
       metrics were extracted (silent failure in _persist_extraction_profile
       during ingestion). Optional create_missing=true synthesises a stub
       profile for any document that has at least one ExtractedMetric row
       but no ExtractionProfile. The stub's period_label is set from the
       document; all other fields are null.

    Query params:
      ticker          — limit to one company (Bloomberg format)
      dry_run         — report without touching the DB
      create_missing  — also synthesise stub profiles for docs with
                        metrics but no profile row
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

    # Second class of bug: documents that have metrics but no
    # ExtractionProfile row at all. These orphans also break the Phase A
    # join. Find docs with ExtractedMetric rows but no matching profile.
    orphans_sql_where = "AND c.ticker = :tticker" if ticker else ""
    if ticker:
        params["tticker"] = ticker.upper()
    orphans_sql = sa_text(f"""
        SELECT DISTINCT
            d.id              AS document_id,
            d.period_label    AS document_period,
            d.document_type   AS document_type,
            c.ticker          AS ticker,
            d.company_id      AS company_id,
            (SELECT COUNT(*) FROM extracted_metrics m WHERE m.document_id = d.id) AS metric_count
        FROM documents d
        JOIN companies c ON d.company_id = c.id
        JOIN extracted_metrics em ON em.document_id = d.id
        WHERE NOT EXISTS (
            SELECT 1 FROM extraction_profiles ep WHERE ep.document_id = d.id
        )
        {orphans_sql_where}
        ORDER BY c.ticker, d.period_label
    """)
    orphan_rows = await db.execute(orphans_sql, params)
    orphans = [
        {
            "document_id":     str(r.document_id),
            "ticker":          r.ticker,
            "document_period": r.document_period,
            "document_type":   r.document_type,
            "company_id":      str(r.company_id),
            "metric_count":    int(r.metric_count or 0),
        }
        for r in orphan_rows.all()
    ]

    if dry_run:
        from collections import Counter
        counts = Counter(
            (m["profile_period"], m["document_period"]) for m in mismatches
        )
        return {
            "ticker": ticker,
            "dry_run": True,
            "mismatches_found": len(mismatches),
            "orphans_found": len(orphans),
            "mismatch_summary": [
                {"from": f, "to": t, "rows": n}
                for (f, t), n in sorted(counts.items(), key=lambda x: -x[1])
            ],
            "mismatch_examples": mismatches[:20],
            "orphan_examples": orphans[:20],
        }

    # Real run: update mismatches
    update_sql = sa_text(f"""
        UPDATE extraction_profiles ep
        SET period_label = d.period_label
        FROM documents d
        WHERE ep.document_id = d.id
          AND ep.period_label IS DISTINCT FROM d.period_label
          {where_ticker}
    """)
    update_result = await db.execute(update_sql, params)

    # Real run: optionally synthesise stubs for orphans
    stubs_created = 0
    if create_missing and orphans:
        import uuid as uuid_mod
        from apps.api.models import ExtractionProfile as EP
        for o in orphans:
            stub = EP(
                id=uuid_mod.uuid4(),
                company_id=uuid_mod.UUID(o["company_id"]),
                document_id=uuid_mod.UUID(o["document_id"]),
                period_label=o["document_period"],
                extraction_method="legacy_backfill",
                items_extracted=o["metric_count"],
            )
            db.add(stub)
            stubs_created += 1

    await db.commit()
    return {
        "ticker": ticker,
        "dry_run": False,
        "mismatches_updated": update_result.rowcount or 0,
        "orphans_found": len(orphans),
        "stubs_created": stubs_created,
        "create_missing": create_missing,
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


# ─────────────────────────────────────────────────────────────────
# Admin — Database backup (streamed JSON download)
# ─────────────────────────────────────────────────────────────────

@router.post("/admin/backup/notify")
async def trigger_backup_notification():
    """Trigger the daily backup summary → Teams notification now (via Celery worker)."""
    try:
        from apps.worker.tasks import daily_backup_report
        daily_backup_report.delay()
        return {"status": "sent_to_celery_worker"}
    except Exception as exc:
        return {"status": "celery_failed", "error": str(exc)}


@router.get("/admin/backup")
async def download_backup(db: AsyncSession = Depends(get_db)):
    """
    Stream a full database backup as a dated JSON download.

    Streams table-by-table so the first bytes arrive immediately and
    Railway's request timeout doesn't kill the connection.

    Hit this from a browser to download, or automate locally:
        curl -o "DB Backup/db_backup_$(date +%F).json" \\
             https://ai-tracker-tool-production.up.railway.app/api/v1/admin/backup
    """
    import json
    from datetime import date, datetime
    from decimal import Decimal
    from uuid import UUID
    from fastapi.responses import StreamingResponse
    from sqlalchemy import text

    TABLES = [
        "companies", "document_sections", "documents", "extracted_metrics",
        "extraction_profiles", "harvested_documents", "harvester_sources",
        "kpi_actuals", "llm_usage_log", "portfolio_holdings", "portfolios",
        "price_records", "processing_jobs", "research_outputs", "review_queue",
        "scenario_snapshots", "valuation_scenarios",
    ]

    def _default(obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")
        raise TypeError(f"Not JSON serializable: {type(obj).__name__}")

    async def _stream():
        """Yield JSON incrementally — one table at a time."""
        yield "{"
        first = True
        for table in TABLES:
            if not first:
                yield ","
            first = False
            yield f"\n{json.dumps(table)}: "
            try:
                result = await db.execute(text(f'SELECT * FROM "{table}"'))
                rows = result.mappings().all()
                yield json.dumps(
                    [dict(row) for row in rows],
                    default=_default,
                    ensure_ascii=False,
                )
            except Exception:
                yield "[]"
        yield "\n}"

    today = date.today().isoformat()
    return StreamingResponse(
        _stream(),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="db_backup_{today}.json"'},
    )


@router.get("/admin/debug-series")
async def admin_debug_series(
    ticker: str,
    window: int = 60,
    db: AsyncSession = Depends(get_db),
):
    """Dump raw USD monthly price series + log returns for one holding.
    Used to diagnose why certain correlations come out suspiciously low."""
    from sqlalchemy import text as sa_text
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns,
    )
    rs = await db.execute(sa_text("SELECT id FROM companies WHERE ticker = :t"), {"t": ticker})
    cid = rs.scalar_one_or_none()
    if cid is None:
        return {"error": f"No company for ticker {ticker}"}
    fx = await _load_fx_by_month(db)
    series = await _load_monthly_series_usd(db, cid, window, fx)
    rets = _log_returns(series)
    # Also dump the raw price_records rows for direct inspection.
    raw = await db.execute(sa_text("""
        SELECT price, currency, price_date, source
          FROM price_records WHERE company_id = :cid
         ORDER BY price_date ASC
    """), {"cid": str(cid)})
    raw_rows = [
        {"price": float(r.price), "ccy": r.currency,
         "date": r.price_date.date().isoformat() if r.price_date else None,
         "src": r.source}
        for r in raw
    ]
    return {
        "ticker": ticker,
        "n_raw_rows": len(raw_rows),
        "n_monthly": len(series),
        "n_returns": len(rets),
        "usd_monthly": [
            {"ym": f"{y:04d}-{m:02d}", "usd_price": p}
            for (y, m), p in sorted(series.items())
        ],
        "log_returns": [
            {"ym": f"{y:04d}-{m:02d}", "ret": r}
            for (y, m), r in sorted(rets.items())
        ],
        "raw_rows_sample": raw_rows[:5] + (raw_rows[-5:] if len(raw_rows) > 10 else []),
        "raw_rows": raw_rows,
    }


@router.get("/admin/benchmark-check")
async def admin_benchmark_check(
    portfolio_id: str,
    benchmarks: str = "SPY,URTH",
    window: int = 60,
    db: AsyncSession = Depends(get_db),
):
    """One-off sanity check: fetch benchmark tickers live from Yahoo,
    compute their monthly log-return vol, and correlate them against
    every holding in the portfolio. Does NOT persist anything.

    Query:
      portfolio_id: UUID
      benchmarks:   comma-separated Yahoo tickers (default SPY,URTH)
      window:       lookback months (default 60)
    """
    import math, httpx
    from datetime import date, datetime, timezone, timedelta
    from sqlalchemy import text as sa_text
    from services.portfolio_analytics import (
        _load_fx_by_month, _load_monthly_series_usd, _log_returns, _corr, _stdev,
    )
    from scripts.backfill_prices import _fetch_yahoo_history, _month_end

    end_date = date.today().replace(day=1) - timedelta(days=1)
    start_date = _month_end(end_date - timedelta(days=int(window * 31) + 62))

    # 1. Fetch each benchmark from Yahoo and build its USD monthly return series.
    bench_tickers = [b.strip() for b in benchmarks.split(",") if b.strip()]
    bench_returns: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for yt in bench_tickers:
            rows = await _fetch_yahoo_history(client, yt, start_date, end_date)
            series = {(r[0].year, r[0].month): r[1] for r in rows}
            rets = _log_returns(series)
            vols = list(rets.values())
            bench_returns[yt] = {
                "series": rets,
                "n_months": len(vols),
                "monthly_vol": _stdev(vols) if vols else None,
                "annualised_vol": (_stdev(vols) * math.sqrt(12)) if vols else None,
            }

    # 2. Holdings in the portfolio.
    rs = await db.execute(sa_text("""
        SELECT c.id AS cid, c.ticker
          FROM portfolio_holdings h
          JOIN companies c ON c.id = h.company_id
         WHERE h.portfolio_id = :pid
         ORDER BY h.weight DESC NULLS LAST
    """), {"pid": portfolio_id})
    holdings = [(row.cid, row.ticker) for row in rs]

    fx = await _load_fx_by_month(db)

    # 3. For each holding, compute its USD monthly returns, then correlate.
    per_holding = []
    corr_to_bench = {yt: [] for yt in bench_tickers}
    for cid, ticker in holdings:
        series = await _load_monthly_series_usd(db, cid, window, fx)
        rets = _log_returns(series)
        row = {"ticker": ticker, "months": len(rets), "corr": {}}
        if len(rets) >= 12:
            for yt in bench_tickers:
                common = sorted(set(rets) & set(bench_returns[yt]["series"]))
                if len(common) >= 12:
                    x = [rets[m] for m in common]
                    y = [bench_returns[yt]["series"][m] for m in common]
                    c = _corr(x, y)
                    row["corr"][yt] = round(c, 4)
                    corr_to_bench[yt].append(c)
                else:
                    row["corr"][yt] = None
        per_holding.append(row)

    # 4. Benchmark cross-correlations (e.g. SPY vs URTH).
    cross = {}
    for i, a in enumerate(bench_tickers):
        for b in bench_tickers[i + 1:]:
            common = sorted(set(bench_returns[a]["series"]) & set(bench_returns[b]["series"]))
            if len(common) >= 12:
                x = [bench_returns[a]["series"][m] for m in common]
                y = [bench_returns[b]["series"][m] for m in common]
                cross[f"{a}__{b}"] = round(_corr(x, y), 4)

    def _stats(vals):
        vs = [v for v in vals if v is not None]
        if not vs:
            return None
        return {"n": len(vs), "min": round(min(vs), 4), "max": round(max(vs), 4),
                "mean": round(sum(vs) / len(vs), 4)}

    return {
        "portfolio_id": portfolio_id,
        "window_months": window,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {
            yt: {
                "n_months": bench_returns[yt]["n_months"],
                "annualised_vol": round(bench_returns[yt]["annualised_vol"], 4)
                    if bench_returns[yt]["annualised_vol"] else None,
            }
            for yt in bench_tickers
        },
        "benchmark_cross_corr": cross,
        "holdings_vs_benchmark_summary": {
            yt: _stats(corr_to_bench[yt]) for yt in bench_tickers
        },
        "per_holding": per_holding,
    }


@router.post("/admin/backfill-prices")
async def admin_backfill_prices(
    ticker: str | None = None,
    years: float = 5.0,
    apply: bool = False,
    clean: bool = False,
):
    """Backfill monthly EOM price history + FX rates from Yahoo Finance.

    Query params:
      - ticker: optional Bloomberg ticker to scope to a single company
      - years:  lookback window (default 5.0)
      - apply:  false → dry-run (no writes); true → commit to DB
      - clean:  true → delete existing source='backfill' rows first.
                Use this after fixing bucketing bugs to overwrite bad data.
    """
    from scripts.backfill_prices import run_backfill
    try:
        summary = await run_backfill(
            ticker=ticker, years=years, apply=apply, clean=clean,
        )
        return {"status": "ok", **summary}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
