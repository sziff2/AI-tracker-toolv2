"""
Agent Orchestrator — the brain of the agent pipeline.

Handles:
  - Dependency DAG resolution and layer-by-layer execution
  - Parallel execution within each layer
  - Context Contract injection into every agent
  - Caching, graceful degradation, budget enforcement
  - Four entry points for different trigger contexts

Entry points:
  run_document_pipeline()   — triggered by "Run Analysis" button
  run_macro_refresh()       — monthly scheduled, no company context
  run_portfolio_pipeline()  — triggered on weight changes
  run_agent_on_demand()     — analyst triggers a specific agent
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import AgentResult
from agents.registry import AgentRegistry
from apps.api.database import AsyncSessionLocal
from apps.api.models import AgentOutput, PipelineRun, ResearchOutput
from configs.settings import settings
from services.budget_guard import BudgetGuard, BudgetExceeded
from services.context_builder import build_agent_context
from services.llm_client import set_budget_guard

logger = logging.getLogger(__name__)

# Only financial_analyst causes a full abort on failure.
# All other agents degrade gracefully.
CRITICAL_AGENTS = {"financial_analyst"}


@dataclass
class PipelineRunResult:
    """Returned by any orchestrator entry point after completion."""
    pipeline_run_id: str
    status: str  # completed | failed | aborted | budget_exceeded | phase_a_incomplete
    agents_completed: list[str] = field(default_factory=list)
    agents_failed: list[str] = field(default_factory=list)
    agents_skipped: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    duration_ms: int = 0
    overall_qc_score: float | None = None
    error_message: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """
    Resolves the dependency DAG and executes agents in the correct order.
    """

    def __init__(self):
        self.cache_enabled = getattr(settings, "agent_cache_enabled", False)

    # ─────────────────────────────────────────────────────────────
    # Entry Point 1 — Document pipeline ("Run Analysis")
    # ─────────────────────────────────────────────────────────────

    async def run_document_pipeline(
        self,
        company_id: str,
        period_label: str,
        agent_ids: list[str] | None = None,
        db: AsyncSession | None = None,
        force_rerun: bool = False,
    ) -> PipelineRunResult:
        """
        Triggered when analyst clicks "Run Analysis" on a period.
        Phase A (extraction) must already be complete.

        If `force_rerun=False` and a completed pipeline run already exists
        for this (company, period), return its id immediately instead of
        re-running all agents. Pass `force_rerun=True` to bypass the cache.
        """
        t0 = time.time()
        own_session = db is None
        if own_session:
            db = AsyncSessionLocal()

        if not await self._is_phase_a_complete(db, company_id, period_label):
            if own_session:
                await db.close()
            return PipelineRunResult(
                pipeline_run_id="",
                status="phase_a_incomplete",
                error_message=(
                    "Document extraction has not completed for this period. "
                    "Process the document first, then run analysis."
                ),
            )

        # Pre-flight gates (Data Completeness + Source Coverage).
        # Default mode = "warn": gates run, reports attach to pipeline_run.
        # warnings, pipeline still proceeds. Flip settings.completeness_gate_mode
        # to "halt" once false-positive rate is known (2-week validation).
        gate_halt, gate_warnings = await self._run_preflight_gates(
            db, company_id, period_label
        )
        if gate_halt is not None:
            # Halt path — create an audit pipeline_run row so the halt is
            # visible in the UI + reports, then return early.
            pr = await self._create_pipeline_run(
                db, company_id, period_label, trigger="manual"
            )
            await self._persist_pipeline_run_warnings(db, pr.id, gate_warnings, status="halted_incomplete")
            if own_session:
                await db.close()
            gate_halt.pipeline_run_id = str(pr.id)
            return gate_halt

        # Cache short-circuit: reuse a recent completed run unless the
        # analyst explicitly forced a re-run.
        if not force_rerun:
            cached = await self._find_cached_run(db, company_id, period_label)
            if cached is not None:
                logger.info(
                    "Returning cached pipeline run %s for %s %s (use force_rerun=true to override)",
                    cached.id, company_id, period_label,
                )
                if own_session:
                    await db.close()
                return PipelineRunResult(
                    pipeline_run_id=str(cached.id),
                    status="cached",
                    total_cost_usd=float(cached.total_cost_usd or 0),
                    duration_ms=cached.duration_ms or 0,
                    overall_qc_score=(
                        float(cached.overall_qc_score)
                        if cached.overall_qc_score is not None else None
                    ),
                )

        pipeline_run = await self._create_pipeline_run(
            db, company_id, period_label, trigger="manual"
        )
        run_id = str(pipeline_run.id)
        logger.info("Document pipeline %s started: %s %s", run_id, company_id, period_label)

        # Attach any gate warnings (warn-mode path) to this pipeline_run before
        # agents start — visible in the UI timeline even if the run is still
        # executing.
        if gate_warnings:
            await self._persist_pipeline_run_warnings(db, pipeline_run.id, gate_warnings)

        return await self._execute_pipeline(
            db=db,
            pipeline_run=pipeline_run,
            company_id=company_id,
            period_label=period_label,
            agent_ids=agent_ids,
            own_session=own_session,
            t0=t0,
        )

    # ─────────────────────────────────────────────────────────────
    # Entry Point 2 — Macro refresh (monthly scheduled)
    # ─────────────────────────────────────────────────────────────

    async def run_macro_refresh(
        self,
        db: AsyncSession | None = None,
    ) -> PipelineRunResult:
        """Runs all macro agents (no company context needed)."""
        t0 = time.time()
        own_session = db is None
        if own_session:
            db = AsyncSessionLocal()

        from agents.base import AgentTier
        macro_agents = AgentRegistry.get_by_tier(AgentTier.MACRO)
        macro_agent_ids = [a.agent_id for a in macro_agents]

        if not macro_agent_ids:
            logger.info("No macro agents registered yet — skipping macro refresh")
            if own_session:
                await db.close()
            return PipelineRunResult(
                pipeline_run_id="",
                status="completed",
                error_message="No macro agents registered.",
            )

        pipeline_run = await self._create_pipeline_run(
            db, company_id=None, period_label=None, trigger="scheduled_macro"
        )

        from services.context_builder import build_context_contract
        inputs = {
            "context_contract": await build_context_contract(db),
            "pipeline_run_id": str(pipeline_run.id),
        }

        return await self._execute_pipeline(
            db=db,
            pipeline_run=pipeline_run,
            company_id=None,
            period_label=None,
            agent_ids=macro_agent_ids,
            own_session=own_session,
            t0=t0,
            base_inputs=inputs,
        )

    # ─────────────────────────────────────────────────────────────
    # Entry Point 3 — Portfolio pipeline
    # ─────────────────────────────────────────────────────────────

    async def run_portfolio_pipeline(
        self,
        portfolio_id: str,
        db: AsyncSession | None = None,
    ) -> PipelineRunResult:
        """Runs portfolio risk agents with the full portfolio context."""
        t0 = time.time()
        own_session = db is None
        if own_session:
            db = AsyncSessionLocal()

        from agents.base import AgentTier
        from services.context_builder import build_context_contract

        portfolio_agents = AgentRegistry.get_by_tier(AgentTier.PORTFOLIO)
        portfolio_agent_ids = [a.agent_id for a in portfolio_agents]

        if not portfolio_agent_ids:
            logger.info("No portfolio agents registered yet")
            if own_session:
                await db.close()
            return PipelineRunResult(
                pipeline_run_id="",
                status="completed",
                error_message="No portfolio agents registered.",
            )

        pipeline_run = await self._create_pipeline_run(
            db, company_id=None, period_label=None, trigger="portfolio"
        )

        inputs = {
            "portfolio_id": portfolio_id,
            "context_contract": await build_context_contract(db),
            "pipeline_run_id": str(pipeline_run.id),
        }

        return await self._execute_pipeline(
            db=db,
            pipeline_run=pipeline_run,
            company_id=None,
            period_label=None,
            agent_ids=portfolio_agent_ids,
            own_session=own_session,
            t0=t0,
            base_inputs=inputs,
        )

    # ─────────────────────────────────────────────────────────────
    # Entry Point 4 — On-demand single agent (Deep Dive buttons)
    # ─────────────────────────────────────────────────────────────

    async def run_agent_on_demand(
        self,
        agent_id: str,
        company_id: str | None = None,
        period_label: str | None = None,
        db: AsyncSession | None = None,
    ) -> PipelineRunResult:
        """
        Analyst manually triggers a specific agent.
        Auto-resolves upstream dependencies.
        """
        t0 = time.time()
        own_session = db is None
        if own_session:
            db = AsyncSessionLocal()

        full_chain = AgentRegistry.get_execution_order([agent_id])
        agent_ids_to_run = [a.agent_id for a in full_chain]

        logger.info(
            "On-demand: %s -> running chain: %s",
            agent_id, agent_ids_to_run,
        )

        pipeline_run = await self._create_pipeline_run(
            db, company_id=company_id, period_label=period_label,
            trigger=f"on_demand_{agent_id}",
        )

        return await self._execute_pipeline(
            db=db,
            pipeline_run=pipeline_run,
            company_id=company_id,
            period_label=period_label,
            agent_ids=agent_ids_to_run,
            own_session=own_session,
            t0=t0,
        )

    # ─────────────────────────────────────────────────────────────
    # Core execution engine (shared by all entry points)
    # ─────────────────────────────────────────────────────────────

    async def _execute_pipeline(
        self,
        db: AsyncSession,
        pipeline_run: PipelineRun,
        company_id: str | None,
        period_label: str | None,
        agent_ids: list[str] | None,
        own_session: bool,
        t0: float,
        base_inputs: dict | None = None,
    ) -> PipelineRunResult:
        run_id = str(pipeline_run.id)
        result = PipelineRunResult(pipeline_run_id=run_id, status="running")
        all_outputs: dict[str, Any] = {}
        execution_log: list[dict] = []

        try:
            if base_inputs is not None:
                inputs = base_inputs
            else:
                inputs = await build_agent_context(db, company_id, period_label)
            inputs["pipeline_run_id"] = run_id

            # Ensure agents are discovered (may be running in Celery worker
            # where FastAPI lifespan autodiscover hasn't run)
            if not AgentRegistry.get_all():
                AgentRegistry.autodiscover()

            execution_order = AgentRegistry.get_execution_order(agent_ids)
            if not execution_order:
                logger.warning("No agents registered after autodiscover()")
                result.status = "completed"
                return result

            layers = self._group_by_layer(execution_order)
            logger.info(
                "Pipeline %s: %d agents, %d layers: %s",
                run_id, len(execution_order), len(layers),
                {layer: [a.agent_id for a in agents] for layer, agents in layers},
            )

            budget = BudgetGuard(settings.agent_pipeline_budget_usd)
            set_budget_guard(budget)

            async with asyncio.timeout(settings.agent_pipeline_timeout_seconds):
                for layer_num, layer_agents in layers:
                    layer_results = await self._run_layer(
                        layer_num=layer_num,
                        layer_agents=layer_agents,
                        inputs=inputs,
                        db=db,
                        pipeline_run_id=run_id,
                        execution_log=execution_log,
                        result=result,
                        budget=budget,
                    )

                    # Abort on critical agent failure
                    failed_critical = [
                        aid for aid, r in layer_results.items()
                        if r.status == "failed" and aid in CRITICAL_AGENTS
                    ]
                    if failed_critical:
                        msg = f"Critical agent(s) failed: {failed_critical}"
                        logger.error("Pipeline %s aborting -- %s", run_id, msg)
                        result.status = "aborted"
                        result.error_message = msg
                        break

                    # Merge outputs into shared inputs for downstream agents
                    for agent_id, agent_result in layer_results.items():
                        if agent_result.output is not None:
                            all_outputs[agent_id] = agent_result.output
                            inputs[agent_id] = agent_result.output
                            result.outputs[agent_id] = agent_result.output

                    # Always give QC the full picture
                    inputs["all_outputs"] = all_outputs

                    # Tier 4.4 — resolve any citations agents emitted, so QC
                    # can score them. Runs deterministically against the DB;
                    # safe to call every layer (cheap — all in-process). The
                    # pilot is bear_case; other agents that start emitting
                    # `sources` arrays get the same treatment automatically.
                    inputs["citation_reports"] = await self._compute_citation_reports(
                        all_outputs, inputs.get("company_id"), period_label,
                    )

                    if budget.is_exceeded():
                        result.status = "budget_exceeded"
                        break

            if result.status == "running":
                result.status = "completed"

        except asyncio.TimeoutError:
            result.status = "failed"
            result.error_message = f"Timed out after {settings.agent_pipeline_timeout_seconds}s"

        except BudgetExceeded as e:
            result.status = "budget_exceeded"
            result.error_message = str(e)

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.exception("Pipeline %s failed: %s", run_id, e)

        finally:
            set_budget_guard(None)
            result.duration_ms = int((time.time() - t0) * 1000)

            qc_output = all_outputs.get("quality_control", {})
            if isinstance(qc_output, dict):
                result.overall_qc_score = qc_output.get("overall_score")

            await self._finalise_pipeline_run(db, pipeline_run, result, execution_log)
            if own_session:
                await db.close()

        logger.info(
            "Pipeline %s %s | %dms | $%.4f | %d ok %d failed",
            run_id, result.status, result.duration_ms,
            result.total_cost_usd,
            len(result.agents_completed), len(result.agents_failed),
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Layer execution
    # ─────────────────────────────────────────────────────────────

    async def _run_layer(
        self,
        layer_num: int,
        layer_agents: list,
        inputs: dict,
        db: AsyncSession,
        pipeline_run_id: str,
        execution_log: list,
        result: PipelineRunResult,
        budget: BudgetGuard,
    ) -> dict[str, AgentResult]:
        logger.info("Layer %d: %s", layer_num, [a.agent_id for a in layer_agents])

        # Apply per-agent model overrides from settings
        for agent_cls in layer_agents:
            override = settings.agent_model_overrides.get(agent_cls.agent_id)
            if override:
                agent_cls.model_override = override

        # Cache check
        to_run = []
        cached_results = {}
        for agent_cls in layer_agents:
            if self.cache_enabled:
                cached = await self._check_cache(db, agent_cls.agent_id, inputs)
                if cached:
                    logger.info("Cache hit: %s", agent_cls.agent_id)
                    cached_results[agent_cls.agent_id] = cached
                    result.agents_completed.append(agent_cls.agent_id)
                    continue
            to_run.append(agent_cls)

        if not to_run:
            return cached_results

        async def run_one(agent_cls) -> tuple[str, AgentResult]:
            agent = agent_cls()
            t = time.time()
            try:
                return agent_cls.agent_id, await agent.run(inputs)
            except Exception as e:
                return agent_cls.agent_id, AgentResult(
                    agent_id=agent_cls.agent_id,
                    status="failed",
                    error=str(e),
                    duration_ms=int((time.time() - t) * 1000),
                )

        pairs = await asyncio.gather(*[run_one(a) for a in to_run])

        layer_results = dict(cached_results)
        for agent_id, agent_result in pairs:
            layer_results[agent_id] = agent_result
            await self._persist_agent_output(db, agent_result, inputs, pipeline_run_id)

            execution_log.append({
                "agent_id":    agent_id,
                "status":      agent_result.status,
                "duration_ms": agent_result.duration_ms,
                "cost_usd":    agent_result.cost_usd,
                "error":       agent_result.error,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            })

            if agent_result.status in ("completed", "degraded"):
                result.agents_completed.append(agent_id)
            elif agent_result.status == "failed":
                result.agents_failed.append(agent_id)
            elif agent_result.status == "skipped":
                result.agents_skipped.append(agent_id)

            result.total_cost_usd += agent_result.cost_usd
            result.total_input_tokens += agent_result.input_tokens
            result.total_output_tokens += agent_result.output_tokens
            if agent_result.status != "skipped":
                result.total_llm_calls += 1

            budget.add_cost(agent_result.cost_usd)

        return layer_results

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    async def _find_cached_run(
        self, db: AsyncSession, company_id: str, period_label: str
    ):
        """Return the most recent completed PipelineRun for this
        (company, period) if one exists, else None. Used by
        run_document_pipeline to short-circuit repeat requests."""
        try:
            from apps.api.models import PipelineRun
            q = await db.execute(
                select(PipelineRun)
                .where(PipelineRun.company_id == company_id)
                .where(PipelineRun.period_label == period_label)
                .where(PipelineRun.status == "completed")
                .order_by(desc(PipelineRun.completed_at))
                .limit(1)
            )
            return q.scalar_one_or_none()
        except Exception as e:
            logger.warning("Cache lookup failed: %s", str(e)[:200])
            return None

    async def _compute_citation_reports(
        self,
        all_outputs: dict,
        company_id: str | None,
        period_label: str,
    ) -> dict:
        """Tier 4.4 — resolve every agent's sources array against the DB.

        Runs once per layer with the current set of completed agent
        outputs. QC reads the resulting dict. Safe when no agent has
        emitted sources yet (returns empty dict) and when company_id
        is missing (skips with a warning)."""
        if not company_id:
            return {}
        try:
            from services.citation_resolver import resolve_citations
            from apps.api.database import AsyncSessionLocal
            reports: dict = {}
            async with AsyncSessionLocal() as s:
                for agent_id, output in all_outputs.items():
                    if not isinstance(output, dict):
                        continue
                    sources = output.get("sources")
                    if not sources:
                        continue
                    report = await resolve_citations(
                        s, company_id, period_label, sources,
                    )
                    reports[agent_id] = report.to_dict()
            return reports
        except Exception as exc:  # noqa: BLE001
            logger.warning("citation report compute failed: %s", str(exc)[:200])
            return {}

    async def _is_phase_a_complete(
        self, db: AsyncSession, company_id: str, period_label: str
    ) -> bool:
        """Phase A is complete only when EVERY document in the period has
        been parsed AND had extraction run against it.

        Parsing alone isn't enough — the batch pipeline parses first,
        then runs section-aware / two-pass extraction, which can take
        1-3 minutes on a dense 10-Q. During that window parsing_status
        reads 'completed' but metrics haven't been written yet. Firing
        agents in that window hands them an empty extraction dataset.

        Two extraction-completion signals are accepted, either suffices:
          • ExtractionProfile row present (clean shutdown)
          • At least one ExtractedMetric row present (worker crashed
            after metric INSERT but before profile UPSERT — Sanofi
            2026_Q1 hit this on the 2026-04-27 OOM restart)

        Intersect with current-period document IDs so orphan
        metric.document_id values from deleted/re-tagged docs don't
        inflate the count past `total_docs`."""
        try:
            from apps.api.models import Document, ExtractionProfile, ExtractedMetric
            from sqlalchemy import func
            # All documents in the period and how many are parsed.
            doc_rows_q = await db.execute(
                select(Document.id, Document.parsing_status)
                .where(Document.company_id == company_id)
                .where(Document.period_label == period_label)
            )
            doc_rows = doc_rows_q.all()
            total_docs = len(doc_rows)
            if total_docs == 0:
                return False
            total_parsed = sum(1 for _id, ps in doc_rows if ps == "completed")
            if total_parsed != total_docs:
                return False

            docs_in_period = {did for did, _ps in doc_rows}

            prof_doc_ids_q = await db.execute(
                select(func.distinct(ExtractionProfile.document_id))
                .where(ExtractionProfile.company_id == company_id)
                .where(ExtractionProfile.period_label == period_label)
            )
            metric_doc_ids_q = await db.execute(
                select(func.distinct(ExtractedMetric.document_id))
                .where(ExtractedMetric.company_id == company_id)
                .where(ExtractedMetric.period_label == period_label)
                .where(ExtractedMetric.document_id.is_not(None))
            )
            extracted_ids = {row[0] for row in prof_doc_ids_q.all() if row[0]}
            extracted_ids |= {row[0] for row in metric_doc_ids_q.all() if row[0]}
            extracted_ids &= docs_in_period

            return len(extracted_ids) == total_docs
        except Exception as e:
            logger.warning("Phase A check failed: %s", str(e)[:200])
            return False

    async def _run_preflight_gates(
        self, db: AsyncSession, company_id: str, period_label: str,
    ) -> tuple[Optional[PipelineRunResult], dict]:
        """Run both pre-flight gates (Data Completeness + Source Coverage).

        Returns a tuple of (halt_result, warnings_dict):
          - halt_result: a PipelineRunResult if we're in "halt" mode AND a
            gate says HALT_INCOMPLETE. None otherwise.
          - warnings_dict: structured reports from both gates, always
            populated so callers can persist to pipeline_run.warnings.

        Gates never raise — a query failure logs + skips the gate rather
        than blocking the pipeline. Correctness tooling must not itself be
        a source of outages."""
        from services.completeness_gate import (
            compute_completeness, compute_source_coverage, HALT_INCOMPLETE,
        )

        warnings: dict = {}
        halt_reasons: list[str] = []

        try:
            comp_report = await compute_completeness(db, company_id, period_label)
            warnings["completeness"] = comp_report.to_dict()
            if comp_report.status == HALT_INCOMPLETE:
                halt_reasons.append(f"completeness: {comp_report.reason}")
        except Exception as exc:
            logger.warning("Completeness gate failed (skipping): %s", str(exc)[:200])
            warnings["completeness_error"] = str(exc)[:500]

        try:
            cov_report = await compute_source_coverage(db, company_id, period_label)
            warnings["source_coverage"] = cov_report.to_dict()
            if cov_report.status == HALT_INCOMPLETE:
                halt_reasons.append(f"source_coverage: {cov_report.reason}")
        except Exception as exc:
            logger.warning("Source coverage gate failed (skipping): %s", str(exc)[:200])
            warnings["source_coverage_error"] = str(exc)[:500]

        # ── Third gate: reconciliation (Tier 1.2, Sprint C) ──
        # Validates extracted metrics against universal + sector rules,
        # surfaces QoQ anomalies, cross-source disagreements, structural
        # mismatches. Halt only if settings.reconciliation_mode == "halt"
        # AND at least one critical issue is raised.
        try:
            from services.reconciliation import compute_reconciliation
            rec_report = await compute_reconciliation(db, company_id, period_label)
            warnings["reconciliation"] = rec_report.to_dict()
            if rec_report.status == HALT_INCOMPLETE:
                rec_mode = getattr(settings, "reconciliation_mode", "warn")
                if rec_mode == "halt":
                    halt_reasons.append(f"reconciliation: {rec_report.reason}")
        except Exception as exc:
            logger.warning("Reconciliation gate failed (skipping): %s", str(exc)[:200])
            warnings["reconciliation_error"] = str(exc)[:500]

        mode = getattr(settings, "completeness_gate_mode", "warn")
        if halt_reasons and mode == "halt":
            logger.warning(
                "[GATE] Halting pipeline for %s %s — %s",
                company_id, period_label, "; ".join(halt_reasons),
            )
            return (
                PipelineRunResult(
                    pipeline_run_id="",  # caller fills this in after creating the row
                    status="halted_incomplete",
                    error_message=(
                        "Pre-flight gates halted this analysis. "
                        + "; ".join(halt_reasons)
                    ),
                ),
                warnings,
            )
        elif halt_reasons:
            # warn mode — log but don't halt
            logger.info(
                "[GATE] Warnings (non-halting) for %s %s — %s",
                company_id, period_label, "; ".join(halt_reasons),
            )

        return (None, warnings)

    async def _persist_pipeline_run_warnings(
        self,
        db: AsyncSession,
        pipeline_run_id,
        warnings: dict,
        status: Optional[str] = None,
    ) -> None:
        """Write the gate warnings JSONB to pipeline_runs.warnings. If
        status is provided, also update pipeline_runs.status (used on the
        halt path to mark the row as halted_incomplete immediately)."""
        from sqlalchemy import update
        from apps.api.models import PipelineRun
        try:
            values: dict = {"warnings": warnings}
            if status:
                values["status"] = status
                values["completed_at"] = datetime.now(timezone.utc)
            await db.execute(
                update(PipelineRun)
                .where(PipelineRun.id == pipeline_run_id)
                .values(**values)
            )
            await db.commit()
        except Exception as exc:
            logger.warning(
                "Failed to persist warnings to pipeline_run %s: %s",
                pipeline_run_id, str(exc)[:200],
            )

    def _group_by_layer(self, agents: list) -> list[tuple[int, list]]:
        """Group agents into execution layers based on dependency depth.

        Agents with no dependencies (or whose dependencies aren't in this
        run) go in layer 0. Agents depending on layer-0 agents go in
        layer 1, and so on. This ensures upstream outputs are available
        before downstream agents run.
        """
        agent_ids = {a.agent_id for a in agents}
        agent_map = {a.agent_id: a for a in agents}
        depth: dict[str, int] = {}

        def _depth(aid: str) -> int:
            if aid in depth:
                return depth[aid]
            cls = agent_map.get(aid)
            if not cls:
                return 0
            deps_in_run = [d for d in (cls.depends_on or []) if d in agent_ids]
            if not deps_in_run:
                depth[aid] = 0
            else:
                depth[aid] = 1 + max(_depth(d) for d in deps_in_run)
            return depth[aid]

        for a in agents:
            _depth(a.agent_id)

        layers: dict[int, list] = {}
        for a in agents:
            d = depth.get(a.agent_id, 0)
            layers.setdefault(d, []).append(a)
        return sorted(layers.items())

    async def _check_cache(
        self, db: AsyncSession, agent_id: str, inputs: dict
    ) -> AgentResult | None:
        try:
            period = inputs.get("period_label")
            q = await db.execute(
                select(AgentOutput)
                .where(AgentOutput.agent_id == agent_id)
                .where(AgentOutput.period_label == period)
                .where(AgentOutput.status == "completed")
                .order_by(desc(AgentOutput.created_at))
                .limit(1)
            )
            row = q.scalar_one_or_none()
            if not row:
                return None
            agent_cls = AgentRegistry.get(agent_id)
            ttl_hours = getattr(agent_cls, "cache_ttl_hours", 24) if agent_cls else 24
            age = datetime.now(timezone.utc) - row.created_at.replace(tzinfo=timezone.utc)
            if age.total_seconds() > ttl_hours * 3600:
                return None
            return AgentResult(
                agent_id=agent_id,
                status="completed",
                output=row.output_json,
                confidence=row.confidence or 1.0,
                cost_usd=0.0,
            )
        except Exception as e:
            logger.debug("Cache check failed for %s: %s", agent_id, e)
            return None

    async def _persist_agent_output(
        self, db, result: AgentResult, inputs: dict, pipeline_run_id: str
    ) -> None:
        try:
            import sqlalchemy
            from apps.api.models import Company
            company_id = inputs.get("company_id")
            if not company_id and inputs.get("ticker"):
                q = await db.execute(
                    sqlalchemy.select(Company).where(Company.ticker == inputs["ticker"])
                )
                c = q.scalar_one_or_none()
                company_id = str(c.id) if c else None

            row = AgentOutput(
                id=uuid.uuid4(),
                agent_id=result.agent_id,
                company_id=company_id,
                period_label=inputs.get("period_label"),
                pipeline_run_id=pipeline_run_id,
                status=result.status,
                output_json=result.output,
                confidence=result.confidence,
                qc_score=result.qc_score,
                duration_ms=result.duration_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                prompt_variant_id=result.prompt_variant_id,
                predictions_json=result.predictions or None,
                error_message=result.error,
            )
            db.add(row)
            await db.commit()
        except Exception as e:
            logger.warning(
                "Failed to persist agent output for %s: %s",
                result.agent_id, str(e)[:100],
            )

    async def _create_pipeline_run(
        self,
        db: AsyncSession,
        company_id: str | None,
        period_label: str | None,
        trigger: str = "manual",
    ) -> PipelineRun:
        import sqlalchemy
        from apps.api.models import Company
        resolved_id = company_id
        if company_id:
            try:
                q = await db.execute(
                    sqlalchemy.select(Company).where(
                        (Company.id == company_id) | (Company.ticker == company_id)
                    )
                )
                c = q.scalar_one_or_none()
                if c:
                    resolved_id = str(c.id)
            except Exception:
                pass

        run = PipelineRun(
            id=uuid.uuid4(),
            company_id=resolved_id,
            period_label=period_label,
            trigger=trigger,
            status="running",
            started_at=datetime.now(timezone.utc),
            agents_planned=len(AgentRegistry.get_all()),
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return run

    async def _finalise_pipeline_run(
        self, db, pipeline_run, result: PipelineRunResult, execution_log: list
    ) -> None:
        try:
            pipeline_run.status = result.status
            pipeline_run.completed_at = datetime.now(timezone.utc)
            pipeline_run.duration_ms = result.duration_ms
            pipeline_run.total_cost_usd = result.total_cost_usd
            pipeline_run.total_input_tokens = result.total_input_tokens
            pipeline_run.total_output_tokens = result.total_output_tokens
            pipeline_run.total_llm_calls = result.total_llm_calls
            pipeline_run.agents_completed = len(result.agents_completed)
            pipeline_run.agents_failed = len(result.agents_failed)
            pipeline_run.agents_skipped = len(result.agents_skipped)
            pipeline_run.overall_qc_score = result.overall_qc_score
            pipeline_run.error_message = result.error_message
            pipeline_run.agent_execution_log = execution_log
            await db.commit()
        except Exception as e:
            logger.warning("Failed to finalise pipeline run: %s", str(e)[:100])
