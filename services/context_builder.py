"""
Context Builder — implements context fatigue reduction.

Principles:
  1. Never pass information the model does not need right now
  2. Use layered summaries instead of raw data
  3. Compress context into structured key-value facts
  4. Each pipeline step gets minimum required context

For Phase 1 agents, use build_agent_context() — the unified
orchestrator-facing function that returns everything an agent
pipeline needs in a single dict.
"""

import json
import logging
import re
from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Company, Document, ExtractedMetric, ExtractionProfile, ThesisVersion,
    EventAssessment, ResearchOutput, TrackedKPI, KPIScore,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Period arithmetic — local to avoid importing from deprecated
# Shared context helpers feeding the agent pipeline
# ─────────────────────────────────────────────────────────────────

def _previous_period(period: str) -> str:
    """
    Return the immediately prior period label.
    2025_Q1 → 2024_Q4,  2025_Q2 → 2025_Q1, etc.
    FY is treated as Q4 throughout the codebase.
    """
    match = re.match(r"(\d{4})_Q(\d)", period)
    if not match:
        return period
    year, quarter = int(match.group(1)), int(match.group(2))
    if quarter == 1:
        return f"{year - 1}_Q4"
    return f"{year}_Q{quarter - 1}"


# ─────────────────────────────────────────────────────────────────
# Unified orchestrator entry point (Phase 1)
# ─────────────────────────────────────────────────────────────────

async def build_agent_context(
    db: AsyncSession,
    company_id,
    period: str,
) -> dict:
    """
    Build the full context dict passed to Phase 1 agents.

    Call this once per pipeline run. Pass the result as the inputs
    dict to each agent's run() method.

    Returns company identity, thesis, metrics, guidance, tracked KPIs,
    enriched extraction context, and the active context contract.
    """
    # Parallelise all independent queries using separate async sessions.
    # SQLAlchemy AsyncSession serialises within a single session, so each
    # parallel query needs its own session.
    import asyncio
    from apps.api.database import AsyncSessionLocal

    async def _with_session(fn, *args, **kwargs):
        async with AsyncSessionLocal() as s:
            return await fn(s, *args, **kwargs)

    prior_period_label = _previous_period(period)

    (
        company_meta,
        thesis,
        kpis,
        guidance,
        prior_guidance,
        prior_period,
        tracked_kpis,
        extraction_ctx,
        context_contract,
        transcript_analysis,
        presentation_analysis,
        annual_report_analysis,
    ) = await asyncio.gather(
        _with_session(_build_company_meta, company_id),
        _with_session(build_thesis_context, company_id),
        _with_session(build_kpi_summary, company_id, period),
        _with_session(build_guidance_summary, company_id, period),
        _with_session(build_guidance_summary, company_id, prior_period_label),
        _with_session(build_prior_period_context, company_id, period),
        _with_session(build_tracked_kpi_context, company_id),
        _with_session(build_extraction_context, company_id, period),
        _with_session(build_context_contract),
        _with_session(_load_document_analysis, company_id, period, "transcript_analysis"),
        _with_session(_load_document_analysis, company_id, period, "presentation_analysis"),
        _with_session(_load_document_analysis, company_id, period, "annual_report_analysis"),
    )

    # Fix 2a — only load raw transcript/presentation text as fallback
    # when pre-built analysis is missing. Saves ~40% tokens per run.
    transcript_text = ""
    presentation_text = ""
    if not transcript_analysis:
        transcript_text = await _with_session(_build_document_text, company_id, period, "transcript")
    if not presentation_analysis:
        presentation_text = await _with_session(_build_document_text, company_id, period, "presentation")

    return {
        # Identity
        "ticker":           company_meta.get("ticker", ""),
        "company_name":     company_meta.get("name", ""),
        "sector":           company_meta.get("sector", ""),
        "industry":         company_meta.get("industry", ""),
        "country":          company_meta.get("country", ""),
        "period_label":     period,
        # Thesis and metrics
        "thesis":           thesis,
        "extracted_metrics": kpis,
        "guidance":          guidance,
        "prior_guidance":    prior_guidance,
        "prior_period":      prior_period,
        "tracked_kpis":      tracked_kpis,
        # Enriched extraction outputs
        "mda_narrative":       extraction_ctx.get("mda_narrative", ""),
        "confidence_profile":  extraction_ctx.get("confidence_profile", {}),
        "disappearance_flags": extraction_ctx.get("disappearance_flags", {}),
        "non_gaap_bridge":     extraction_ctx.get("non_gaap_bridge", []),
        "segment_data":        extraction_ctx.get("segment_data"),
        "detected_period":     extraction_ctx.get("detected_period", ""),
        # Document text (raw) and pre-built analyses (from ingestion)
        "transcript_text":         transcript_text,
        "presentation_text":       presentation_text,
        "transcript_deep_dive":    transcript_analysis,
        "presentation_analysis":   presentation_analysis,
        "annual_report_analysis":  annual_report_analysis,
        # Shared macro assumptions — injected into every agent prompt
        # No agent may contradict these assumptions (enforced by QC agent)
        "context_contract":    context_contract,
    }


# ─────────────────────────────────────────────────────────────────
# Enriched extraction context
# ─────────────────────────────────────────────────────────────────

async def build_extraction_context(
    db: AsyncSession,
    company_id,
    period: str,
) -> dict:
    """
    Load enriched extraction context persisted by metric_extractor.py.
    Queries research_outputs WHERE output_type='extraction_context'.
    Returns empty dict if extraction has not run yet for this period.
    """
    try:
        q = await db.execute(
            select(ResearchOutput)
            .where(ResearchOutput.company_id == company_id)
            .where(ResearchOutput.period_label == period)
            .where(ResearchOutput.output_type == "extraction_context")
            .order_by(desc(ResearchOutput.created_at))
            .limit(1)
        )
        row = q.scalar_one_or_none()
        if row and row.content_json:
            return json.loads(row.content_json)
    except Exception as e:
        logger.warning("Failed to load extraction context for %s %s: %s",
                       company_id, period, str(e)[:100])
    return {}


# ─────────────────────────────────────────────────────────────────
# Context Contract — shared macro assumptions for all agents
# ─────────────────────────────────────────────────────────────────

async def build_context_contract(db: AsyncSession) -> dict:
    """
    Load the active ContextContract — the shared macro assumptions that
    every agent must operate within and not contradict.

    Returns a dict of macro assumptions if a contract is active.
    Returns an empty dict if no contract has been set yet (safe default —
    agents will still run, they just won't have macro constraints injected).
    """
    try:
        from apps.api.models import ContextContract
        q = await db.execute(
            select(ContextContract)
            .where(ContextContract.is_active == True)
            .order_by(ContextContract.version.desc())
            .limit(1)
        )
        contract = q.scalar_one_or_none()
        if contract and contract.macro_assumptions:
            return {
                "version":          contract.version,
                "macro_assumptions": contract.macro_assumptions,
                "analyst_overrides": contract.analyst_overrides or {},
                "authored_by":       contract.authored_by or "system",
            }
    except Exception as e:
        logger.warning("Failed to load context contract: %s", str(e)[:100])
    return {}


# ─────────────────────────────────────────────────────────────────
# Company metadata
# ─────────────────────────────────────────────────────────────────

async def _build_document_text(
    db: AsyncSession, company_id, period: str, doc_type: str,
    *, max_chars: int = 30000,
) -> str:
    """Pull parsed text for a specific document type (transcript, presentation).
    Concatenates all sections from matching documents for this company+period."""
    from apps.api.models import DocumentSection
    try:
        q = await db.execute(
            select(Document).where(
                Document.company_id == company_id,
                Document.period_label == period,
                Document.document_type == doc_type,
            ).order_by(desc(Document.created_at)).limit(1)
        )
        doc = q.scalar_one_or_none()
        if not doc:
            return ""
        sec_q = await db.execute(
            select(DocumentSection.text_content).where(
                DocumentSection.document_id == doc.id
            ).order_by(DocumentSection.page_number)
        )
        sections = sec_q.scalars().all()
        text = "\n\n".join(s for s in sections if s)
        return text[:max_chars]
    except Exception as e:
        logger.warning("Failed to load %s text for %s %s: %s",
                       doc_type, company_id, period, str(e)[:100])
        return ""


async def _load_document_analysis(
    db: AsyncSession, company_id, period: str, output_type: str,
) -> dict:
    """Load pre-built document analysis (transcript or presentation)
    stored during ingestion in research_outputs."""
    try:
        q = await db.execute(
            select(ResearchOutput)
            .where(ResearchOutput.company_id == company_id)
            .where(ResearchOutput.period_label == period)
            .where(ResearchOutput.output_type == output_type)
            .order_by(desc(ResearchOutput.created_at))
            .limit(1)
        )
        row = q.scalar_one_or_none()
        if row and row.content_json:
            return json.loads(row.content_json)
    except Exception as e:
        logger.warning("Failed to load %s for %s %s: %s",
                       output_type, company_id, period, str(e)[:100])
    return {}


async def _build_company_meta(db: AsyncSession, company_id) -> dict:
    """Query company identity fields."""
    try:
        q = await db.execute(select(Company).where(Company.id == company_id))
        company = q.scalar_one_or_none()
        if company:
            return {
                "ticker":   company.ticker or "",
                "name":     company.name or "",
                "sector":   company.sector or "",
                "industry": company.industry or "",
                "country":  company.country or "",
            }
    except Exception as e:
        logger.warning("Failed to load company meta for %s: %s",
                       company_id, str(e)[:100])
    return {}


# ─────────────────────────────────────────────────────────────────
# Individual context builders (also used by legacy pipeline)
# ─────────────────────────────────────────────────────────────────

async def build_thesis_context(db: AsyncSession, company_id) -> str:
    """Return only the core thesis — not the full version history."""
    q = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company_id,
            ThesisVersion.active == True,
        ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = q.scalar_one_or_none()
    if not thesis:
        return "No thesis on file."

    parts = [f"Core thesis: {thesis.core_thesis}"]
    if thesis.key_risks:
        parts.append(f"Key risks: {thesis.key_risks}")
    if thesis.valuation_framework:
        parts.append(f"Valuation: {thesis.valuation_framework}")
    return "\n".join(parts)


async def build_kpi_summary(
    db: AsyncSession, company_id, period: str, *, max_metrics: int = 50
) -> str:
    """
    Build a compressed KPI summary — structured key-value facts.
    Prioritises: high-confidence metrics, non-guidance, deduped.
    """
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.segment != "guidance",
        ).order_by(ExtractedMetric.confidence.desc()).limit(max_metrics)
    )
    metrics = q.scalars().all()

    if not metrics:
        return "No metrics extracted for this period."

    seen = set()
    lines = []
    for m in metrics:
        key = m.metric_name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"{m.metric_name}: {val}")

    return "\n".join(lines)


async def build_guidance_summary(
    db: AsyncSession, company_id, period: str
) -> str:
    """Extract only guidance items, compressed."""
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.segment == "guidance",
        ).order_by(ExtractedMetric.confidence.desc()).limit(15)
    )
    items = q.scalars().all()

    if not items:
        return "No guidance items found."

    lines = []
    for m in items:
        name = m.metric_name.replace("GUIDANCE: ", "")
        val = m.metric_text if m.metric_text else f"{m.metric_value} {m.unit}"
        lines.append(f"{name}: {val}")
    return "\n".join(lines)


async def build_prior_period_context(
    db: AsyncSession, company_id, current_period: str
) -> str:
    """
    Compressed prior period summary.
    Uses stored research output if available, otherwise falls back to metrics.
    """
    prior = _previous_period(current_period)

    q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company_id,
            ResearchOutput.period_label == prior,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(desc(ResearchOutput.created_at)).limit(1)
    )
    output = q.scalar_one_or_none()

    if output and output.content_json:
        try:
            data = json.loads(output.content_json)
            parts = []
            briefing = data.get("briefing") or data.get("synthesis")
            if briefing:
                if isinstance(briefing, dict):
                    if briefing.get("bottom_line"):
                        parts.append(f"Prior bottom line: {briefing['bottom_line']}")
                    if briefing.get("what_happened"):
                        parts.append(f"Prior summary: {briefing['what_happened'][:300]}")
            if parts:
                return "\n".join(parts)
        except Exception:
            pass

    return await build_kpi_summary(db, company_id, prior, max_metrics=15)


async def build_tracked_kpi_context(
    db: AsyncSession, company_id
) -> str:
    """Build context from tracked KPIs with their scores — what the analyst cares about."""
    q = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == company_id)
        .order_by(TrackedKPI.display_order)
    )
    kpis = q.scalars().all()
    if not kpis:
        return ""

    lines = []
    for kpi in kpis:
        scores_q = await db.execute(
            select(KPIScore).where(KPIScore.tracked_kpi_id == kpi.id)
            .order_by(KPIScore.period_label.desc()).limit(3)
        )
        scores = scores_q.scalars().all()
        history = ", ".join(
            f"{s.period_label}: {s.value}{' (score '+str(s.score)+'/5)' if s.score else ''}"
            for s in scores
        )
        lines.append(f"{kpi.kpi_name}: {history}" if history else kpi.kpi_name)

    return "Tracked KPIs:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# ExtractionProfile-based builders (used by legacy pipeline)
# ─────────────────────────────────────────────────────────────────

async def build_confidence_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Build context from extraction confidence profiles — management language signals."""
    q = await db.execute(
        select(ExtractionProfile).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
        ).order_by(ExtractionProfile.created_at.desc()).limit(3)
    )
    profiles = q.scalars().all()
    if not profiles:
        return ""

    lines = []
    for p in profiles:
        cp = p.confidence_profile
        if not cp or not isinstance(cp, dict):
            continue
        signal = cp.get("overall_signal", "unknown")
        hedge = cp.get("hedge_rate", 0)
        one_off = cp.get("one_off_rate", 0)
        lines.append(
            f"Signal: {signal} | hedge_rate: {hedge:.0%} | one_off_rate: {one_off:.0%}"
        )
        if cp.get("hedge_terms_used"):
            lines.append(f"  Hedging language: {', '.join(cp['hedge_terms_used'][:5])}")
    return "Management language analysis:\n" + "\n".join(lines) if lines else ""


async def build_segment_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Build context from segment decomposition data."""
    q = await db.execute(
        select(ExtractionProfile).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
            ExtractionProfile.segment_data.isnot(None),
        ).order_by(ExtractionProfile.created_at.desc()).limit(1)
    )
    profile = q.scalar_one_or_none()
    if not profile or not profile.segment_data:
        return ""

    sd = profile.segment_data
    if not isinstance(sd, dict):
        return ""

    segments = sd.get("segments", [])
    if not segments:
        return ""

    lines = []
    for seg in segments[:10]:
        if isinstance(seg, dict):
            name = seg.get("name", "Unknown")
            rev = seg.get("revenue", "")
            margin = seg.get("margin", "")
            parts = [name]
            if rev:
                parts.append(f"rev={rev}")
            if margin:
                parts.append(f"margin={margin}")
            lines.append(" | ".join(parts))
    return "Segment breakdown:\n" + "\n".join(lines) if lines else ""


async def build_one_off_context(
    db: AsyncSession, company_id, period: str
) -> str:
    """Summarise one-off / non-recurring items for this period."""
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.is_one_off == True,
        ).order_by(ExtractedMetric.confidence.desc()).limit(15)
    )
    items = q.scalars().all()
    if not items:
        return ""

    lines = []
    for m in items:
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"[ONE-OFF] {m.metric_name}: {val}")
    return "Non-recurring items:\n" + "\n".join(lines)


async def build_mda_narrative_context(
    db: AsyncSession, company_id, period: str, *, max_chars: int = 8000
) -> str:
    """Return the stored MD&A narrative text (already captured during extraction)."""
    q = await db.execute(
        select(ExtractionProfile.mda_narrative).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period,
            ExtractionProfile.mda_narrative.isnot(None),
        ).order_by(ExtractionProfile.created_at.desc()).limit(1)
    )
    row = q.scalar_one_or_none()
    if not row:
        return ""
    return row[:max_chars]


# ─────────────────────────────────────────────────────────────────
# Legacy step-specific builders (kept for backward compat —
# use build_agent_context() for Phase 1 agents)
# ─────────────────────────────────────────────────────────────────

async def build_briefing_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """LEGACY — use build_agent_context() for Phase 1 agents."""
    return {
        "thesis": await build_thesis_context(db, company_id),
        "kpis": await build_kpi_summary(db, company_id, period),
        "guidance": await build_guidance_summary(db, company_id, period),
        "prior_period": await build_prior_period_context(db, company_id, period),
        "tracked_kpis": await build_tracked_kpi_context(db, company_id),
        "confidence": await build_confidence_context(db, company_id, period),
        "segments": await build_segment_context(db, company_id, period),
        "one_offs": await build_one_off_context(db, company_id, period),
    }


async def build_comparison_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """LEGACY — use build_agent_context() for Phase 1 agents."""
    return {
        "thesis": await build_thesis_context(db, company_id),
        "current_kpis": await build_kpi_summary(db, company_id, period),
        "prior_period": await build_prior_period_context(db, company_id, period),
        "confidence": await build_confidence_context(db, company_id, period),
        "one_offs": await build_one_off_context(db, company_id, period),
    }


async def build_surprise_context(
    db: AsyncSession, company_id, period: str
) -> dict:
    """LEGACY — use build_agent_context() for Phase 1 agents."""
    prior = _previous_period(period)
    return {
        "guidance": await build_guidance_summary(db, company_id, prior),
        "actuals": await build_kpi_summary(db, company_id, period),
    }
