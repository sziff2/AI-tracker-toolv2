"""
Data Completeness + Source Coverage Gates.

Two deterministic pre-flight checks the orchestrator runs between Phase A
(extraction) and Phase B (analysis pipeline). Goal: catch the failure
mode from `Dev plans/2_pipeline_improvements.md` §1 — confident bearish
assessments generated on catastrophically incomplete data, where every
guidance field returned `unknown: None` and the system still produced a
"MISS / WEAKENED" call.

Both gates are:
  - Pure Python — no LLM, no per-call cost
  - Deterministic — same inputs → same decision
  - Warn-only by default via `settings.completeness_gate_mode` — they
    attach structured reports to `pipeline_run.warnings` but do not halt
    the pipeline until the analyst opts in by flipping the flag to "halt"

Separation of concerns:
  - `compute_completeness()` checks the EXTRACTED DATA shape — do we
    have enough metrics, guidance, comparators, management language to
    support an analysis?
  - `compute_source_coverage()` checks the DOCUMENTS AVAILABLE — do we
    have the right document TYPES (press release, transcript, etc.) to
    support a full qualitative + quantitative read?

Distinct from the daily Coverage Monitor (`agents/ingestion/coverage_monitor.py`)
which runs at company × period × time level; these gates run per-
assessment at the start of pipeline execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric, ExtractionProfile

logger = logging.getLogger(__name__)


# Status values
PROCEED              = "proceed"
PROCEED_WITH_CAVEATS = "proceed_with_caveats"
HALT_INCOMPLETE      = "halt_incomplete"


# ─────────────────────────────────────────────────────────────────
# Keyword sets — deterministic, not LLM-judged.
# Kept here rather than in a config file because these are structural
# definitions of what a completeness check IS, not user-tunable policy.
# ─────────────────────────────────────────────────────────────────

_EPS_KEYWORDS = {"eps", "earnings per share", "diluted eps", "basic eps"}
_MARGIN_KEYWORDS = {
    "margin", "nim", "net interest margin",
    "gross margin", "operating margin", "ebitda margin",
    "ebit margin", "net margin", "contribution margin",
}
_REVENUE_KEYWORDS = {"revenue", "sales", "total revenue", "net sales", "turnover"}
_BALANCE_SHEET_KEYWORDS = {
    "total assets", "deposits", "total equity", "total liabilities",
    "common equity", "tangible equity", "shareholders equity",
}
_CASH_FLOW_KEYWORDS = {
    "operating cash flow", "ocf", "fcf", "free cash flow",
    "cash from operations", "cash flow from operations",
}
# Doc types that satisfy the "core results disclosure" requirement
_RESULTS_DOC_TYPES = {
    "earnings_release", "8-K", "10-Q", "10-K", "20-F", "40-F", "6-K",
    "annual_report", "interim_report", "half_year_report",
}


def _metric_matches_any(metric_name: str, keywords: set[str]) -> bool:
    """Case-insensitive substring match of any keyword against metric_name.
    Deliberately loose — metric names from the extractor are not fully
    standardised, so we match on semantic fragments."""
    if not metric_name:
        return False
    low = metric_name.lower()
    return any(kw in low for kw in keywords)


# ─────────────────────────────────────────────────────────────────
# Completeness report data shape
# ─────────────────────────────────────────────────────────────────

@dataclass
class CompletenessReport:
    """Structured result of the completeness check — safe to serialise
    into pipeline_run.warnings JSONB for audit + UI display."""
    status: str                           # proceed | proceed_with_caveats | halt_incomplete
    required: dict = field(default_factory=dict)     # {check_name: bool}
    recommended: dict = field(default_factory=dict)  # {check_name: bool}
    recommended_score: float = 0.0        # 0.0-1.0 fraction of recommended checks passed
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    reason: str = ""                      # human-readable one-line summary
    checked_at: str = ""                  # ISO timestamp

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceCoverageReport:
    """Which document TYPES do we have for this assessment?"""
    status: str                           # proceed | proceed_with_caveats | halt_incomplete
    has_results_doc: bool = False         # earnings_release OR equivalent
    has_transcript: bool = False
    has_presentation: bool = False
    missing_required: list[str] = field(default_factory=list)
    missing_recommended: list[str] = field(default_factory=list)
    reason: str = ""
    checked_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
# Completeness gate
# ─────────────────────────────────────────────────────────────────

async def compute_completeness(
    db: AsyncSession, company_id, period_label: str,
) -> CompletenessReport:
    """Check that the extracted data is rich enough for a credible
    assessment. Returns a structured report; does NOT decide whether to
    halt — that's the orchestrator's call based on settings.

    The four REQUIRED checks come from plan 2 §1 and are the minimum
    floor to produce a credible assessment:
      1. Current period EPS extracted (GAAP or adjusted)
      2. At least one margin metric
      3. At least one prior-period comparator
      4. At least one forward guidance data point

    The seven RECOMMENDED checks reflect a healthier extraction. Need
    70%+ populated or status downgrades to proceed_with_caveats.
    """
    checked_at = datetime.now(timezone.utc).isoformat()

    # Pull all current-period metrics once; everything else derives from this.
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
        )
    )
    metrics = q.scalars().all()

    # ── Required checks ──────────────────────────────────────────
    has_eps    = any(_metric_matches_any(m.metric_name, _EPS_KEYWORDS) for m in metrics)
    has_margin = any(_metric_matches_any(m.metric_name, _MARGIN_KEYWORDS) for m in metrics)

    # Guidance = segment tagged "guidance" OR metric_name prefixed "GUIDANCE:"
    has_guidance = any(
        m.segment == "guidance" or (m.metric_name or "").startswith("GUIDANCE:")
        for m in metrics
    )

    # Prior-period comparator: does ANY metric name from the current period
    # also exist for a prior period (same company)? Cheap signal — we just
    # need to know that some comparator data is available to the pipeline.
    has_prior_comparator = await _has_prior_comparator(
        db, company_id, period_label, metrics
    )

    required = {
        "current_eps":             has_eps,
        "margin_metric":           has_margin,
        "prior_period_comparator": has_prior_comparator,
        "forward_guidance":        has_guidance,
    }
    missing_required = [name for name, ok in required.items() if not ok]

    # ── Recommended checks ───────────────────────────────────────
    has_revenue = any(_metric_matches_any(m.metric_name, _REVENUE_KEYWORDS) for m in metrics)
    # Segment breakdown: ≥2 distinct non-null, non-guidance segments
    segments = {
        m.segment for m in metrics
        if m.segment and m.segment != "guidance"
    }
    has_segment_breakdown = len(segments) >= 2
    has_balance_sheet = any(
        _metric_matches_any(m.metric_name, _BALANCE_SHEET_KEYWORDS) for m in metrics
    )
    has_cash_flow = any(
        _metric_matches_any(m.metric_name, _CASH_FLOW_KEYWORDS) for m in metrics
    )
    # Management language analysis from extraction profile confidence_profile
    has_mgmt_language = await _has_mgmt_language(db, company_id, period_label)
    # Dual comparators — both QoQ and YoY available
    has_dual_comparators = await _has_dual_comparators(
        db, company_id, period_label, metrics
    )
    # Industry KPIs (bank / insurance-specific from the sector-aware
    # extraction pipeline) — presence of any KPI_TABLE-type rows
    has_industry_kpis = await _has_industry_kpis(db, company_id, period_label)

    recommended = {
        "revenue":            has_revenue,
        "segment_breakdown":  has_segment_breakdown,
        "balance_sheet":      has_balance_sheet,
        "cash_flow":          has_cash_flow,
        "mgmt_language":      has_mgmt_language,
        "dual_comparators":   has_dual_comparators,
        "industry_kpis":      has_industry_kpis,
    }
    missing_recommended = [name for name, ok in recommended.items() if not ok]
    recommended_score = (
        sum(1 for ok in recommended.values() if ok) / len(recommended)
        if recommended else 0.0
    )

    # ── Decide status ────────────────────────────────────────────
    if missing_required:
        status = HALT_INCOMPLETE
        reason = f"Missing {len(missing_required)} required field(s): {', '.join(missing_required)}"
    elif recommended_score < 0.5:
        # <50% recommended is a stricter halt even with all required present
        status = HALT_INCOMPLETE
        reason = (
            f"Only {int(recommended_score * 100)}% of recommended fields populated "
            f"(threshold 50% for halt)"
        )
    elif recommended_score < 0.7:
        status = PROCEED_WITH_CAVEATS
        reason = (
            f"{int(recommended_score * 100)}% of recommended fields populated; "
            f"missing: {', '.join(missing_recommended)}"
        )
    else:
        status = PROCEED
        reason = f"All required + {int(recommended_score * 100)}% of recommended populated"

    return CompletenessReport(
        status=status,
        required=required,
        recommended=recommended,
        recommended_score=round(recommended_score, 3),
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        reason=reason,
        checked_at=checked_at,
    )


# ─────────────────────────────────────────────────────────────────
# Source coverage gate
# ─────────────────────────────────────────────────────────────────

async def compute_source_coverage(
    db: AsyncSession, company_id, period_label: str,
) -> SourceCoverageReport:
    """Check that the documents needed for a full read are ingested +
    parsed for this (company, period). Distinct from the daily Coverage
    Monitor: this is the PER-ASSESSMENT check at pipeline kick-off."""
    checked_at = datetime.now(timezone.utc).isoformat()

    q = await db.execute(
        select(Document.document_type).where(
            Document.company_id == company_id,
            Document.period_label == period_label,
            Document.parsing_status == "completed",
        )
    )
    types_present = {row[0] for row in q.all() if row[0]}

    has_results_doc = bool(types_present & _RESULTS_DOC_TYPES)
    has_transcript  = "transcript" in types_present
    has_presentation = bool(
        types_present & {"presentation", "investor_presentation"}
    )

    missing_required = []
    if not has_results_doc:
        missing_required.append("results_doc (earnings_release / 10-Q / 10-K / 6-K / interim_report)")

    missing_recommended = []
    if not has_transcript:
        missing_recommended.append("transcript")
    if not has_presentation:
        missing_recommended.append("presentation")

    if missing_required:
        status = HALT_INCOMPLETE
        reason = f"Missing required document type(s): {', '.join(missing_required)}"
    elif not has_transcript:
        # Transcript is critical per plan 2 §4 — without it Bear/Bull lose
        # roughly 40% of their analytical surface (tone, guidance, Q&A)
        status = PROCEED_WITH_CAVEATS
        reason = "Transcript missing — analysis limited to quantitative metrics"
    elif missing_recommended:
        status = PROCEED_WITH_CAVEATS
        reason = f"Missing recommended document(s): {', '.join(missing_recommended)}"
    else:
        status = PROCEED
        reason = "All required + recommended documents present"

    return SourceCoverageReport(
        status=status,
        has_results_doc=has_results_doc,
        has_transcript=has_transcript,
        has_presentation=has_presentation,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        reason=reason,
        checked_at=checked_at,
    )


# ─────────────────────────────────────────────────────────────────
# Private helpers (DB queries)
# ─────────────────────────────────────────────────────────────────

async def _has_prior_comparator(
    db: AsyncSession, company_id, period_label: str,
    current_metrics: list[ExtractedMetric],
) -> bool:
    """True if ANY of the current period's metric names exists for a
    prior period for the same company. Doesn't require a specific prior
    (Q-1, Y-1 etc.) — just that at least one metric has a historical
    value to compare against."""
    if not current_metrics:
        return False
    # Sample at most 20 names to keep the IN list small
    names_to_check = [m.metric_name for m in current_metrics if m.metric_name][:20]
    if not names_to_check:
        return False

    q = await db.execute(
        select(func.count(ExtractedMetric.id)).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label != period_label,
            ExtractedMetric.metric_name.in_(names_to_check),
        )
    )
    return (q.scalar() or 0) > 0


async def _has_dual_comparators(
    db: AsyncSession, company_id, period_label: str,
    current_metrics: list[ExtractedMetric],
) -> bool:
    """True if at least one current metric has BOTH a prior-quarter AND
    prior-year comparator available. Stricter than _has_prior_comparator."""
    from services.harvester.coverage_advanced import _period_end
    from datetime import timedelta

    current_end = _period_end(period_label)
    if not current_end:
        return False

    # Find prior quarter period_label — naive: one quarter earlier
    # (acceptable for QoQ comparator detection; refinement would parse
    # _Q{n} → _Q{n-1})
    prior_q = _shift_period(period_label, quarters=-1)
    prior_y = _shift_period(period_label, quarters=-4)
    if not prior_q or not prior_y:
        return False

    names_to_check = [m.metric_name for m in current_metrics if m.metric_name][:20]
    if not names_to_check:
        return False

    q = await db.execute(
        select(ExtractedMetric.metric_name, ExtractedMetric.period_label).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label.in_([prior_q, prior_y]),
            ExtractedMetric.metric_name.in_(names_to_check),
        )
    )
    rows = q.all()
    # Group: need a metric_name that appears in BOTH prior_q and prior_y
    by_name: dict[str, set[str]] = {}
    for name, period in rows:
        by_name.setdefault(name, set()).add(period)
    return any(
        prior_q in periods and prior_y in periods
        for periods in by_name.values()
    )


def _shift_period(period_label: str, *, quarters: int) -> Optional[str]:
    """Re-export of services.period_utils.shift_period for tests that
    import this symbol from this module. Prefer importing from
    period_utils directly in new code."""
    from services.period_utils import shift_period
    return shift_period(period_label, quarters=quarters)


async def _has_mgmt_language(
    db: AsyncSession, company_id, period_label: str,
) -> bool:
    """True if the extraction profile's confidence_profile has >= 10 items.
    Plan 2 §1 uses 10 as the threshold for 'meaningful management
    language analysis'."""
    q = await db.execute(
        select(ExtractionProfile.confidence_profile).where(
            ExtractionProfile.company_id == company_id,
            ExtractionProfile.period_label == period_label,
        )
    )
    for row in q.all():
        prof = row[0]
        if not isinstance(prof, dict):
            continue
        # confidence_profile structure varies across extraction versions.
        # Accept either a 'signals' / 'items' list or a total count field.
        signals = prof.get("signals") or prof.get("items") or []
        if isinstance(signals, list) and len(signals) >= 10:
            return True
        total = prof.get("total_signals") or prof.get("item_count") or 0
        if isinstance(total, (int, float)) and total >= 10:
            return True
    return False


async def _has_industry_kpis(
    db: AsyncSession, company_id, period_label: str,
) -> bool:
    """True if any KPI-type rows were extracted — indicates the sector-
    aware extraction pipeline (bank supplementary tables, insurance
    combined ratios, etc.) actually found industry-specific data.

    Uses a keyword heuristic rather than a schema flag because the
    extractor stores industry KPIs in the same ExtractedMetric table
    without an explicit type marker."""
    industry_kpi_keywords = {
        "cet1", "tier 1", "tier1", "rwa", "nco rate", "charge-off",
        "combined ratio", "loss ratio", "expense ratio",
        "rotce", "tbvps", "allowance coverage",
    }
    q = await db.execute(
        select(ExtractedMetric.metric_name).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
        )
    )
    for row in q.all():
        name = (row[0] or "").lower()
        if any(kw in name for kw in industry_kpi_keywords):
            return True
    return False
