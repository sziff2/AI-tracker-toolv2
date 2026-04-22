"""
Reconciliation Module — deterministic pre-flight gate sitting alongside
completeness + source-coverage.

Purpose
-------
Catch four classes of error BEFORE agent execution:

  1. Formula / denominator errors — "18% charge-off rate against
     revenue" is fabricated because charge-off rate uses average loans
     as the denominator, not revenue. Caught by the validation
     registry in metric_definitions.py.
  2. Out-of-range values — bank NIM of 8% is physically implausible;
     almost certainly an extraction error. Caught by per-KPI
     reasonable-range bounds.
  3. Structural inconsistencies — Q1+Q2+Q3+Q4 ≠ FY, segment revenues
     don't sum to consolidated, assets ≠ liabilities+equity, net
     income on P&L ≠ net income on CF. Read from the ExtractionProfile
     reconciliation column (populated by metric_extractor hook).
  4. Anomalous QoQ moves — recovery revenue jumping 26,000% without a
     disclosed cause. Caught by extraction_comparator.compare_to_prior.

Mirrors services/completeness_gate.py almost exactly — same status
semantics, same shape of warn-only mode, same JSONB persistence into
pipeline_run.warnings. Orchestrator wires it as a third pre-flight
gate alongside the existing two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Company, ExtractedMetric, ExtractionProfile
from services.extraction_comparator import compare_to_prior
from services.metric_definitions import (
    ValidationIssue, validate_metrics_batch,
)
from services.period_utils import shift_period

logger = logging.getLogger(__name__)


# Status values (deliberately match completeness_gate for consistency)
PROCEED              = "proceed"
PROCEED_WITH_CAVEATS = "proceed_with_caveats"
HALT_INCOMPLETE      = "halt_incomplete"

# Cross-source consistency threshold (percent). Same metric extracted
# from two different documents should agree within this band.
# 2% accommodates "3.4%" vs "3.45%" rounding across press release /
# 10-Q. The plan calls for 2% as a safer threshold than 1%.
_CROSS_SOURCE_TOLERANCE = 0.02


@dataclass
class ReconciliationReport:
    """Structured output persisted to pipeline_run.warnings JSONB."""
    status:                 str
    issues:                 list[dict] = field(default_factory=list)
    validation_issues:      list[dict] = field(default_factory=list)
    structural_checks:      dict       = field(default_factory=dict)
    anomaly_checks:         dict       = field(default_factory=dict)
    cross_source_checks:    dict       = field(default_factory=dict)
    methodology_checks:     dict       = field(default_factory=dict)
    critical_count:         int = 0
    warning_count:          int = 0
    info_count:             int = 0
    reason:                 str = ""
    checked_at:             str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
# Sub-checks
# ─────────────────────────────────────────────────────────────────

async def _validation_issues(
    db: AsyncSession, company_id, period_label: str,
) -> tuple[list[ValidationIssue], int]:
    """Load extracted metrics for the period, validate each against the
    universal + sector-specific rules. Returns the list of issues + the
    count of metrics checked."""
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
            ExtractedMetric.segment != "guidance",
        )
    )
    rows = q.scalars().all()

    # Sector routing — one query for the company
    cq = await db.execute(select(Company).where(Company.id == company_id))
    company = cq.scalar_one_or_none()
    sector = (company.sector if company else "") or ""
    industry = (company.industry if company else "") or ""

    metric_dicts = []
    for r in rows:
        # Skip non-numeric and bridge-gap/guidance rows (not real
        # metrics; validation against their values is meaningless)
        if r.metric_value is None:
            continue
        if r.metric_name and ":" in r.metric_name:
            prefix = r.metric_name.split(":", 1)[0].strip().upper()
            if prefix.startswith(("BRIDGE_GAP", "BRIDGE_ADJ", "GUIDANCE")):
                continue
        # Denominator may live in qualifier_json
        denom = None
        if r.qualifier_json and isinstance(r.qualifier_json, dict):
            denom = r.qualifier_json.get("denominator")
        metric_dicts.append({
            "metric_name":  r.metric_name,
            "metric_value": r.metric_value,
            "unit":         r.unit,
            "denominator":  denom,
        })

    issues = validate_metrics_batch(metric_dicts, sector=sector, industry=industry)
    return issues, len(metric_dicts)


async def _structural_checks(
    db: AsyncSession, company_id, period_label: str,
) -> dict:
    """Read the ExtractionProfile.reconciliation column (populated by
    metric_extractor._persist_extraction_profile). Returns {} if nothing
    has been persisted yet — warn-only fallback."""
    try:
        q = await db.execute(
            select(ExtractionProfile.reconciliation).where(
                ExtractionProfile.company_id == company_id,
                ExtractionProfile.period_label == period_label,
                ExtractionProfile.reconciliation.isnot(None),
            ).order_by(desc(ExtractionProfile.created_at)).limit(1)
        )
        row = q.scalar_one_or_none()
        if row and isinstance(row, dict):
            return row
    except Exception as exc:
        logger.warning("Failed to load structural reconciliation: %s", exc)
    return {}


async def _anomaly_checks(
    db: AsyncSession, company_id, period_label: str,
) -> dict:
    """QoQ anomaly detection using the orphaned extraction_comparator.
    Mirrors what metric_store already does but aggregates at the gate
    level (different output shape — structured issues)."""
    qoq_period = shift_period(period_label, quarters=-1) if not period_label.endswith("_FY") else None
    if not qoq_period:
        return {"anomalies": [], "note": "No QoQ prior period (FY or non-quarter)"}

    async def _period_rows(label: str) -> list[ExtractedMetric]:
        q = await db.execute(
            select(ExtractedMetric).where(
                ExtractedMetric.company_id == company_id,
                ExtractedMetric.period_label == label,
                ExtractedMetric.segment != "guidance",
            )
        )
        return q.scalars().all()

    current = await _period_rows(period_label)
    prior   = await _period_rows(qoq_period)

    current_items = [
        {"metric_name": r.metric_name, "metric_value": r.metric_value}
        for r in current
    ]
    prior_items = [
        {"metric_name": r.metric_name, "metric_value": r.metric_value}
        for r in prior
    ]
    if not prior_items:
        return {"anomalies": [], "note": f"No prior-period data for {qoq_period}"}

    result = compare_to_prior(
        current_items, prior_items,
        current_period=period_label, prior_period=qoq_period,
    )
    return {
        "anomalies":         result.get("anomalies", []),
        "missing_in_current": result.get("missing_in_current", []),
        "qoq_period":        qoq_period,
    }


async def _cross_source_checks(
    db: AsyncSession, company_id, period_label: str,
) -> dict:
    """For each metric extracted from multiple documents, verify the
    values agree within _CROSS_SOURCE_TOLERANCE.

    Most common failure: a press release headline value differs from
    the 10-Q value by more than rounding — flags mis-extraction or
    legitimate "as reported vs as restated" discrepancy.
    """
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
            ExtractedMetric.segment != "guidance",
        )
    )
    rows = q.scalars().all()

    # Bucket: canonical metric name → list[(document_id, value)]
    buckets: dict[str, list[tuple]] = {}
    for r in rows:
        if r.metric_value is None or not r.metric_name:
            continue
        if ":" in r.metric_name:
            prefix = r.metric_name.split(":", 1)[0].strip().upper()
            if prefix.startswith(("BRIDGE_GAP", "BRIDGE_ADJ", "GUIDANCE")):
                continue
        try:
            val = float(r.metric_value)
        except (TypeError, ValueError):
            continue
        key = r.metric_name.strip().lower()
        buckets.setdefault(key, []).append((str(r.document_id), val))

    disagreements = []
    for name, pairs in buckets.items():
        if len(pairs) < 2:
            continue
        # Dedupe by document — same-doc same-metric is expected (two
        # extractor passes), not a cross-source conflict.
        by_doc: dict[str, float] = {}
        for doc_id, val in pairs:
            if doc_id not in by_doc:
                by_doc[doc_id] = val
        if len(by_doc) < 2:
            continue
        vals = list(by_doc.values())
        mn, mx = min(vals), max(vals)
        # Guard division by zero
        denom = abs(max(abs(mn), abs(mx))) or 1.0
        spread = abs(mx - mn) / denom
        if spread > _CROSS_SOURCE_TOLERANCE:
            disagreements.append({
                "metric":     name,
                "min":        mn,
                "max":        mx,
                "spread_pct": round(spread * 100, 2),
                "document_count": len(by_doc),
            })

    return {
        "disagreements": disagreements,
        "checked_metrics": len([v for v in buckets.values() if len({d for d, _ in v}) >= 2]),
    }


async def _safe_methodology_report(fn, db, company_id, period_label) -> dict:
    """Wrap methodology tracker so a failure never blocks the gate.
    Methodology checks are an enrichment layer, not a hard requirement —
    a DB query issue here should degrade quietly."""
    try:
        report = await fn(db, company_id, period_label)
        return report.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Methodology report failed: %s", exc)
        return {"flags": [], "error": str(exc)[:200]}


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

async def compute_reconciliation(
    db: AsyncSession, company_id, period_label: str,
) -> ReconciliationReport:
    """Run all reconciliation sub-checks and aggregate into a single
    report. Warn-only by policy — status is advisory unless the
    orchestrator is in `halt` mode and a critical issue is raised."""
    checked_at = datetime.now(timezone.utc).isoformat()

    from services.methodology_tracker import compute_methodology_report

    validation_issues, metrics_checked = await _validation_issues(
        db, company_id, period_label
    )
    structural  = await _structural_checks(db, company_id, period_label)
    anomaly     = await _anomaly_checks(db, company_id, period_label)
    cross       = await _cross_source_checks(db, company_id, period_label)
    methodology = await _safe_methodology_report(
        compute_methodology_report, db, company_id, period_label,
    )

    # Aggregate severity counts
    critical = sum(1 for i in validation_issues if i.severity == "critical")
    warning  = sum(1 for i in validation_issues if i.severity == "warning")
    info     = sum(1 for i in validation_issues if i.severity == "info")

    # Cross-source disagreements ≥5% promote to warning
    cross_disagreements = cross.get("disagreements", [])
    severe_cross = [d for d in cross_disagreements if d.get("spread_pct", 0) >= 5.0]
    warning += len(severe_cross)

    # Anomalies with severity=high count as warnings (a 26,000% QoQ
    # jump is almost certainly an extraction error, but we don't halt
    # on it — let the agent examine it and explain)
    anomaly_items = anomaly.get("anomalies", [])
    anomaly_high = [a for a in anomaly_items if a.get("severity") == "high"]
    warning += len(anomaly_high)

    # Structural failures from reconcile_extractions output
    struct_issues = structural.get("issues", []) if isinstance(structural, dict) else []
    for si in struct_issues:
        sev = (si.get("severity") or "warning").lower()
        if sev == "critical":
            critical += 1
        elif sev in {"warning", "medium"}:
            warning += 1

    # Methodology drift — new/removed adjustments, gap widening, restatements
    methodology_flags = methodology.get("flags", []) if isinstance(methodology, dict) else []
    for mf in methodology_flags:
        sev = (mf.get("severity") or "info").lower()
        if sev == "critical":
            critical += 1
        elif sev == "warning":
            warning += 1
        else:
            info += 1

    # Build human-readable issues list — each entry: one line summary
    issues_flat: list[dict] = []
    for vi in validation_issues:
        issues_flat.append({
            "source":    "validation",
            "severity":  vi.severity,
            "metric":    vi.metric_name,
            "message":   vi.suggested_check,
            "rule":      vi.rule_violated,
        })
    for d in cross_disagreements:
        issues_flat.append({
            "source":    "cross_source",
            "severity":  "warning" if d.get("spread_pct", 0) >= 5.0 else "info",
            "metric":    d["metric"],
            "message":   (
                f"Cross-source spread {d['spread_pct']}% (min={d['min']}, "
                f"max={d['max']}, docs={d['document_count']})"
            ),
            "rule":      "cross_source_disagreement",
        })
    for a in anomaly_items:
        issues_flat.append({
            "source":    "qoq_anomaly",
            "severity":  a.get("severity") or "warning",
            "metric":    a.get("metric"),
            "message":   (
                f"QoQ {a.get('prior')} → {a.get('current')} "
                f"({a.get('change_pct')}%)"
            ),
            "rule":      "qoq_anomaly",
        })
    for si in struct_issues:
        issues_flat.append({
            "source":    "structural",
            "severity":  (si.get("severity") or "warning").lower(),
            "metric":    si.get("metric") or si.get("check"),
            "message":   si.get("message") or si.get("description", ""),
            "rule":      si.get("check") or "structural",
        })
    for mf in methodology_flags:
        issues_flat.append({
            "source":    "methodology",
            "severity":  (mf.get("severity") or "info").lower(),
            "metric":    mf.get("metric"),
            "message":   mf.get("message", ""),
            "rule":      mf.get("kind", "methodology"),
        })

    # Status decision — warn-only by default, halts only on critical.
    # The orchestrator will soften further via settings.reconciliation_mode.
    if critical > 0:
        status = HALT_INCOMPLETE
        reason = f"{critical} critical issue(s) found ({warning} warning)"
    elif warning > 0 or severe_cross or anomaly_high:
        status = PROCEED_WITH_CAVEATS
        methodology_warn_count = sum(
            1 for mf in methodology_flags
            if (mf.get("severity") or "info").lower() == "warning"
        )
        reason = (
            f"{warning} warning(s), {len(anomaly_high)} anomaly, "
            f"{methodology_warn_count} methodology, "
            f"{metrics_checked} metrics validated"
        )
    else:
        status = PROCEED
        reason = f"{metrics_checked} metrics validated — no issues"

    return ReconciliationReport(
        status=status,
        issues=issues_flat,
        validation_issues=[i.to_dict() for i in validation_issues],
        structural_checks=structural if isinstance(structural, dict) else {},
        anomaly_checks=anomaly,
        cross_source_checks=cross,
        methodology_checks=methodology if isinstance(methodology, dict) else {},
        critical_count=critical,
        warning_count=warning,
        info_count=info,
        reason=reason,
        checked_at=checked_at,
    )
