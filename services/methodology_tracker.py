"""
Methodology Change Tracker — longitudinal diff on non-GAAP bridge data
and prior-period metric restatements.

Tier 2.2 — the Ally case: Q3 quietly restated Core ROTCE from 15.3% to
12.3%. The prior quarter's bull thesis had leaned on the 15.3% figure;
no alert fired. This module surfaces four classes of methodology drift
so the reconciliation gate can flag them before agents run:

  1. New bridge adjustments (e.g. "strategic repositioning charges"
     appear this period but did not exist last period)
  2. Removed bridge adjustments (a line item disappears — either it
     genuinely didn't happen, or it was rolled into another item)
  3. Gap drift (GAAP-to-adjusted gap widening or narrowing materially
     across periods)
  4. Prior-period metric restatements (the same metric+period
     combination is extracted from multiple filings with materially
     different values — the later value is a restatement of the
     earlier one)

All checks are deterministic — read from `extracted_metrics` rows
persisted by `services/non_gaap_tracker.persist_bridge_data`. No LLM
calls. Input is the current period label; output is a structured
report the reconciliation module aggregates into its warnings.

Scope boundary: this module does NOT extract bridges. Extraction lives
in `services/non_gaap_tracker.py` and runs at ingestion time. This
module only reads what was stored.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, ExtractedMetric
from services.period_utils import shift_period

logger = logging.getLogger(__name__)


# Minimum QoQ gap-drift threshold (percentage-points) above which we
# flag the shift. 20% relative change on a 3% gap is 0.6pp — that's
# noise from rounding. Require both: relative change ≥25% AND absolute
# change ≥0.5pp. Stops hair-trigger alerts on tiny bridges.
_GAP_DRIFT_REL_THRESHOLD = 0.25
_GAP_DRIFT_ABS_THRESHOLD = 0.5   # percentage points

# Restatement sensitivity: same metric+period extracted from filings
# published at different times. Materially different = >1% relative
# spread, ignoring rounding. Tight enough to catch 15.3→12.3 (20%
# spread) but loose enough to tolerate 4.12% vs 4.13% rounding.
_RESTATEMENT_TOLERANCE = 0.01


@dataclass
class MethodologyFlag:
    """One methodology-change issue."""
    kind: str            # new_adjustment | removed_adjustment | gap_drift | restatement
    severity: str        # info | warning | critical
    metric: str
    message: str
    prior_period: Optional[str] = None
    current_value: Optional[float] = None
    prior_value: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MethodologyReport:
    """Aggregated methodology checks for one period."""
    period_label:            str
    prior_period:            Optional[str] = None
    flags:                   list[dict] = field(default_factory=list)
    bridge_labels_current:   list[str] = field(default_factory=list)
    bridge_labels_prior:     list[str] = field(default_factory=list)
    gap_current_pct:         Optional[float] = None
    gap_prior_pct:           Optional[float] = None
    restatement_count:       int = 0
    new_adjustment_count:    int = 0
    removed_adjustment_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _canonical_label(label: str) -> str:
    """Collapse label into word-set for fuzzy matching.

    'Restructuring Charges (Non-recurring)' → 'restructuring'
    'Restructuring costs' → 'restructuring'
    """
    if not label:
        return ""
    words = re.findall(r"[a-z]+", label.lower())
    filler = {
        "and", "the", "of", "in", "for", "to", "a", "an", "on", "from",
        "costs", "charges", "items", "expense", "expenses", "other",
        "non", "recurring", "related", "associated",
    }
    keep = [w for w in words if w not in filler and len(w) > 2]
    return " ".join(sorted(set(keep))) if keep else (words[0] if words else "")


async def _load_bridge_rows(
    db: AsyncSession, company_id, period_label: str,
) -> list[ExtractedMetric]:
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period_label,
            ExtractedMetric.segment == "non_gaap_bridge",
        )
    )
    return q.scalars().all()


def _extract_labels_and_gap(rows: list[ExtractedMetric]) -> tuple[list[str], Optional[float]]:
    """Pull BRIDGE_ADJ:* labels and the primary BRIDGE_GAP_PCT value
    out of a batch of bridge metric rows."""
    labels: list[str] = []
    gap_pct: Optional[float] = None
    # If multiple GAP_PCT rows exist (e.g. one per bridged metric),
    # take the largest magnitude — that's the most material drift
    # signal. Otherwise a tiny 0.2% EPS bridge could mask a 15% EBITDA
    # bridge.
    max_abs_gap = -1.0
    for r in rows:
        name = (r.metric_name or "").strip()
        if name.startswith("BRIDGE_ADJ:"):
            labels.append(name[len("BRIDGE_ADJ:"):].strip())
        elif name.startswith("BRIDGE_GAP_PCT"):
            try:
                val = float(r.metric_value) if r.metric_value is not None else None
            except (TypeError, ValueError):
                val = None
            if val is not None and abs(val) > max_abs_gap:
                max_abs_gap = abs(val)
                gap_pct = val
    return labels, gap_pct


# ─────────────────────────────────────────────────────────────────
# Bridge methodology diff
# ─────────────────────────────────────────────────────────────────

def _diff_labels(current: list[str], prior: list[str]) -> tuple[list[str], list[str]]:
    """Return (new_labels, removed_labels). Fuzzy-matched so
    "Restructuring costs" and "Restructuring Charges" are treated as
    the same item."""
    current_canon = {_canonical_label(l): l for l in current if l}
    prior_canon   = {_canonical_label(l): l for l in prior if l}

    new_keys = [k for k in current_canon if k and k not in prior_canon]
    removed_keys = [k for k in prior_canon if k and k not in current_canon]

    return (
        [current_canon[k] for k in new_keys],
        [prior_canon[k] for k in removed_keys],
    )


def _gap_drift_flag(
    current_gap: Optional[float], prior_gap: Optional[float],
    current_period: str, prior_period: str,
) -> Optional[MethodologyFlag]:
    if current_gap is None or prior_gap is None:
        return None
    abs_change = current_gap - prior_gap
    if abs(abs_change) < _GAP_DRIFT_ABS_THRESHOLD:
        return None
    # Relative change vs prior magnitude. Guard zero prior.
    denom = abs(prior_gap) if prior_gap else 1.0
    rel_change = abs(abs_change) / denom
    if rel_change < _GAP_DRIFT_REL_THRESHOLD:
        return None
    direction = "widened" if abs(current_gap) > abs(prior_gap) else "narrowed"
    severity = "warning" if direction == "widened" else "info"
    return MethodologyFlag(
        kind="gap_drift",
        severity=severity,
        metric="non_gaap_bridge_gap_pct",
        message=(
            f"GAAP-to-adjusted gap {direction} from {prior_gap:+.2f}% "
            f"({prior_period}) to {current_gap:+.2f}% ({current_period})"
        ),
        prior_period=prior_period,
        current_value=current_gap,
        prior_value=prior_gap,
    )


# ─────────────────────────────────────────────────────────────────
# Restatement detection
# ─────────────────────────────────────────────────────────────────

async def detect_metric_restatements(
    db: AsyncSession, company_id, current_period: str,
) -> list[MethodologyFlag]:
    """Flag prior-period metric values that differ materially between
    extractions sourced from different filings.

    Mechanism: for every ExtractedMetric with `period_label = prior_P`
    (i.e. it RELATES TO prior_P), join to `Document.period_label` (the
    filing the row was extracted from). If two rows for the same
    (metric_name, prior_P) exist but came from different filings AND
    the later filing reports a materially different value, the later
    value is a restatement of the earlier one.

    Conservative: skips bridge rows (own diff upstream), guidance
    rows, and non-numeric metrics.
    """
    q = await db.execute(
        select(
            ExtractedMetric.metric_name,
            ExtractedMetric.period_label,
            ExtractedMetric.metric_value,
            ExtractedMetric.document_id,
            Document.period_label.label("source_period"),
            Document.published_at,
        )
        .join(Document, Document.id == ExtractedMetric.document_id)
        .where(ExtractedMetric.company_id == company_id)
        .where(ExtractedMetric.segment != "guidance")
        .where(ExtractedMetric.segment != "non_gaap_bridge")
        .where(ExtractedMetric.metric_value.isnot(None))
        .where(ExtractedMetric.period_label.isnot(None))
    )
    rows = q.all()

    # Bucket by (metric_name, refers_to_period)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        name = (r.metric_name or "").strip()
        if not name or ":" in name:
            # Skip BRIDGE_* and other prefixed metadata rows
            continue
        try:
            val = float(r.metric_value)
        except (TypeError, ValueError):
            continue
        refers_to = r.period_label
        # Only consider refers-to periods STRICTLY BEFORE the current
        # filing's period — a restatement is by definition the new
        # filing reporting a different value for an older period.
        if refers_to == current_period:
            # The current filing's own-period value isn't a restatement,
            # it's just the current measurement. Skip.
            continue
        buckets.setdefault((name.lower(), refers_to), []).append({
            "value":         val,
            "source_period": r.source_period,
            "published_at":  r.published_at,
            "document_id":   r.document_id,
        })

    flags: list[MethodologyFlag] = []
    for (metric_key, refers_to), entries in buckets.items():
        if len(entries) < 2:
            continue
        # Dedupe by source_period — multiple docs from the same filing
        # period reporting the same value are not a restatement, just
        # re-extraction.
        by_source: dict[str, float] = {}
        order: list[tuple[str, float, object]] = []
        for e in entries:
            sp = e["source_period"] or ""
            if sp not in by_source:
                by_source[sp] = e["value"]
                order.append((sp, e["value"], e["published_at"]))
        if len(by_source) < 2:
            continue

        # Order chronologically by source filing period
        from services.period_utils import period_to_tuple
        order_sorted = sorted(
            order,
            key=lambda t: period_to_tuple(t[0]) if t[0] else (0, 0),
        )
        earliest_period, earliest_val, _ = order_sorted[0]
        latest_period,   latest_val,   _ = order_sorted[-1]

        if earliest_period == latest_period:
            continue
        denom = max(abs(earliest_val), abs(latest_val), 1e-9)
        rel_spread = abs(latest_val - earliest_val) / denom
        if rel_spread < _RESTATEMENT_TOLERANCE:
            continue

        # Only flag if the latest filing is the current filing (or later).
        # Otherwise we're looking at history an analyst has already seen.
        from services.period_utils import period_to_tuple as pt
        if pt(latest_period) < pt(current_period):
            continue

        flags.append(MethodologyFlag(
            kind="restatement",
            severity="warning",
            metric=metric_key,
            message=(
                f"{metric_key} for {refers_to} restated from "
                f"{earliest_val:.4g} (as reported in {earliest_period}) to "
                f"{latest_val:.4g} (as reported in {latest_period}); "
                f"{rel_spread*100:.1f}% relative change — review any prior "
                f"assessment that referenced the original figure."
            ),
            prior_period=refers_to,
            current_value=latest_val,
            prior_value=earliest_val,
        ))
    return flags


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

async def compute_methodology_report(
    db: AsyncSession, company_id, period_label: str,
) -> MethodologyReport:
    """Run all methodology checks and return a single aggregated report.

    Called by services/reconciliation.py during the pre-flight gate.
    Safe to call repeatedly (idempotent; no writes).
    """
    prior_period = shift_period(period_label, quarters=-1)

    current_rows = await _load_bridge_rows(db, company_id, period_label)
    current_labels, current_gap = _extract_labels_and_gap(current_rows)

    prior_labels: list[str] = []
    prior_gap: Optional[float] = None
    if prior_period:
        prior_rows = await _load_bridge_rows(db, company_id, prior_period)
        prior_labels, prior_gap = _extract_labels_and_gap(prior_rows)

    flags: list[MethodologyFlag] = []

    # Bridge diff only meaningful when we have prior data AND current data
    if current_labels or prior_labels:
        new_labels, removed_labels = _diff_labels(current_labels, prior_labels)
        for lbl in new_labels:
            flags.append(MethodologyFlag(
                kind="new_adjustment",
                severity="warning",
                metric=f"bridge_adj:{lbl}",
                message=(
                    f"New bridge adjustment '{lbl}' appeared in {period_label} "
                    f"that was not in {prior_period or 'any prior period'} — "
                    "management may be introducing a new one-off category. "
                    "Check whether this is genuinely non-recurring."
                ),
                prior_period=prior_period,
            ))
        for lbl in removed_labels:
            flags.append(MethodologyFlag(
                kind="removed_adjustment",
                severity="info",
                metric=f"bridge_adj:{lbl}",
                message=(
                    f"Bridge adjustment '{lbl}' present in {prior_period} is "
                    f"absent in {period_label} — either the underlying event "
                    "did not recur, or it was rolled into another line item."
                ),
                prior_period=prior_period,
            ))

    # Gap drift
    if prior_period:
        drift = _gap_drift_flag(current_gap, prior_gap, period_label, prior_period)
        if drift:
            flags.append(drift)

    # Restatements (independent of bridge data)
    restatement_flags = await detect_metric_restatements(db, company_id, period_label)
    flags.extend(restatement_flags)

    return MethodologyReport(
        period_label=period_label,
        prior_period=prior_period,
        flags=[f.to_dict() for f in flags],
        bridge_labels_current=current_labels,
        bridge_labels_prior=prior_labels,
        gap_current_pct=current_gap,
        gap_prior_pct=prior_gap,
        restatement_count=sum(1 for f in flags if f.kind == "restatement"),
        new_adjustment_count=sum(1 for f in flags if f.kind == "new_adjustment"),
        removed_adjustment_count=sum(1 for f in flags if f.kind == "removed_adjustment"),
    )
