"""
Persistent Metric Store — exposes QoQ and YoY comparators to agents as
**structured** data (dict) rather than the narrative blob they see today.

Why this exists
---------------
The Financial Analyst agent has been reaching for `BRIDGE_GAP_PCT:*`
rows when it wants to make a period-over-period statement, because no
structured QoQ/YoY comparator was in its inputs. BRIDGE_GAP rows are
same-period Adjusted-minus-Reported gaps — not temporal deltas — so
using them as YoY values produces confident but wrong claims
(observed live on NWC CN Q3 FY2025 in April 2026).

`get_metrics_history()` returns a dict keyed by normalised metric
name, each value containing current + prior-quarter + prior-year
absolute values plus computed change (percent or bps). Agents can
read this as a table rather than computing deltas themselves.

Reuses
------
- `services/period_utils.py::shift_period` for QoQ/YoY label arithmetic
- `services/extraction_comparator.py::compare_to_prior` for anomaly
  detection (the orphaned function we wired in here)
- A single indexed query (`ix_metrics_company_period`)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ExtractedMetric
from services.extraction_comparator import compare_to_prior
from services.period_utils import shift_period

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Metric name normalisation
# ─────────────────────────────────────────────────────────────────

# Alias dict: maps multiple surface forms of the same metric to a
# canonical key. Kept intentionally conservative — add entries when
# we see real mismatches in the wild.
_ALIAS_MAP: dict[str, str] = {
    "eps diluted":           "eps_diluted",
    "diluted eps":           "eps_diluted",
    "earnings per share diluted": "eps_diluted",
    "eps basic":             "eps_basic",
    "basic eps":             "eps_basic",
    "earnings per share basic": "eps_basic",
    "eps adjusted":          "eps_adjusted",
    "adjusted eps":          "eps_adjusted",
    "adjusted diluted eps":  "eps_adjusted",
    "net income":            "net_income",
    "net earnings":          "net_income",
    "net profit":            "net_income",
    "total revenue":         "revenue",
    "net revenue":           "revenue",
    "sales":                 "revenue",
    "net sales":             "revenue",
    "operating income":      "operating_income",
    "operating profit":      "operating_income",
    "ebit":                  "ebit",
    "ebitda":                "ebitda",
    "adjusted ebitda":       "ebitda_adjusted",
    "free cash flow":        "free_cash_flow",
    "fcf":                   "free_cash_flow",
    "net interest margin":   "nim",
    "nim":                   "nim",
}


def _normalise_metric_name(raw: str) -> str:
    """Lowercase, strip, remove punctuation, apply alias map.

    `"EPS (diluted)"` and `"Diluted EPS"` both map to `"eps_diluted"`.
    Unknown names fall through as a lowercased/cleaned key.
    """
    if not raw:
        return ""
    # Lowercase, strip
    s = raw.strip().lower()
    # Replace punctuation (incl. parentheses) with spaces — keep the
    # contents of parentheticals ("EPS (diluted)" → "eps diluted")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Alias lookup
    if s in _ALIAS_MAP:
        return _ALIAS_MAP[s]
    # Fallback: snake-case the cleaned name
    return s.replace(" ", "_")


# ─────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────

@dataclass
class MetricTimeline:
    """One row per metric — current value + QoQ + YoY comparators."""
    metric_name:    str                     # canonical name
    display_name:   str                     # original as-extracted name (most-confident)
    unit:           Optional[str] = None
    current:        Optional[float] = None
    qoq_prior:      Optional[float] = None
    qoq_change:     Optional[str]   = None  # "+226%"  or "+10 bps"  or "n/a"
    yoy_prior:      Optional[float] = None
    yoy_change:     Optional[str]   = None

    def to_dict(self) -> dict:
        return {
            "metric_name":  self.metric_name,
            "display_name": self.display_name,
            "unit":         self.unit,
            "current":      self.current,
            "qoq_prior":    self.qoq_prior,
            "qoq_change":   self.qoq_change,
            "yoy_prior":    self.yoy_prior,
            "yoy_change":   self.yoy_change,
        }


@dataclass
class MetricsHistoryResult:
    """Full response: per-metric timelines + anomaly findings."""
    period_label:   str
    qoq_period:     Optional[str]
    yoy_period:     Optional[str]
    timelines:      dict[str, MetricTimeline] = field(default_factory=dict)
    anomalies:      list[dict] = field(default_factory=list)
    missing_vs_prior: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "period_label":      self.period_label,
            "qoq_period":        self.qoq_period,
            "yoy_period":        self.yoy_period,
            "timelines":         {k: v.to_dict() for k, v in self.timelines.items()},
            "anomalies":         self.anomalies,
            "missing_vs_prior":  self.missing_vs_prior,
        }


# ─────────────────────────────────────────────────────────────────
# Change formatting
# ─────────────────────────────────────────────────────────────────

def _format_change(current: float, prior: float, unit: Optional[str]) -> str:
    """Return a human-readable change string.

    - Percent-unit metrics → basis-point change  ("+10 bps")
    - Everything else      → percent change       ("+2.5%")
    - Zero prior           → "n/a (prior=0)"
    """
    if prior == 0:
        if current == 0:
            return "0.0%"
        return "n/a (prior=0)"
    # Treat a %-unit metric as already-a-percentage — bps delta is more useful
    is_percent = (unit or "").strip() in {"%", "pct", "percent", "bps"}
    if is_percent:
        bps = round((current - prior) * 100, 1)
        sign = "+" if bps >= 0 else ""
        return f"{sign}{bps:g} bps"
    delta = (current - prior) / abs(prior)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{round(delta * 100, 1):g}%"


# ─────────────────────────────────────────────────────────────────
# Period resolution — handles FY==Q4 equivalence
# ─────────────────────────────────────────────────────────────────

def _fy_equivalent(period_label: str) -> Optional[str]:
    """Return the FY label that corresponds to a Q4 period, or vice versa.

    `2024_Q4` → `2024_FY`
    `2024_FY` → `2024_Q4`
    Anything else → None
    """
    if not period_label:
        return None
    if period_label.endswith("_Q4"):
        return period_label[:-3] + "_FY"
    if period_label.endswith("_FY"):
        return period_label[:-3] + "_Q4"
    return None


def _period_candidates(period_label: str) -> list[str]:
    """All DB labels to treat as this period.

    For `2024_Q4` returns `['2024_Q4', '2024_FY']` — both refer to the
    same underlying reporting period in many companies.
    """
    if not period_label:
        return []
    peers = [period_label]
    eq = _fy_equivalent(period_label)
    if eq:
        peers.append(eq)
    return peers


def _compute_comparator_periods(period_label: str) -> tuple[Optional[str], Optional[str]]:
    """Return (qoq_period, yoy_period) labels for a given period.

    For FY labels we map to Q4 first (FY2024 → Q4-2024) before shifting.
    """
    canonical = period_label
    if period_label.endswith("_FY"):
        canonical = period_label[:-3] + "_Q4"
    qoq = shift_period(canonical, quarters=-1)
    yoy = shift_period(canonical, quarters=-4)
    return qoq, yoy


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

async def get_metrics_history(
    db: AsyncSession,
    company_id,
    period_label: str,
) -> MetricsHistoryResult:
    """Return per-metric current + QoQ + YoY timelines for a company/period.

    One indexed query hits `extracted_metrics` across the three labels;
    then we group and format. The `ix_metrics_company_period` index
    already covers this.
    """
    qoq_period, yoy_period = _compute_comparator_periods(period_label)

    labels: list[str] = []
    for lbl in [period_label, qoq_period, yoy_period]:
        if lbl:
            labels.extend(_period_candidates(lbl))
    # Dedupe while preserving order
    seen: set[str] = set()
    labels = [lbl for lbl in labels if not (lbl in seen or seen.add(lbl))]

    if not labels:
        return MetricsHistoryResult(
            period_label=period_label, qoq_period=qoq_period, yoy_period=yoy_period,
        )

    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label.in_(labels),
            ExtractedMetric.segment != "guidance",
        ).order_by(ExtractedMetric.confidence.desc())
    )
    rows = q.scalars().all()

    # Bucket rows by (canonical metric_name, logical period).
    # `logical period` maps _FY rows onto their _Q4 twin so the dict
    # collapses `2024_Q4` and `2024_FY` into the same bucket.
    def _logical_period(p: Optional[str]) -> Optional[str]:
        if p and p.endswith("_FY"):
            return p[:-3] + "_Q4"
        return p

    current_logical = _logical_period(period_label)
    qoq_logical     = _logical_period(qoq_period)
    yoy_logical     = _logical_period(yoy_period)

    # per-metric per-logical-period: take the highest-confidence numeric
    # value we see (rows are pre-sorted desc by confidence).
    buckets: dict[str, dict[str, dict]] = {}
    display_names: dict[str, str] = {}
    units: dict[str, str] = {}

    for r in rows:
        if r.metric_value is None:
            continue
        # Skip rows with our misleading prefix — agents must not read
        # BRIDGE_GAP / BRIDGE_ADJ / GUIDANCE rows as temporal values.
        if r.metric_name and ":" in r.metric_name:
            prefix = r.metric_name.split(":", 1)[0].strip().upper()
            if prefix.startswith(("BRIDGE_GAP", "BRIDGE_ADJ", "GUIDANCE")):
                continue
        canonical = _normalise_metric_name(r.metric_name)
        if not canonical:
            continue
        try:
            val = float(r.metric_value)
        except (TypeError, ValueError):
            continue

        logical = _logical_period(r.period_label)
        bucket = buckets.setdefault(canonical, {})
        # First occurrence wins (highest confidence)
        if logical not in bucket:
            bucket[logical] = {"value": val, "source_period": r.period_label}
        if canonical not in display_names:
            display_names[canonical] = r.metric_name
            units[canonical] = r.unit or ""

    timelines: dict[str, MetricTimeline] = {}
    for canonical, periods in buckets.items():
        current_val  = periods.get(current_logical, {}).get("value")
        qoq_val      = periods.get(qoq_logical, {}).get("value")    if qoq_logical    else None
        yoy_val      = periods.get(yoy_logical, {}).get("value")    if yoy_logical    else None

        if current_val is None:
            # No current-period data: nothing useful to tell the agent.
            continue

        unit = units.get(canonical) or None

        tl = MetricTimeline(
            metric_name=canonical,
            display_name=display_names.get(canonical, canonical),
            unit=unit,
            current=current_val,
            qoq_prior=qoq_val,
            qoq_change=_format_change(current_val, qoq_val, unit) if qoq_val is not None else None,
            yoy_prior=yoy_val,
            yoy_change=_format_change(current_val, yoy_val, unit) if yoy_val is not None else None,
        )
        timelines[canonical] = tl

    # Reuse the orphaned compare_to_prior() for anomaly detection
    # against the QoQ period (direct prior quarter).
    current_items = [
        {"metric_name": r.metric_name, "metric_value": r.metric_value}
        for r in rows if _logical_period(r.period_label) == current_logical
    ]
    qoq_items = [
        {"metric_name": r.metric_name, "metric_value": r.metric_value}
        for r in rows if qoq_logical and _logical_period(r.period_label) == qoq_logical
    ]
    comparator = compare_to_prior(
        current_items, qoq_items,
        current_period=period_label, prior_period=qoq_period or "",
    ) if qoq_items else {"anomalies": [], "missing_in_current": [], "new_in_current": []}

    return MetricsHistoryResult(
        period_label=period_label,
        qoq_period=qoq_period,
        yoy_period=yoy_period,
        timelines=timelines,
        anomalies=comparator.get("anomalies", []),
        missing_vs_prior=comparator.get("missing_in_current", []),
    )


# ─────────────────────────────────────────────────────────────────
# Formatter for prompt injection
# ─────────────────────────────────────────────────────────────────

def format_for_prompt(result: MetricsHistoryResult) -> str:
    """Render the history result as a compact text block for the
    financial_analyst prompt. Pipe-separated columns, one metric per
    line — agents read this much more reliably than JSON."""
    if not result.timelines:
        return "No prior-period metrics available for comparison."
    lines = [
        f"Period: {result.period_label}  (QoQ={result.qoq_period or 'n/a'}, YoY={result.yoy_period or 'n/a'})",
        "metric | current | QoQ prior | QoQ change | YoY prior | YoY change",
    ]
    # Stable ordering: alphabetical on canonical name
    for name in sorted(result.timelines.keys()):
        tl = result.timelines[name]
        unit_s = f" {tl.unit}" if tl.unit else ""
        cur = f"{tl.current:g}{unit_s}" if tl.current is not None else "—"
        qp  = f"{tl.qoq_prior:g}{unit_s}" if tl.qoq_prior is not None else "—"
        yp  = f"{tl.yoy_prior:g}{unit_s}" if tl.yoy_prior is not None else "—"
        qc  = tl.qoq_change or "—"
        yc  = tl.yoy_change or "—"
        lines.append(f"{tl.display_name} | {cur} | {qp} | {qc} | {yp} | {yc}")
    if result.anomalies:
        lines.append("")
        lines.append("QoQ anomalies flagged (review carefully):")
        for a in result.anomalies[:5]:
            lines.append(
                f"  - {a.get('metric')}: {a.get('prior')} → {a.get('current')} "
                f"({a.get('change_pct')}% — {a.get('severity')})"
            )
    return "\n".join(lines)
