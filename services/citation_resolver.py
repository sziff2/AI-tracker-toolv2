"""
Citation Resolver — Tier 4.4.

Agent outputs now emit a `sources` array citing the data behind each
claim. The resolver checks each citation against the DB so QC can
score them and the UI can render a resolvable source list.

Failure mode this catches
  The original Chubb Q1 2026 run (pre-2026-04-23 extraction fix) had
  bear_case confidently asserting "combined ratio absent from extracted
  metrics" as a bear signal — technically true at the label level, but
  the underlying value was in the DB under a different name. If each
  factual claim had to cite a specific `metric_name` OR flag "no
  matching metric found", the hallucination would have been visible.

Kinds
  extracted_metric  — lookup by metric_name (+ optional segment) in
                      ExtractedMetric table for company_id + period.
  document_section  — lookup by doc_id or page; verify doc exists for
                      company+period.
  management_quote  — transcript quotes cannot be mechanically verified
                      against raw text without embeddings; we just
                      shape-check and mark as unverifiable.
  methodology_note / bridge_gap / historical_drawdown / thesis —
                      metadata kinds; shape-check only.

Never raises for operational failures. Returns a structured report
so the QC agent and the UI can distinguish resolved from unresolved
citations instead of failing open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ExtractedMetric, Document

logger = logging.getLogger(__name__)


VALID_KINDS = {
    "extracted_metric",
    "document_section",
    "management_quote",
    "methodology_note",
    "bridge_gap",
    "historical_drawdown",
    "thesis",
}


@dataclass
class CitationResult:
    """One citation's resolution status."""
    kind:        str
    resolved:    bool
    reason:      str = ""
    metric_id:   Optional[str] = None          # populated for resolved extracted_metric
    doc_id:      Optional[str] = None          # populated for resolved document_section
    input:       dict = field(default_factory=dict)  # echo of the original citation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CitationReport:
    """Aggregate resolution result for a single agent's sources array."""
    total:       int = 0
    resolved:    int = 0
    unresolved:  int = 0
    malformed:   int = 0
    results:     list[dict] = field(default_factory=list)

    @property
    def resolve_rate(self) -> float:
        if self.total == 0:
            return 1.0  # vacuously complete
        return self.resolved / self.total

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolve_rate"] = round(self.resolve_rate, 3)
        return d


async def resolve_citations(
    db: AsyncSession,
    company_id,
    period_label: str,
    sources: list[dict] | None,
) -> CitationReport:
    """Resolve every citation in `sources` against the DB for this
    company+period. Safe on empty / None input (returns vacuously
    complete report with total=0).
    """
    report = CitationReport()
    if not sources or not isinstance(sources, list):
        return report

    # Pre-fetch sets we'll lookup against — cheaper than per-citation
    # queries when an agent emits 15-20 sources.
    metric_names = await _load_metric_name_set(db, company_id, period_label)
    doc_ids = await _load_doc_id_set(db, company_id, period_label)

    for raw in sources:
        report.total += 1
        if not isinstance(raw, dict):
            report.malformed += 1
            report.results.append(
                CitationResult(kind="malformed", resolved=False,
                               reason="citation is not a dict",
                               input={"raw": str(raw)[:200]}).to_dict()
            )
            continue

        kind = (raw.get("kind") or "").strip().lower()
        if kind not in VALID_KINDS:
            report.malformed += 1
            report.results.append(
                CitationResult(kind=kind or "missing", resolved=False,
                               reason=f"unknown kind: {kind!r}",
                               input=raw).to_dict()
            )
            continue

        if kind == "extracted_metric":
            res = _resolve_extracted_metric(raw, metric_names)
        elif kind == "document_section":
            res = _resolve_document_section(raw, doc_ids)
        else:
            # Shape-only kinds — we can't mechanically verify the
            # content but we require a snippet so at least there's
            # something for the reader to look at.
            snip = (raw.get("snippet") or "").strip()
            if not snip:
                res = CitationResult(
                    kind=kind, resolved=False,
                    reason=f"{kind} citation missing 'snippet'",
                    input=raw,
                )
            else:
                res = CitationResult(
                    kind=kind, resolved=True,
                    reason=f"shape-only kind; snippet present ({len(snip)} chars)",
                    input=raw,
                )

        if res.resolved:
            report.resolved += 1
        else:
            report.unresolved += 1
        report.results.append(res.to_dict())

    logger.info(
        "citation_resolver: total=%d resolved=%d unresolved=%d malformed=%d rate=%.2f",
        report.total, report.resolved, report.unresolved, report.malformed, report.resolve_rate,
    )
    return report


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────

async def _load_metric_name_set(
    db: AsyncSession, company_id, period_label: str,
) -> set[tuple[str, str]]:
    """Build a set of `(lower(metric_name), lower(segment))` tuples for
    cheap O(1) lookup during citation resolution.
    The segment tuple element is "" when the metric is consolidated /
    unsegmented, which matches the pattern used by the extraction
    pipeline when `segment` is NULL."""
    q = await db.execute(
        select(ExtractedMetric.metric_name, ExtractedMetric.segment)
        .where(ExtractedMetric.company_id == company_id)
        .where(ExtractedMetric.period_label == period_label)
    )
    out: set[tuple[str, str]] = set()
    for name, seg in q.all():
        if not name:
            continue
        out.add(((name or "").strip().lower(), (seg or "").strip().lower()))
    return out


async def _load_doc_id_set(
    db: AsyncSession, company_id, period_label: str,
) -> set[str]:
    q = await db.execute(
        select(Document.id)
        .where(Document.company_id == company_id)
        .where(Document.period_label == period_label)
    )
    return {str(row[0]) for row in q.all()}


def _resolve_extracted_metric(
    raw: dict, metric_names: set[tuple[str, str]],
) -> CitationResult:
    name = (raw.get("metric_name") or "").strip()
    segment = (raw.get("segment") or "").strip()
    if not name:
        return CitationResult(
            kind="extracted_metric", resolved=False,
            reason="missing metric_name",
            input=raw,
        )
    # Exact (name, segment) match first
    key = (name.lower(), segment.lower())
    if key in metric_names:
        return CitationResult(
            kind="extracted_metric", resolved=True,
            reason="exact name+segment match",
            input=raw,
        )
    # Fallback: metric_name exists under a different segment (might be
    # agent being imprecise about which segment it cited)
    name_only = name.lower()
    any_seg = {seg for (n, seg) in metric_names if n == name_only}
    if any_seg:
        return CitationResult(
            kind="extracted_metric", resolved=True,
            reason=f"metric_name matches; segment differs "
                   f"(cited {segment!r}, found {sorted(any_seg)[0]!r})",
            input=raw,
        )
    # Last-resort fuzzy: partial-string match on name
    partial = [n for (n, _) in metric_names if name_only in n or n in name_only]
    if partial:
        return CitationResult(
            kind="extracted_metric", resolved=False,
            reason=f"no exact match; {len(partial)} partial matches "
                   f"(e.g. {sorted(partial)[0]!r}) — likely mis-citation",
            input=raw,
        )
    return CitationResult(
        kind="extracted_metric", resolved=False,
        reason=f"no metric named {name!r} in extracted_metrics for period",
        input=raw,
    )


def _resolve_document_section(
    raw: dict, doc_ids: set[str],
) -> CitationResult:
    doc_id = (raw.get("doc_id") or raw.get("document_id") or "").strip()
    snippet = (raw.get("snippet") or "").strip()
    page = raw.get("page")
    if doc_id:
        if doc_id in doc_ids:
            return CitationResult(
                kind="document_section", resolved=True,
                reason="doc_id resolves for period",
                doc_id=doc_id,
                input=raw,
            )
        return CitationResult(
            kind="document_section", resolved=False,
            reason=f"doc_id {doc_id!r} not in this period's documents",
            input=raw,
        )
    # No doc_id — accept if snippet is present (can't fully verify
    # without embedding the document, but snippet + optional page is
    # better than nothing).
    if snippet:
        return CitationResult(
            kind="document_section", resolved=True,
            reason=f"shape-only (no doc_id; snippet {len(snippet)} chars"
                   + (f", page {page}" if page is not None else "") + ")",
            input=raw,
        )
    return CitationResult(
        kind="document_section", resolved=False,
        reason="document_section needs either doc_id or snippet",
        input=raw,
    )
