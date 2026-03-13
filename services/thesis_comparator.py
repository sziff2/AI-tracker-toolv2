"""
Thesis Comparison Service — improved version.

Improvements:
  1. Handles all period types: Q1-Q4, HY, FY (not just quarterly)
  2. Uses raw document text alongside extracted metrics
  3. Uses structured context from context_builder
  4. Includes tracked KPIs and thesis risks in comparison
  5. Better prompt with explicit scoring framework
"""

import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Company, Document, DocumentSection, EventAssessment,
    ExtractedMetric, ReviewQueueItem, ThesisVersion,
)
from configs.settings import settings
from schemas import ThesisComparison
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Period logic — handles Q1-Q4, HY, FY
# ─────────────────────────────────────────────────────────────────

def _previous_period(period_label: str) -> str:
    """
    Derive the prior period label.
    Handles: 2025_Q1→2024_Q4, 2025_Q2→2025_Q1, 2025_HY→2024_HY,
             2025_FY→2024_FY, 2025_H1→2024_H1, etc.
    """
    if not period_label or "_" not in period_label:
        return ""
    try:
        parts = period_label.split("_", 1)
        year = int(parts[0])
        suffix = parts[1].upper()

        if suffix.startswith("Q"):
            q = int(suffix[1:])
            if q == 1:
                return f"{year - 1}_Q4"
            return f"{year}_Q{q - 1}"
        elif suffix in ("HY", "H1"):
            return f"{year - 1}_{suffix}"
        elif suffix == "H2":
            return f"{year}_H1"
        elif suffix == "FY":
            return f"{year - 1}_FY"
        else:
            return f"{year - 1}_{suffix}"
    except Exception:
        return ""


def _comparable_periods(period_label: str) -> list[str]:
    """
    Return a list of periods to compare against, in priority order.
    E.g. for 2025_Q4: try 2025_Q3 first, then 2024_Q4 (YoY), then any available.
    """
    if not period_label or "_" not in period_label:
        return []
    try:
        parts = period_label.split("_", 1)
        year = int(parts[0])
        suffix = parts[1].upper()

        candidates = []
        if suffix.startswith("Q"):
            q = int(suffix[1:])
            # Sequential quarter
            if q > 1:
                candidates.append(f"{year}_Q{q-1}")
            else:
                candidates.append(f"{year-1}_Q4")
            # Year-on-year
            candidates.append(f"{year-1}_Q{q}")
        elif suffix in ("HY", "H1"):
            candidates.append(f"{year-1}_{suffix}")
            candidates.append(f"{year-1}_H2")
        elif suffix == "H2":
            candidates.append(f"{year}_H1")
            candidates.append(f"{year-1}_H2")
        elif suffix == "FY":
            candidates.append(f"{year-1}_FY")
            candidates.append(f"{year-1}_H2")
        return candidates
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# Context assembly
# ─────────────────────────────────────────────────────────────────

async def _get_active_thesis(db: AsyncSession, company_id: uuid.UUID) -> ThesisVersion | None:
    result = await db.execute(
        select(ThesisVersion)
        .where(ThesisVersion.company_id == company_id, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_metrics_for_period(db: AsyncSession, company_id: uuid.UUID, period: str) -> list[ExtractedMetric]:
    result = await db.execute(
        select(ExtractedMetric)
        .where(ExtractedMetric.company_id == company_id, ExtractedMetric.period_label == period)
    )
    return list(result.scalars().all())


async def _find_best_prior_period(db: AsyncSession, company_id: uuid.UUID, current_period: str) -> tuple[str, list[ExtractedMetric]]:
    """Find the best available prior period with data."""
    candidates = _comparable_periods(current_period)
    for candidate in candidates:
        metrics = await _get_metrics_for_period(db, company_id, candidate)
        if metrics:
            return candidate, metrics
    return "", []


def _metrics_to_text(metrics: list[ExtractedMetric], max_items: int = 30) -> str:
    """Convert metrics to compressed key-value text."""
    seen = set()
    lines = []
    for m in metrics:
        if len(lines) >= max_items:
            break
        key = m.metric_name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
        lines.append(f"- {m.metric_name}: {val}")
    return "\n".join(lines) if lines else "No data available."


async def _get_document_summary(db: AsyncSession, company_id: uuid.UUID, period: str, max_chars: int = 3000) -> str:
    """
    Get a compressed summary of the raw document text for this period.
    Uses document sections stored in the DB — no filesystem dependency.
    """
    docs_q = await db.execute(
        select(Document).where(
            Document.company_id == company_id,
            Document.period_label == period,
        )
    )
    docs = docs_q.scalars().all()
    if not docs:
        return ""

    all_text = []
    for doc in docs:
        sections_q = await db.execute(
            select(DocumentSection.text_content)
            .where(DocumentSection.document_id == doc.id)
            .order_by(DocumentSection.page_number)
            .limit(10)  # First 10 pages — key content is usually up front
        )
        for row in sections_q.all():
            if row[0]:
                all_text.append(row[0])

    combined = "\n".join(all_text)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "…"
    return combined


async def _get_prior_analysis_summary(db: AsyncSession, company_id: uuid.UUID, current_period: str) -> str:
    """Get the bottom line from a prior analysis if available."""
    from apps.api.models import ResearchOutput
    candidates = _comparable_periods(current_period)
    for candidate in candidates:
        q = await db.execute(
            select(ResearchOutput).where(
                ResearchOutput.company_id == company_id,
                ResearchOutput.period_label == candidate,
                ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
            ).order_by(ResearchOutput.created_at.desc()).limit(1)
        )
        output = q.scalar_one_or_none()
        if output and output.content_json:
            try:
                data = json.loads(output.content_json)
                briefing = data.get("synthesis") or data.get("briefing")
                if isinstance(briefing, dict):
                    parts = []
                    if briefing.get("bottom_line"):
                        parts.append(briefing["bottom_line"])
                    if briefing.get("what_happened"):
                        parts.append(briefing["what_happened"][:300])
                    if parts:
                        return f"Prior period ({candidate}) summary:\n" + "\n".join(parts)
            except Exception:
                pass
    return ""


# ─────────────────────────────────────────────────────────────────
# Improved prompt
# ─────────────────────────────────────────────────────────────────

THESIS_COMPARATOR_V2 = """\
You are a senior buy-side equity analyst assessing whether new results
strengthen or weaken the investment thesis.

INVESTMENT THESIS:
{thesis}

KEY RISKS FROM THESIS:
{thesis_risks}

TRACKED KPIs (what the analyst cares about most):
{tracked_kpis}

=== NEW PERIOD DATA ({current_period}) ===

Key metrics:
{current_metrics}

Document highlights:
{document_summary}

=== PRIOR PERIOD DATA ({prior_period}) ===

Key metrics:
{prior_metrics}

Prior analysis summary:
{prior_analysis}

=== INSTRUCTIONS ===

Assess the thesis through three lenses:
1. FUNDAMENTALS — are the numbers confirming or contradicting the thesis?
2. NARRATIVE — is management's story consistent with your thesis? Any credibility concerns?
3. RISKS — are thesis risks materialising, receding, or are there new ones?

Weight your assessment by what matters most to the thesis, not just what changed most.
If a core thesis pillar is weakening, that matters more than a minor beat on a secondary metric.

Respond ONLY with a JSON object. No preamble, no markdown fences.

Schema:
{{
  "thesis_direction": "strengthened" | "weakened" | "unchanged",
  "confidence": <0.0-1.0>,
  "supporting_signals": ["<specific evidence that supports the thesis>"],
  "weakening_signals": ["<specific evidence that challenges the thesis>"],
  "new_risks": ["<risks not in the original thesis>"],
  "risks_receding": ["<thesis risks that appear to be diminishing>"],
  "unresolved_questions": ["<key questions that remain open>"],
  "summary": "<3-4 sentence assessment — be direct and opinionated>"
}}
"""


# ─────────────────────────────────────────────────────────────────
# Main comparison function
# ─────────────────────────────────────────────────────────────────

async def compare_thesis(
    db: AsyncSession,
    company_id: uuid.UUID,
    document_id: uuid.UUID,
    period_label: str,
) -> ThesisComparison:
    """
    Run the thesis comparator with rich context and persist an EventAssessment.
    """
    thesis = await _get_active_thesis(db, company_id)
    if thesis is None:
        raise ValueError("No active thesis found for this company.")

    # Current period data
    current_metrics = await _get_metrics_for_period(db, company_id, period_label)

    # Best available prior period
    prior_period, prior_metrics = await _find_best_prior_period(db, company_id, period_label)

    # Raw document summary (first 10 pages, compressed)
    doc_summary = await _get_document_summary(db, company_id, period_label)

    # Prior analysis summary (layered — uses stored output, not raw text)
    prior_analysis = await _get_prior_analysis_summary(db, company_id, period_label)

    # Tracked KPIs context
    tracked_kpis_text = ""
    try:
        from services.context_builder import build_tracked_kpi_context
        tracked_kpis_text = await build_tracked_kpi_context(db, company_id)
    except Exception:
        pass

    # Build thesis risks text
    thesis_risks = thesis.key_risks or "No specific risks documented."

    prompt = THESIS_COMPARATOR_V2.format(
        thesis=thesis.core_thesis,
        thesis_risks=thesis_risks,
        tracked_kpis=tracked_kpis_text or "No tracked KPIs set up.",
        current_period=period_label,
        current_metrics=_metrics_to_text(current_metrics),
        document_summary=doc_summary[:2000] if doc_summary else "No document text available.",
        prior_period=prior_period or "N/A",
        prior_metrics=_metrics_to_text(prior_metrics),
        prior_analysis=prior_analysis or "No prior analysis available.",
    )

    data = call_llm_json(prompt)
    comparison = ThesisComparison(**data)

    # Persist assessment
    assessment = EventAssessment(
        id=uuid.uuid4(),
        company_id=company_id,
        document_id=document_id,
        event_type="earnings",
        thesis_direction=comparison.thesis_direction,
        surprise_level="minor",  # refined by surprise detector
        summary=comparison.summary,
        confidence=data.get("confidence", 0.85),
        needs_review=True,
    )
    db.add(assessment)

    db.add(ReviewQueueItem(
        id=uuid.uuid4(),
        entity_type="assessment",
        entity_id=assessment.id,
        queue_reason="Thesis comparison requires human review",
        priority="high",
    ))

    await db.commit()
    logger.info("Thesis comparison for %s / %s → %s (confidence: %s)",
                company_id, period_label, comparison.thesis_direction,
                data.get("confidence", "N/A"))

    # Write drift JSON
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one()
    drift_dir = Path(settings.storage_base_path) / "outputs" / company.ticker / period_label
    drift_dir.mkdir(parents=True, exist_ok=True)
    (drift_dir / "thesis_drift.json").write_text(comparison.model_dump_json(indent=2))

    return comparison
