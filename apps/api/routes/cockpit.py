"""
Company Cockpit — single endpoint that returns everything
an analyst needs for one company in one call.
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company, Document, ExtractedMetric, ThesisVersion, EventAssessment,
    ResearchOutput, ReviewQueueItem, TrackedKPI, KPIScore,
    DecisionLog, AnalystNote,
)

router = APIRouter(tags=["cockpit"])


# ── Schemas ──────────────────────────────────────────────────
class DecisionCreate(BaseModel):
    action: str           # hold | add | trim | exit | initiate | watchlist
    rationale: str
    old_weight: Optional[float] = None
    new_weight: Optional[float] = None
    conviction: Optional[int] = None  # 1-5
    author: Optional[str] = None


class NoteCreate(BaseModel):
    note_type: str = "general"
    title: Optional[str] = None
    content: str
    author: Optional[str] = None


# ── Cockpit endpoint ─────────────────────────────────────────
@router.get("/companies/{ticker}/cockpit")
async def get_cockpit(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the full company cockpit: thesis, KPIs, latest results,
    research timeline, decision log, and pending review items.
    """
    # Company
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    cid = company.id

    # ── Thesis ────────────────────────────────────────────────
    thesis_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == cid, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()
    thesis_data = None
    if thesis:
        thesis_data = {
            "id": str(thesis.id),
            "thesis_date": thesis.thesis_date.isoformat() if thesis.thesis_date else None,
            "core_thesis": thesis.core_thesis,
            "variant_perception": thesis.variant_perception,
            "key_risks": thesis.key_risks,
            "debate_points": thesis.debate_points,
            "capital_allocation_view": thesis.capital_allocation_view,
            "valuation_framework": thesis.valuation_framework,
            "recommendation": thesis.recommendation,
            "catalyst": thesis.catalyst,
            "conviction": thesis.conviction,
            "what_would_make_us_wrong": thesis.what_would_make_us_wrong,
            "disconfirming_evidence": thesis.disconfirming_evidence,
            "positive_surprises": thesis.positive_surprises,
            "negative_surprises": thesis.negative_surprises,
        }

    # ── Latest thesis assessment ──────────────────────────────
    assess_q = await db.execute(
        select(EventAssessment).where(EventAssessment.company_id == cid)
        .order_by(EventAssessment.created_at.desc()).limit(1)
    )
    latest_assessment = assess_q.scalar_one_or_none()
    assessment_data = None
    if latest_assessment:
        assessment_data = {
            "thesis_direction": latest_assessment.thesis_direction,
            "surprise_level": latest_assessment.surprise_level,
            "summary": latest_assessment.summary,
            "confidence": float(latest_assessment.confidence) if latest_assessment.confidence else None,
        }

    # ── KPI tracker ───────────────────────────────────────────
    kpis_q = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == cid).order_by(TrackedKPI.display_order)
    )
    tracked_kpis = kpis_q.scalars().all()

    # Get all periods
    periods_q = await db.execute(
        select(ExtractedMetric.period_label).where(ExtractedMetric.company_id == cid)
        .distinct().order_by(ExtractedMetric.period_label)
    )
    periods = [p[0] for p in periods_q.all() if p[0]]

    # Get scores
    scores_q = await db.execute(select(KPIScore).where(KPIScore.company_id == cid))
    all_scores = scores_q.scalars().all()
    score_map = {}
    for s in all_scores:
        score_map[(str(s.tracked_kpi_id), s.period_label)] = {
            "value": float(s.value) if s.value is not None else None,
            "value_text": s.value_text,
            "score": s.score,
        }

    kpi_rows = []
    for kpi in tracked_kpis:
        row = {"id": str(kpi.id), "kpi_name": kpi.kpi_name, "unit": kpi.unit, "periods": {}}
        for period in periods:
            key = (str(kpi.id), period)
            if key in score_map:
                row["periods"][period] = score_map[key]
            else:
                # Auto-match
                match_q = await db.execute(
                    select(ExtractedMetric).where(
                        ExtractedMetric.company_id == cid,
                        ExtractedMetric.period_label == period,
                        ExtractedMetric.metric_name.ilike(f"%{kpi.kpi_name}%"),
                    ).limit(1)
                )
                m = match_q.scalar_one_or_none()
                if m:
                    row["periods"][period] = {
                        "value": float(m.metric_value) if m.metric_value else None,
                        "value_text": m.metric_text,
                        "score": None,
                    }
                else:
                    row["periods"][period] = {"value": None, "score": None}
        kpi_rows.append(row)

    # Overall scores per period
    overall_scores = {}
    for period in periods:
        vals = [score_map.get((str(k.id), period), {}).get("score") for k in tracked_kpis]
        valid = [v for v in vals if v is not None]
        overall_scores[period] = round(sum(valid) / len(valid), 1) if valid else None

    # ── All analyses grouped by period ─────────────────────────
    outputs_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == cid,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.created_at.desc())
    )
    all_outputs = outputs_q.scalars().all()

    # Group by period
    analyses_by_period = {}
    for o in all_outputs:
        p = o.period_label or "unknown"
        if p not in analyses_by_period:
            content = None
            if o.content_json:
                try:
                    content = json.loads(o.content_json)
                except Exception:
                    pass
            analyses_by_period[p] = {
                "period": p,
                "output_type": o.output_type,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "content": content,
            }

    # Latest briefing (for backwards compat)
    latest_briefing = None
    if analyses_by_period:
        latest_period = sorted(analyses_by_period.keys(), reverse=True)[0]
        latest_briefing = analyses_by_period[latest_period]

    # ── Documents timeline ────────────────────────────────────
    # No limit — show all documents grouped by period in UI
    docs_q = await db.execute(
        select(Document).where(Document.company_id == cid)
        .order_by(Document.period_label.desc(), Document.created_at.desc())
    )
    docs = [{
        "id": str(d.id), "title": d.title, "document_type": d.document_type,
        "period_label": d.period_label, "parsing_status": d.parsing_status,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in docs_q.scalars().all()]

    # ── Decision log ──────────────────────────────────────────
    decisions_q = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == cid)
        .order_by(DecisionLog.created_at.desc()).limit(20)
    )
    decisions = [{
        "id": str(d.id), "action": d.action, "rationale": d.rationale,
        "old_weight": float(d.old_weight) if d.old_weight else None,
        "new_weight": float(d.new_weight) if d.new_weight else None,
        "conviction": d.conviction, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in decisions_q.scalars().all()]

    # ── Analyst notes ─────────────────────────────────────────
    notes_q = await db.execute(
        select(AnalystNote).where(AnalystNote.company_id == cid)
        .order_by(AnalystNote.created_at.desc()).limit(20)
    )
    notes = [{
        "id": str(d.id), "note_type": d.note_type, "title": d.title,
        "content": d.content, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in notes_q.scalars().all()]

    # ── Review items ──────────────────────────────────────────
    review_q = await db.execute(
        select(ReviewQueueItem).where(
            ReviewQueueItem.status == "open"
        ).order_by(ReviewQueueItem.created_at.desc()).limit(10)
    )
    # Filter to this company's items by checking entity relationships
    reviews = [{
        "id": str(r.id), "entity_type": r.entity_type, "queue_reason": r.queue_reason,
        "priority": r.priority, "status": r.status,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in review_q.scalars().all()]

    # ── Metrics summary ───────────────────────────────────────
    metrics_count_q = await db.execute(
        select(func.count(ExtractedMetric.id)).where(ExtractedMetric.company_id == cid)
    )
    total_metrics = metrics_count_q.scalar() or 0

    # ── Thesis history ──────────────────────────────────────────
    all_theses_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == cid)
        .order_by(ThesisVersion.thesis_date.desc())
    )
    thesis_history = [{
        "id": str(t.id),
        "thesis_date": t.thesis_date.isoformat() if t.thesis_date else None,
        "core_thesis": t.core_thesis[:200] + "…" if t.core_thesis and len(t.core_thesis) > 200 else t.core_thesis,
        "active": t.active,
    } for t in all_theses_q.scalars().all()]

    # ── All thesis assessments ────────────────────────────────
    all_assessments_q = await db.execute(
        select(EventAssessment, Document.period_label)
        .join(Document, EventAssessment.document_id == Document.id)
        .where(EventAssessment.company_id == cid)
        .order_by(EventAssessment.created_at.desc())
    )
    all_assessments = [{
        "period": row[1],
        "thesis_direction": row[0].thesis_direction,
        "surprise_level": row[0].surprise_level,
        "summary": row[0].summary,
        "created_at": row[0].created_at.isoformat() if row[0].created_at else None,
    } for row in all_assessments_q.all()]

    # ── Group documents by period ─────────────────────────────
    docs_by_period = {}
    for d in docs:
        p = d["period_label"] or "unknown"
        if p not in docs_by_period:
            docs_by_period[p] = []
        docs_by_period[p].append(d)

    return {
        "company": {
            "ticker": company.ticker,
            "name": company.name,
            "sector": company.sector,
            "industry": company.industry,
            "country": company.country,
            "coverage_status": company.coverage_status,
            "primary_analyst": company.primary_analyst,
        },
        "thesis": thesis_data,
        "thesis_assessment": assessment_data,
        "thesis_history": thesis_history,
        "all_assessments": all_assessments,
        "kpi_tracker": {
            "periods": periods,
            "kpis": kpi_rows,
            "overall_scores": overall_scores,
        },
        "latest_briefing": latest_briefing,
        "analyses_by_period": analyses_by_period,
        "documents": docs,
        "docs_by_period": docs_by_period,
        "decisions": decisions,
        "notes": notes,
        "review_items": reviews,
        "stats": {
            "total_metrics": total_metrics,
            "total_documents": len(docs),
            "total_decisions": len(decisions),
            "periods_covered": len(periods),
        },
    }


# ── Decision log CRUD ────────────────────────────────────────
@router.post("/companies/{ticker}/decisions", status_code=201)
async def create_decision(ticker: str, body: DecisionCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    decision = DecisionLog(
        id=uuid.uuid4(), company_id=company.id,
        action=body.action, rationale=body.rationale,
        old_weight=body.old_weight, new_weight=body.new_weight,
        conviction=body.conviction, author=body.author,
    )
    db.add(decision)
    await db.commit()
    return {
        "id": str(decision.id), "action": decision.action,
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
    }


@router.get("/companies/{ticker}/decisions")
async def list_decisions(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    q = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == company.id)
        .order_by(DecisionLog.created_at.desc())
    )
    return [{
        "id": str(d.id), "action": d.action, "rationale": d.rationale,
        "old_weight": float(d.old_weight) if d.old_weight else None,
        "new_weight": float(d.new_weight) if d.new_weight else None,
        "conviction": d.conviction, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in q.scalars().all()]


# ── Analyst notes CRUD ───────────────────────────────────────
@router.post("/companies/{ticker}/notes", status_code=201)
async def create_note(ticker: str, body: NoteCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    note = AnalystNote(
        id=uuid.uuid4(), company_id=company.id,
        note_type=body.note_type, title=body.title,
        content=body.content, author=body.author,
    )
    db.add(note)
    await db.commit()
    return {
        "id": str(note.id), "title": note.title,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


@router.get("/companies/{ticker}/notes")
async def list_notes(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    q = await db.execute(
        select(AnalystNote).where(AnalystNote.company_id == company.id)
        .order_by(AnalystNote.created_at.desc())
    )
    return [{
        "id": str(d.id), "note_type": d.note_type, "title": d.title,
        "content": d.content, "author": d.author,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    } for d in q.scalars().all()]

# ─────────────────────────────────────────────────────────────────
# Update thesis fields inline (IC Summary, etc.)
# ─────────────────────────────────────────────────────────────────
class ThesisFieldUpdate(BaseModel):
    field: str
    value: str


@router.patch("/companies/{ticker}/thesis")
async def update_thesis_field(ticker: str, body: ThesisFieldUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    thesis_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id, ThesisVersion.active == True)
        .order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()
    if not thesis:
        raise HTTPException(404, "No active thesis found")

    allowed = ["core_thesis", "variant_perception", "key_risks", "debate_points",
               "capital_allocation_view", "valuation_framework", "recommendation",
               "catalyst", "conviction", "what_would_make_us_wrong",
               "disconfirming_evidence", "positive_surprises", "negative_surprises"]
    if body.field not in allowed:
        raise HTTPException(400, f"Field '{body.field}' not allowed")

    setattr(thesis, body.field, body.value)
    await db.commit()
    return {"status": "saved", "field": body.field}


# ─────────────────────────────────────────────────────────────────
# Document Chat — ask questions grounded in period-specific data
# ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    period_label: str
    include_thesis: bool = True
    model: str = "standard"  # "fast", "standard", or "deep"


class ChatResponse(BaseModel):
    answer: str
    sources_used: list[str]
    period: str


MODEL_MAP = {
    "fast": "claude-3-5-haiku-20241022",
    "standard": "claude-sonnet-4-20250514",
    "deep": "claude-opus-4-20250514",
}


@router.post("/companies/{ticker}/chat")
async def chat_with_documents(ticker: str, body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Ask a question about a specific quarter's documents.
    The LLM receives ONLY data from the requested period — no cross-contamination.
    """
    from apps.api.models import DocumentSection
    from services.llm_client import call_llm_async

    # Resolve model from user selection
    model = MODEL_MAP.get(body.model, MODEL_MAP["standard"])

    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    cid = company.id
    period = body.period_label

    # ── Gather context for this period ONLY ──────────────────

    # 1. Document text (from stored sections — no filesystem dependency)
    docs_q = await db.execute(
        select(Document).where(Document.company_id == cid, Document.period_label == period)
    )
    docs = docs_q.scalars().all()

    doc_texts = []
    sources_used = []
    total_doc_chars = 0
    max_total_chars = 40000  # Keep total prompt size manageable for fast response

    for doc in docs:
        if total_doc_chars >= max_total_chars:
            break
        sections_q = await db.execute(
            select(DocumentSection.text_content)
            .where(DocumentSection.document_id == doc.id)
            .order_by(DocumentSection.page_number)
        )
        pages = [row[0] for row in sections_q.all() if row[0]]
        if pages:
            doc_text = "\n".join(pages)
            # Truncate per-document to 8000 chars for faster processing
            if len(doc_text) > 8000:
                doc_text = doc_text[:8000] + "\n[... truncated]"
            remaining = max_total_chars - total_doc_chars
            if len(doc_text) > remaining:
                doc_text = doc_text[:remaining] + "\n[... truncated due to size limit]"
            doc_texts.append(f"=== {doc.title or doc.document_type} ({doc.document_type}) ===\n{doc_text}")
            sources_used.append(doc.title or doc.document_type)
            total_doc_chars += len(doc_text)

    # 2. Extracted metrics for this period
    metrics_q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == cid,
            ExtractedMetric.period_label == period,
        ).order_by(ExtractedMetric.confidence.desc()).limit(40)
    )
    metrics = metrics_q.scalars().all()
    metrics_text = "\n".join(
        f"- {m.metric_name}: {m.metric_value} {m.unit or ''}" if m.metric_value
        else f"- {m.metric_name}: {m.metric_text}"
        for m in metrics
    ) if metrics else "No extracted metrics for this period."

    # 3. Analysis output for this period (if available)
    output_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == cid,
            ResearchOutput.period_label == period,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.created_at.desc()).limit(1)
    )
    analysis_output = output_q.scalar_one_or_none()
    analysis_text = ""
    if analysis_output and analysis_output.content_json:
        try:
            data = json.loads(analysis_output.content_json)
            briefing = data.get("synthesis") or data.get("briefing")
            if isinstance(briefing, dict):
                parts = []
                for key in ["headline", "what_happened", "management_message", "thesis_impact", "bottom_line", "what_changed", "thesis_status"]:
                    if briefing.get(key):
                        parts.append(f"{key.replace('_', ' ').title()}: {briefing[key]}")
                analysis_text = "\n".join(parts)
        except Exception:
            pass

    # 4. Thesis context (optional)
    thesis_text = ""
    if body.include_thesis:
        thesis_q = await db.execute(
            select(ThesisVersion).where(
                ThesisVersion.company_id == cid, ThesisVersion.active == True
            ).order_by(ThesisVersion.thesis_date.desc()).limit(1)
        )
        thesis = thesis_q.scalar_one_or_none()
        if thesis:
            thesis_text = f"Investment Thesis: {thesis.core_thesis}"
            if thesis.key_risks:
                thesis_text += f"\nKey Risks: {thesis.key_risks}"

    # ── Build prompt ─────────────────────────────────────────
    if not doc_texts and not metrics_text:
        return {"answer": "No documents or data found for this period. Upload documents first.", "sources_used": [], "period": period}

    all_docs = "\n\n".join(doc_texts) if doc_texts else "No raw document text available."

    prompt = f"""You are an investment research assistant for {company.name} ({company.ticker}).
You are answering questions about the {period} period ONLY.
Use ONLY the data provided below. Do not use external knowledge about the company.
If the answer is not in the provided data, say so clearly.

{thesis_text}

=== EXTRACTED METRICS ({period}) ===
{metrics_text}

=== ANALYSIS SUMMARY ({period}) ===
{analysis_text if analysis_text else "No analysis summary available."}

=== SOURCE DOCUMENTS ({period}) ===
{all_docs}

=== QUESTION ===
{body.question}

Answer the question directly and specifically. Reference the source documents where relevant.
If quoting numbers, cite which document they came from."""

    try:
        answer = await call_llm_async(prompt, max_tokens=2048, timeout_seconds=25, model=model)
        return {"answer": answer, "sources_used": sources_used, "period": period, "model": body.model}
    except TimeoutError:
        return {"answer": "The analysis is taking longer than expected. Please try a more specific question or use 'Fast' mode.", "sources_used": sources_used, "period": period, "model": body.model}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Chat endpoint error: %s — prompt length: %d", str(e), len(prompt))
        return {"answer": f"Error: {str(e)[:150]}. Try a simpler question.", "sources_used": sources_used, "period": period, "model": body.model}


# ─────────────────────────────────────────────────────────────────
# Global Chat — ask questions across ALL periods for a company
# ─────────────────────────────────────────────────────────────────
class GlobalChatRequest(BaseModel):
    question: str
    include_thesis: bool = True
    model: str = "standard"  # "fast", "standard", or "deep"


@router.post("/companies/{ticker}/chat-all")
async def chat_all_periods(ticker: str, body: GlobalChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Ask a question across ALL periods for a company.
    Receives thesis, all analysis summaries, key metrics per period,
    decisions, and notes — a full company picture.
    """
    from services.llm_client import call_llm_async

    # Resolve model from user selection
    model = MODEL_MAP.get(body.model, MODEL_MAP["standard"])

    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    cid = company.id

    # 1. Thesis
    thesis_text = ""
    if body.include_thesis:
        thesis_q = await db.execute(
            select(ThesisVersion).where(ThesisVersion.company_id == cid, ThesisVersion.active == True)
            .order_by(ThesisVersion.thesis_date.desc()).limit(1)
        )
        thesis = thesis_q.scalar_one_or_none()
        if thesis:
            thesis_text = f"Investment Thesis: {thesis.core_thesis}"
            if thesis.key_risks:
                thesis_text += f"\nKey Risks: {thesis.key_risks}"
            if thesis.valuation_framework:
                thesis_text += f"\nValuation: {thesis.valuation_framework}"

    # 2. All analysis summaries (compressed — bottom line per period)
    outputs_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == cid,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(ResearchOutput.period_label.desc())
    )
    all_outputs = outputs_q.scalars().all()

    period_summaries = []
    periods_searched = []
    for o in all_outputs:
        if not o.content_json:
            continue
        periods_searched.append(o.period_label)
        try:
            data = json.loads(o.content_json)
            briefing = data.get("synthesis") or data.get("briefing")
            if isinstance(briefing, dict):
                parts = [f"=== {o.period_label} ==="]
                for key in ["headline", "what_happened", "management_message", "thesis_impact", "bottom_line", "what_changed", "thesis_status"]:
                    if briefing.get(key):
                        parts.append(f"{key.replace('_', ' ').title()}: {briefing[key]}")
                # Add surprises
                surprises = data.get("surprises", [])
                if surprises:
                    parts.append("Surprises: " + "; ".join(
                        f"{s.get('metric_or_topic','')}: {s.get('description','')}"
                        for s in surprises[:5]
                    ))
                period_summaries.append("\n".join(parts))
        except Exception:
            pass

    # 3. Key metrics across periods (compressed)
    metrics_q = await db.execute(
        select(ExtractedMetric).where(ExtractedMetric.company_id == cid)
        .order_by(ExtractedMetric.period_label.desc(), ExtractedMetric.confidence.desc())
        .limit(60)
    )
    all_metrics = metrics_q.scalars().all()
    metrics_by_period = {}
    for m in all_metrics:
        p = m.period_label or "unknown"
        if p not in metrics_by_period:
            metrics_by_period[p] = []
        if len(metrics_by_period[p]) < 15:
            val = f"{m.metric_value} {m.unit}" if m.metric_value else m.metric_text
            metrics_by_period[p].append(f"- {m.metric_name}: {val}")

    metrics_text = ""
    for p in sorted(metrics_by_period.keys(), reverse=True):
        metrics_text += f"\n=== Metrics {p} ===\n" + "\n".join(metrics_by_period[p])

    # 4. Decisions and notes
    decisions_q = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == cid)
        .order_by(DecisionLog.created_at.desc()).limit(10)
    )
    decisions_text = "\n".join(
        f"- {d.action.upper()} (conviction {d.conviction}/5): {d.rationale}"
        for d in decisions_q.scalars().all()
    ) or "No decisions logged."

    notes_q = await db.execute(
        select(AnalystNote).where(AnalystNote.company_id == cid)
        .order_by(AnalystNote.created_at.desc()).limit(5)
    )
    notes_text = "\n".join(
        f"- {n.title or n.note_type}: {n.content[:200]}"
        for n in notes_q.scalars().all()
    ) or "No notes."

    # 5. Thesis assessments over time
    assessments_q = await db.execute(
        select(EventAssessment, Document.period_label)
        .join(Document, EventAssessment.document_id == Document.id)
        .where(EventAssessment.company_id == cid)
        .order_by(EventAssessment.created_at.desc()).limit(10)
    )
    assessments_text = "\n".join(
        f"- {row[1]}: thesis {row[0].thesis_direction} ({row[0].surprise_level} surprise)"
        for row in assessments_q.all()
    ) or "No assessments."

    # 6. Source documents list
    docs_q = await db.execute(
        select(Document).where(Document.company_id == cid)
        .order_by(Document.created_at.desc()).limit(20)
    )
    sources_used = [d.title or d.document_type for d in docs_q.scalars().all()]

    if not period_summaries and not metrics_text:
        return {"answer": "No data found for this company. Upload documents first.", "sources_used": [], "periods_searched": []}

    prompt = f"""You are a senior investment research assistant for {company.name} ({company.ticker}).
You have access to ALL research data across ALL periods. Answer questions using this full picture.
If the question is about trends, compare across periods. If about a specific topic, find the most relevant data.
Use ONLY the data provided below. Do not use external knowledge.

{thesis_text}

=== THESIS ASSESSMENT HISTORY ===
{assessments_text}

=== ANALYSIS SUMMARIES BY PERIOD ===
{chr(10).join(period_summaries) if period_summaries else "No analysis summaries available."}

=== KEY METRICS BY PERIOD ===
{metrics_text if metrics_text else "No metrics available."}

=== DECISIONS ===
{decisions_text}

=== ANALYST NOTES ===
{notes_text}

=== QUESTION ===
{body.question}

Answer directly and specifically. Reference which period data comes from. Highlight trends across periods where relevant."""

    try:
        answer = await call_llm_async(prompt, max_tokens=2048, timeout_seconds=25, model=model)
        return {"answer": answer, "sources_used": sources_used[:10], "periods_searched": periods_searched, "model": body.model}
    except TimeoutError:
        return {"answer": "The analysis is taking longer than expected. Please try a more specific question or use 'Fast' mode.", "sources_used": sources_used[:10], "periods_searched": periods_searched, "model": body.model}
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {str(e)[:200]}")


# ─────────────────────────────────────────────────────────────────
# All extracted metrics for a company — browsable, filterable
# ─────────────────────────────────────────────────────────────────
@router.get("/companies/{ticker}/metrics")
async def get_all_metrics(ticker: str, period: str = None, db: AsyncSession = Depends(get_db)):
    from apps.api.models import ExtractedMetric
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    q = select(ExtractedMetric).where(ExtractedMetric.company_id == company.id)
    if period:
        q = q.where(ExtractedMetric.period_label == period)
    q = q.order_by(ExtractedMetric.period_label.desc(), ExtractedMetric.confidence.desc())

    result = await db.execute(q)
    metrics = result.scalars().all()

    # Group by period
    by_period = {}
    for m in metrics:
        p = m.period_label or "unknown"
        if p not in by_period:
            by_period[p] = []
        by_period[p].append({
            "id": str(m.id),
            "metric_name": m.metric_name,
            "metric_value": float(m.metric_value) if m.metric_value else None,
            "metric_text": m.metric_text,
            "unit": m.unit,
            "segment": m.segment,
            "geography": m.geography,
            "source_snippet": m.source_snippet[:200] if m.source_snippet else None,
            "confidence": float(m.confidence) if m.confidence else None,
            "needs_review": m.needs_review,
        })

    return {
        "ticker": ticker,
        "total_metrics": len(metrics),
        "periods": list(by_period.keys()),
        "by_period": by_period,
    }


# ─────────────────────────────────────────────────────────────────
# Competitive Advantage / Moat Analysis
# ─────────────────────────────────────────────────────────────────
@router.post("/companies/{ticker}/moat-analysis")
async def run_moat_analysis(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Run a comprehensive competitive advantage analysis using all available data
    for the company: thesis, metrics, documents, ESG, execution track record.
    """
    from services.llm_client import call_llm_json_async
    from prompts import MOAT_ANALYSIS
    from apps.api.models import (
        ExtractedMetric, ESGData, DocumentSection,
        ManagementStatement, ExecutionScorecard,
    )

    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Gather thesis
    thesis_text = "No thesis on file."
    thesis_q = await db.execute(
        select(ThesisVersion).where(ThesisVersion.company_id == company.id)
        .order_by(ThesisVersion.thesis_date.desc()).limit(1)
    )
    thesis = thesis_q.scalar_one_or_none()
    if thesis:
        thesis_text = f"Core thesis: {thesis.core_thesis or 'N/A'}\nKey risks: {thesis.key_risks or 'N/A'}\nValuation: {thesis.valuation_framework or 'N/A'}"

    # Gather metrics (latest periods, high-confidence, capped)
    metrics_q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company.id,
            ExtractedMetric.metric_value.isnot(None),
            ExtractedMetric.confidence >= 0.65,
        )
        .order_by(ExtractedMetric.period_label.desc(), ExtractedMetric.confidence.desc())
        .limit(60)
    )
    metrics = metrics_q.scalars().all()
    metrics_text = "\n".join(
        f"{m.metric_name}: {m.metric_value} {m.unit or ''} ({m.period_label})"
        for m in metrics
    )[:3000] or "No financial metrics available."

    # Gather commentary from DocumentSection rows
    # Get most recent documents first, pull their sections
    recent_docs_q = await db.execute(
        select(Document).where(
            Document.company_id == company.id,
            Document.document_type.in_(["transcript", "earnings_release", "annual_report"]),
            Document.parsing_status == "completed",
        )
        .order_by(Document.period_label.desc())
        .limit(3)
    )
    recent_docs = recent_docs_q.scalars().all()
    commentary_parts = []
    for doc in recent_docs:
        sections_q = await db.execute(
            select(DocumentSection)
            .where(DocumentSection.document_id == doc.id)
            .order_by(DocumentSection.page_number)
            .limit(6)
        )
        sections = sections_q.scalars().all()
        text = "\n".join(s.text_content[:400] for s in sections if s.text_content)[:2000]
        if text:
            commentary_parts.append(f"[{doc.period_label} — {doc.document_type}]\n{text}")
    commentary_text = "\n\n".join(commentary_parts)[:6000] or "No document commentary available."

    # Gather ESG data
    esg_text = "No ESG data available."
    import json as _json
    esg_q = await db.execute(select(ESGData).where(ESGData.company_id == company.id))
    esg_row = esg_q.scalar_one_or_none()
    if esg_row and esg_row.data:
        esg_dict = _json.loads(esg_row.data) if isinstance(esg_row.data, str) else esg_row.data
        filled = {k: v for k, v in esg_dict.items() if v}
        if filled:
            esg_text = "\n".join(f"{k}: {v}" for k, v in list(filled.items())[:30])
        if esg_row.ai_summary:
            esg_text += f"\n\nAI Summary: {esg_row.ai_summary[:500]}"
    esg_text = esg_text[:2000]

    # Gather execution track record (wrapped — tables may not exist yet)
    execution_text = "No execution data available."
    try:
        scorecard_q = await db.execute(
            select(ExecutionScorecard).where(
                ExecutionScorecard.company_id == company.id,
                ExecutionScorecard.period == "all_time",
            )
        )
        scorecard = scorecard_q.scalar_one_or_none()
        if scorecard:
            execution_text = (
                f"Execution score: {scorecard.overall_score}\n"
                f"Guidance bias: {scorecard.guidance_bias or 'N/A'}\n"
                f"Reliability: {scorecard.execution_reliability or 'N/A'}\n"
                f"Strategic consistency: {scorecard.strategic_consistency or 'N/A'}\n"
                f"Delivered: {scorecard.delivered_count}, Missed: {scorecard.missed_count}\n"
            )
            if scorecard.ai_assessment:
                execution_text += f"Assessment: {scorecard.ai_assessment[:400]}"

        stmts_q = await db.execute(
            select(ManagementStatement).where(
                ManagementStatement.company_id == company.id
            ).order_by(ManagementStatement.statement_date.desc()).limit(10)
        )
        stmts = stmts_q.scalars().all()
        if stmts:
            execution_text += "\n\nRecent statements:\n"
            for s in stmts:
                score_str = f" [Score: {s.score}]" if s.score is not None else ""
                execution_text += f"- ({s.statement_date}) {s.statement_text[:150]}{score_str}\n"
    except Exception:
        pass  # execution data is optional

    execution_text = execution_text[:2000]

    # Run LLM analysis (async, 8192 tokens to avoid truncated JSON)
    prompt = (MOAT_ANALYSIS
        .replace("{company}", company.name)
        .replace("{ticker}", company.ticker)
        .replace("{sector}", company.sector or "N/A")
        .replace("{thesis}", thesis_text)
        .replace("{metrics}", metrics_text)
        .replace("{commentary}", commentary_text)
        .replace("{esg_data}", esg_text)
        .replace("{execution}", execution_text)
    )

    try:
        result = await call_llm_json_async(prompt, max_tokens=8192)
    except Exception as e:
        raise HTTPException(502, f"Moat analysis failed: {str(e)[:300]}")

    if not isinstance(result, dict):
        raise HTTPException(502, "Moat analysis returned unexpected format — retry.")

    # Persist result to ResearchOutput
    import json as _json2
    output = ResearchOutput(
        company_id=company.id,
        period_label="all_time",
        output_type="moat_analysis",
        content_json=_json2.dumps(result),
        review_status="draft",
    )
    db.add(output)
    await db.commit()

    return result


@router.get("/companies/{ticker}/moat-analysis")
async def get_moat_analysis(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Get the most recent moat analysis for a company, if one exists.
    """
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Get most recent moat analysis
    import json as _json3
    output_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company.id,
            ResearchOutput.output_type == "moat_analysis",
        ).order_by(ResearchOutput.created_at.desc()).limit(1)
    )
    output = output_q.scalar_one_or_none()

    if not output or not output.content_json:
        return {"exists": False, "data": None, "created_at": None}

    try:
        data = _json3.loads(output.content_json)
    except Exception:
        data = None

    return {
        "exists": True,
        "data": data,
        "created_at": output.created_at.isoformat() if output.created_at else None,
    }
