"""
Management Execution Assessment — tracks management promises vs actual delivery.

Endpoints:
  POST /companies/{ticker}/execution/extract         — extract statements from documents
  POST /companies/{ticker}/execution/assess           — assess open statements against actual results
  GET  /companies/{ticker}/execution/statements       — list all statements
  PUT  /companies/{ticker}/execution/statements/{id}  — manually update a statement's outcome
  GET  /companies/{ticker}/execution/scorecard        — aggregated scorecard
  GET  /companies/{ticker}/execution/time-series      — credibility over time
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from apps.api.rate_limit import limiter
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company, Document, ExtractedMetric,
    ManagementStatement, ExecutionScorecard,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["execution"])

SCORE_MAP = {
    "delivered": 2, "mostly_delivered": 1, "neutral": 0,
    "missed": -1, "major_miss": -2,
}


# ── Schemas ──────────────────────────────────────────────────

class StatementUpdate(BaseModel):
    status: str                     # delivered | mostly_delivered | neutral | missed | major_miss
    outcome_value: Optional[str] = None
    outcome_evidence: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────

async def _get_company(db, ticker):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company


# ═══════════════════════════════════════════════════════════════
# EXTRACT STATEMENTS from documents
# ═══════════════════════════════════════════════════════════════

@router.post("/companies/{ticker}/execution/extract")
@limiter.limit("10/minute")
async def extract_statements(request: Request, ticker: str, period: str = None, db: AsyncSession = Depends(get_db)):
    """
    Extract forward-looking management statements from all documents for this company.
    If period is specified, only process that period's documents.
    """
    from services.llm_client import call_llm_json
    from prompts import MGMT_STATEMENT_EXTRACTOR
    from pathlib import Path
    from configs.settings import settings

    company = await _get_company(db, ticker)

    # Find documents to process
    q = select(Document).where(Document.company_id == company.id)
    if period:
        q = q.where(Document.period_label == period)
    q = q.order_by(Document.period_label.desc())
    result = await db.execute(q)
    docs = result.scalars().all()

    if not docs:
        raise HTTPException(404, "No documents found for this company")

    all_statements = []

    for doc in docs:
        # Load parsed text — try disk first, fall back to DB sections
        full_text = ""
        try:
            text_path = Path(settings.storage_base_path) / "processed" / ticker.upper() / (doc.period_label or "misc") / "parsed_text.json"
            if text_path.exists():
                pages = json.loads(text_path.read_text())
                full_text = "\n\n".join(p["text"] for p in pages)
        except Exception:
            pass

        # Fall back to DocumentSections in DB
        if not full_text:
            try:
                from apps.api.models import DocumentSection
                sections_q = await db.execute(
                    select(DocumentSection).where(DocumentSection.document_id == doc.id)
                    .order_by(DocumentSection.page_number)
                )
                sections = sections_q.scalars().all()
                if sections:
                    full_text = "\n\n".join(s.text_content for s in sections if s.text_content)
            except Exception as e:
                logger.warning("Failed to load sections for doc %s: %s", doc.id, str(e)[:100])

        if not full_text:
            logger.warning("No text available for doc %s (no file, no sections)", doc.id)
            continue

        # Truncate for LLM context
        if len(full_text) > 15000:
            full_text = full_text[:15000]

        # Extract statements via LLM
        try:
            prompt = MGMT_STATEMENT_EXTRACTOR.format(text=full_text)
            items = call_llm_json(prompt, max_tokens=4096)
            if not isinstance(items, list):
                items = [items] if isinstance(items, dict) else []
        except Exception as e:
            logger.warning("LLM extraction failed for doc %s: %s", doc.id, str(e)[:100])
            continue

        # Persist statements
        for item in items:
            if not isinstance(item, dict):
                continue
            stmt_text = item.get("statement_text", "")
            if not stmt_text:
                continue

            # Check for duplicates (same company, similar statement text)
            existing_q = await db.execute(
                select(ManagementStatement).where(
                    ManagementStatement.company_id == company.id,
                    ManagementStatement.statement_text == stmt_text,
                )
            )
            if existing_q.scalar_one_or_none():
                continue

            stmt = ManagementStatement(
                id=uuid.uuid4(),
                company_id=company.id,
                document_id=doc.id,
                statement_date=doc.period_label,
                speaker=item.get("speaker", "management"),
                category=item.get("category", "strategy"),
                statement_text=stmt_text,
                target_metric=item.get("target_metric"),
                target_value=item.get("target_value"),
                target_direction=item.get("target_direction"),
                target_timeframe=item.get("target_timeframe"),
                confidence_type=item.get("confidence_type", "directional"),
                source_snippet=(item.get("source_snippet") or "")[:500],
                status="open",
            )
            db.add(stmt)
            all_statements.append({
                "statement": stmt_text,
                "category": item.get("category"),
                "speaker": item.get("speaker"),
                "target_metric": item.get("target_metric"),
                "target_value": item.get("target_value"),
                "timeframe": item.get("target_timeframe"),
                "source_period": doc.period_label,
            })

    await db.commit()
    return {
        "extracted": len(all_statements),
        "documents_processed": len(docs),
        "statements": all_statements,
    }


# ═══════════════════════════════════════════════════════════════
# ASSESS OUTCOMES — compare statements against actual results
# ═══════════════════════════════════════════════════════════════

@router.post("/companies/{ticker}/execution/assess")
@limiter.limit("10/minute")
async def assess_outcomes(request: Request, ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Use LLM to assess open management statements against actual extracted metrics.
    Updates statement statuses and scores, rebuilds the scorecard.
    """
    from services.llm_client import call_llm_json
    from prompts import MGMT_OUTCOME_ASSESSOR

    company = await _get_company(db, ticker)

    # Get open statements
    stmt_q = await db.execute(
        select(ManagementStatement).where(
            ManagementStatement.company_id == company.id,
            ManagementStatement.status == "open",
        ).order_by(ManagementStatement.statement_date)
    )
    statements = stmt_q.scalars().all()

    if not statements:
        raise HTTPException(400, "No open statements to assess. Extract statements first.")

    # Build statements text
    stmt_text = ""
    for i, s in enumerate(statements):
        stmt_text += f"\n[{i}] ({s.statement_date}, {s.speaker}, {s.category})\n"
        stmt_text += f"    Statement: {s.statement_text}\n"
        if s.target_metric:
            stmt_text += f"    Target: {s.target_metric} = {s.target_value or 'N/A'}\n"
        if s.target_timeframe:
            stmt_text += f"    Timeframe: {s.target_timeframe}\n"
        stmt_text += f"    Confidence: {s.confidence_type}\n"

    # Get actual results (extracted metrics)
    metrics_q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company.id
        ).order_by(ExtractedMetric.period_label.desc()).limit(200)
    )
    metrics = metrics_q.scalars().all()

    actual_text = ""
    if metrics:
        by_period = {}
        for m in metrics:
            p = m.period_label or "unknown"
            if p not in by_period:
                by_period[p] = []
            if len(by_period[p]) < 20:  # Limit per period
                by_period[p].append(f"  {m.metric_name}: {m.metric_value} {m.unit or ''}")

        for p in sorted(by_period.keys(), reverse=True)[:6]:
            actual_text += f"\n[{p}]\n" + "\n".join(by_period[p]) + "\n"
    else:
        actual_text = "No extracted metrics available. Assess based on statement content and general knowledge."

    # LLM assessment
    prompt = MGMT_OUTCOME_ASSESSOR.format(
        company=company.name, ticker=company.ticker,
        statements=stmt_text, actual_results=actual_text,
    )

    try:
        result = call_llm_json(prompt, max_tokens=4096)
    except Exception as e:
        raise HTTPException(502, f"Assessment failed: {str(e)[:200]}")

    # Apply assessments
    assessments = result.get("assessments", [])
    updated = 0
    for a in assessments:
        if not isinstance(a, dict):
            continue
        idx = a.get("statement_index")
        if idx is None or idx >= len(statements):
            continue

        stmt = statements[idx]
        status = a.get("status", "neutral")
        stmt.status = status
        stmt.score = SCORE_MAP.get(status, 0)
        stmt.outcome_value = a.get("outcome_value")
        stmt.outcome_evidence = a.get("evidence")
        stmt.outcome_date = datetime.now(timezone.utc).isoformat()[:10]
        stmt.assessed_by = "auto"
        updated += 1

    # Rebuild scorecard
    overall = result.get("overall", {})
    await _rebuild_scorecard(db, company, overall)

    await db.commit()
    return {
        "assessed": updated,
        "total_open": len(statements),
        "overall": overall,
    }


async def _rebuild_scorecard(db, company, overall_assessment: dict = None):
    """Rebuild the execution scorecard from all scored statements."""
    stmt_q = await db.execute(
        select(ManagementStatement).where(
            ManagementStatement.company_id == company.id,
            ManagementStatement.status != "open",
        )
    )
    scored = stmt_q.scalars().all()

    if not scored:
        return

    total = len(scored)
    delivered = sum(1 for s in scored if s.status in ("delivered", "mostly_delivered"))
    missed = sum(1 for s in scored if s.status in ("missed", "major_miss"))
    scores = [s.score for s in scored if s.score is not None]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0

    # Category breakdown
    cat_scores = {}
    for s in scored:
        cat = s.category or "other"
        if cat not in cat_scores:
            cat_scores[cat] = []
        if s.score is not None:
            cat_scores[cat].append(s.score)
    cat_avgs = {k: round(sum(v) / len(v), 2) for k, v in cat_scores.items() if v}

    # Count open
    open_q = await db.execute(
        select(func.count(ManagementStatement.id)).where(
            ManagementStatement.company_id == company.id,
            ManagementStatement.status == "open",
        )
    )
    open_count = open_q.scalar() or 0

    # Upsert scorecard
    existing_q = await db.execute(
        select(ExecutionScorecard).where(
            ExecutionScorecard.company_id == company.id,
            ExecutionScorecard.period == "all_time",
        )
    )
    card = existing_q.scalar_one_or_none()

    if not card:
        card = ExecutionScorecard(
            id=uuid.uuid4(), company_id=company.id, period="all_time",
        )
        db.add(card)

    card.overall_score = avg_score
    card.total_statements = total + open_count
    card.delivered_count = delivered
    card.missed_count = missed
    card.open_count = open_count
    card.category_scores = json.dumps(cat_avgs)
    if overall_assessment:
        card.guidance_bias = overall_assessment.get("guidance_bias")
        card.execution_reliability = overall_assessment.get("execution_reliability")
        card.strategic_consistency = overall_assessment.get("strategic_consistency")
        card.ai_assessment = overall_assessment.get("narrative")


# ═══════════════════════════════════════════════════════════════
# LIST / UPDATE STATEMENTS
# ═══════════════════════════════════════════════════════════════

@router.get("/companies/{ticker}/execution/statements")
async def list_statements(ticker: str, status: str = None, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)
    q = select(ManagementStatement).where(ManagementStatement.company_id == company.id)
    if status:
        q = q.where(ManagementStatement.status == status)
    q = q.order_by(ManagementStatement.statement_date.desc())
    result = await db.execute(q)

    return [{
        "id": str(s.id), "statement_date": s.statement_date,
        "speaker": s.speaker, "category": s.category,
        "statement_text": s.statement_text,
        "target_metric": s.target_metric, "target_value": s.target_value,
        "target_direction": s.target_direction,
        "target_timeframe": s.target_timeframe,
        "confidence_type": s.confidence_type,
        "source_snippet": s.source_snippet,
        "status": s.status, "score": s.score,
        "outcome_value": s.outcome_value,
        "outcome_evidence": s.outcome_evidence,
        "outcome_date": s.outcome_date,
        "assessed_by": s.assessed_by,
    } for s in result.scalars().all()]


@router.put("/companies/{ticker}/execution/statements/{statement_id}")
async def update_statement(ticker: str, statement_id: uuid.UUID, body: StatementUpdate, db: AsyncSession = Depends(get_db)):
    """Manually update a statement's outcome — analyst override."""
    company = await _get_company(db, ticker)
    result = await db.execute(
        select(ManagementStatement).where(ManagementStatement.id == statement_id)
    )
    stmt = result.scalar_one_or_none()
    if not stmt:
        raise HTTPException(404, "Statement not found")

    stmt.status = body.status
    stmt.score = SCORE_MAP.get(body.status, 0)
    if body.outcome_value:
        stmt.outcome_value = body.outcome_value
    if body.outcome_evidence:
        stmt.outcome_evidence = body.outcome_evidence
    stmt.outcome_date = datetime.now(timezone.utc).isoformat()[:10]
    stmt.assessed_by = "analyst"

    await _rebuild_scorecard(db, company)
    await db.commit()
    return {"status": "updated", "score": stmt.score}


# ═══════════════════════════════════════════════════════════════
# SCORECARD
# ═══════════════════════════════════════════════════════════════

@router.get("/companies/{ticker}/execution/scorecard")
async def get_scorecard(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)

    result = await db.execute(
        select(ExecutionScorecard).where(
            ExecutionScorecard.company_id == company.id,
            ExecutionScorecard.period == "all_time",
        )
    )
    card = result.scalar_one_or_none()

    if not card:
        return {
            "ticker": ticker, "has_data": False,
            "message": "No execution data. Extract statements first.",
        }

    return {
        "ticker": ticker, "has_data": True,
        "overall_score": float(card.overall_score) if card.overall_score else 0,
        "guidance_bias": card.guidance_bias,
        "execution_reliability": card.execution_reliability,
        "strategic_consistency": card.strategic_consistency,
        "total_statements": card.total_statements,
        "delivered": card.delivered_count,
        "missed": card.missed_count,
        "open": card.open_count,
        "category_scores": json.loads(card.category_scores) if card.category_scores else {},
        "ai_assessment": card.ai_assessment,
    }


# ═══════════════════════════════════════════════════════════════
# TIME SERIES — credibility over time
# ═══════════════════════════════════════════════════════════════

@router.get("/companies/{ticker}/execution/time-series")
async def execution_time_series(ticker: str, db: AsyncSession = Depends(get_db)):
    company = await _get_company(db, ticker)

    result = await db.execute(
        select(ManagementStatement).where(
            ManagementStatement.company_id == company.id,
            ManagementStatement.score.isnot(None),
        ).order_by(ManagementStatement.statement_date)
    )
    statements = result.scalars().all()

    # Group by year
    by_year = {}
    for s in statements:
        year = (s.statement_date or "unknown")[:4]
        if year == "unkn":
            continue
        if year not in by_year:
            by_year[year] = {"scores": [], "delivered": 0, "missed": 0, "total": 0}
        by_year[year]["scores"].append(s.score)
        by_year[year]["total"] += 1
        if s.status in ("delivered", "mostly_delivered"):
            by_year[year]["delivered"] += 1
        elif s.status in ("missed", "major_miss"):
            by_year[year]["missed"] += 1

    series = []
    for year in sorted(by_year.keys()):
        d = by_year[year]
        avg = round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else 0
        series.append({
            "year": year, "score": avg,
            "delivered": d["delivered"], "missed": d["missed"], "total": d["total"],
            "delivery_rate": round(d["delivered"] / d["total"] * 100) if d["total"] else 0,
        })

    return {"ticker": ticker, "series": series}
