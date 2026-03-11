"""
KPI Tracker endpoints — manage tracked KPIs and scores per company/period.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, TrackedKPI, KPIScore, ExtractedMetric

router = APIRouter(tags=["kpi-tracker"])


# ── Schemas ──────────────────────────────────────────────────
class KPICreate(BaseModel):
    kpi_name: str
    unit: Optional[str] = None
    display_order: int = 0


class KPIScoreUpdate(BaseModel):
    value: Optional[float] = None
    value_text: Optional[str] = None
    score: Optional[int] = None  # 1-5
    comment: Optional[str] = None


class KPIBulkSetup(BaseModel):
    kpis: list[KPICreate]


# ── Get tracked KPIs for a company ───────────────────────────
@router.get("/companies/{ticker}/kpis")
async def get_tracked_kpis(ticker: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    kpis_q = await db.execute(
        select(TrackedKPI)
        .where(TrackedKPI.company_id == company.id)
        .order_by(TrackedKPI.display_order)
    )
    kpis = kpis_q.scalars().all()

    return [
        {
            "id": str(k.id),
            "kpi_name": k.kpi_name,
            "unit": k.unit,
            "display_order": k.display_order,
        }
        for k in kpis
    ]


# ── Set up tracked KPIs (bulk) ───────────────────────────────
@router.post("/companies/{ticker}/kpis", status_code=201)
async def setup_tracked_kpis(ticker: str, body: KPIBulkSetup, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    created = []
    for i, kpi in enumerate(body.kpis):
        # Check if already exists
        existing = await db.execute(
            select(TrackedKPI).where(
                TrackedKPI.company_id == company.id,
                TrackedKPI.kpi_name == kpi.kpi_name,
            )
        )
        if existing.scalar_one_or_none():
            continue

        tk = TrackedKPI(
            id=uuid.uuid4(),
            company_id=company.id,
            kpi_name=kpi.kpi_name,
            unit=kpi.unit,
            display_order=kpi.display_order or i,
        )
        db.add(tk)
        created.append(kpi.kpi_name)

    await db.commit()
    return {"created": created}


# ── Delete a tracked KPI ─────────────────────────────────────
@router.delete("/companies/{ticker}/kpis/{kpi_id}")
async def delete_tracked_kpi(ticker: str, kpi_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    await db.execute(delete(KPIScore).where(KPIScore.tracked_kpi_id == kpi_id))
    await db.execute(delete(TrackedKPI).where(TrackedKPI.id == kpi_id))
    await db.commit()
    return {"status": "deleted"}


# ── Get KPI tracker grid (all KPIs × all periods) ────────────
@router.get("/companies/{ticker}/kpi-tracker")
async def get_kpi_tracker(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the full tracker grid: tracked KPIs as rows, periods as columns,
    with values and scores filled in.
    """
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Get tracked KPIs
    kpis_q = await db.execute(
        select(TrackedKPI)
        .where(TrackedKPI.company_id == company.id)
        .order_by(TrackedKPI.display_order)
    )
    tracked_kpis = kpis_q.scalars().all()

    # Get all scores
    scores_q = await db.execute(
        select(KPIScore).where(KPIScore.company_id == company.id)
    )
    all_scores = scores_q.scalars().all()

    # Get all periods with data
    periods_q = await db.execute(
        select(ExtractedMetric.period_label)
        .where(ExtractedMetric.company_id == company.id)
        .distinct()
        .order_by(ExtractedMetric.period_label)
    )
    periods = [p[0] for p in periods_q.all() if p[0]]

    # Also include periods from scores
    score_periods = set(s.period_label for s in all_scores if s.period_label)
    all_periods = sorted(set(periods) | score_periods)

    # Build score lookup
    score_map = {}
    for s in all_scores:
        score_map[(str(s.tracked_kpi_id), s.period_label)] = {
            "id": str(s.id),
            "value": float(s.value) if s.value is not None else None,
            "value_text": s.value_text,
            "score": s.score,
            "comment": s.comment,
        }

    # Try to auto-match extracted metrics to tracked KPIs
    rows = []
    for kpi in tracked_kpis:
        row = {
            "id": str(kpi.id),
            "kpi_name": kpi.kpi_name,
            "unit": kpi.unit,
            "periods": {},
        }
        for period in all_periods:
            key = (str(kpi.id), period)
            if key in score_map:
                row["periods"][period] = score_map[key]
            else:
                # Try to find matching extracted metric
                match = await _find_matching_metric(db, company.id, kpi.kpi_name, period)
                if match:
                    row["periods"][period] = {
                        "value": float(match.metric_value) if match.metric_value else None,
                        "value_text": match.metric_text,
                        "score": None,
                        "comment": None,
                        "auto_matched": True,
                    }
                else:
                    row["periods"][period] = {"value": None, "value_text": None, "score": None, "comment": None}
        rows.append(row)

    # Overall score per period
    overall_scores = {}
    for period in all_periods:
        scores_for_period = [score_map.get((str(k.id), period), {}).get("score") for k in tracked_kpis]
        valid = [s for s in scores_for_period if s is not None]
        overall_scores[period] = round(sum(valid) / len(valid), 1) if valid else None

    return {
        "company": {"ticker": company.ticker, "name": company.name},
        "periods": all_periods,
        "kpis": rows,
        "overall_scores": overall_scores,
    }


async def _find_matching_metric(db, company_id, kpi_name, period):
    """Fuzzy match a tracked KPI name against extracted metrics."""
    # Try exact match first
    q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == company_id,
            ExtractedMetric.period_label == period,
            ExtractedMetric.metric_name.ilike(f"%{kpi_name}%"),
        ).limit(1)
    )
    match = q.scalar_one_or_none()
    if match:
        return match

    # Try matching individual words
    words = kpi_name.lower().split()
    if len(words) >= 2:
        q = await db.execute(
            select(ExtractedMetric).where(
                ExtractedMetric.company_id == company_id,
                ExtractedMetric.period_label == period,
                ExtractedMetric.metric_name.ilike(f"%{words[0]}%"),
                ExtractedMetric.metric_name.ilike(f"%{words[-1]}%"),
            ).limit(1)
        )
        return q.scalar_one_or_none()
    return None


# ── Update a KPI score for a period ──────────────────────────
@router.put("/companies/{ticker}/kpis/{kpi_id}/scores/{period_label}")
async def update_kpi_score(
    ticker: str,
    kpi_id: uuid.UUID,
    period_label: str,
    body: KPIScoreUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    # Find or create score
    existing = await db.execute(
        select(KPIScore).where(
            KPIScore.tracked_kpi_id == kpi_id,
            KPIScore.period_label == period_label,
        )
    )
    score = existing.scalar_one_or_none()

    if score:
        if body.value is not None:
            score.value = body.value
        if body.value_text is not None:
            score.value_text = body.value_text
        if body.score is not None:
            score.score = body.score
        if body.comment is not None:
            score.comment = body.comment
    else:
        score = KPIScore(
            id=uuid.uuid4(),
            tracked_kpi_id=kpi_id,
            company_id=company.id,
            period_label=period_label,
            value=body.value,
            value_text=body.value_text,
            score=body.score,
            comment=body.comment,
        )
        db.add(score)

    await db.commit()
    return {"status": "updated"}


# ── Suggest KPIs from extracted metrics ──────────────────────
@router.get("/companies/{ticker}/suggest-kpis")
async def suggest_kpis(ticker: str, db: AsyncSession = Depends(get_db)):
    """Suggest KPIs based on the most commonly extracted metrics."""
    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")

    from sqlalchemy import func
    q = await db.execute(
        select(
            ExtractedMetric.metric_name,
            ExtractedMetric.unit,
            func.count(ExtractedMetric.id).label("count"),
        )
        .where(
            ExtractedMetric.company_id == company.id,
            ExtractedMetric.segment != "guidance",
            ExtractedMetric.confidence >= 0.9,
        )
        .group_by(ExtractedMetric.metric_name, ExtractedMetric.unit)
        .order_by(func.count(ExtractedMetric.id).desc())
        .limit(20)
    )

    return [
        {"kpi_name": row[0], "unit": row[1], "frequency": row[2]}
        for row in q.all()
    ]
