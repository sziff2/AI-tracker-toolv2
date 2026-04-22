"""
Briefing PDF download (Tier 3.3).

Endpoint: GET /companies/{ticker}/briefing.pdf?period=YYYY_Qn
Returns a PDF rendering of the latest synthesis output for the
period, with thesis + scenarios + tracked KPIs + recent decisions.

Uses reportlab (see services/briefing_pdf.py) rather than WeasyPrint
— pure Python, no native deps on the Railway image.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company, ResearchOutput, ThesisVersion, ValuationScenario,
    TrackedKPI, KPIScore, ExtractedMetric, DecisionLog,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["briefing"])


@router.get("/companies/{ticker}/briefing.pdf")
async def download_briefing_pdf(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """Stream a PDF briefing for the (ticker, period) pair.

    404 if the company doesn't exist. 409 with a structured error if
    no synthesis has run yet for this period (UI can show "run
    analysis first").
    """
    ticker_u = ticker.strip().upper()

    # 1. Company
    cq = await db.execute(select(Company).where(Company.ticker == ticker_u))
    company = cq.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

    # 2. Latest synthesis for period
    out_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company.id,
            ResearchOutput.period_label == period,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(desc(ResearchOutput.created_at)).limit(1)
    )
    analysis = out_q.scalar_one_or_none()
    briefing: dict = {}
    if analysis and analysis.content_json:
        try:
            data = json.loads(analysis.content_json)
            briefing = data.get("synthesis") or data.get("briefing") or {}
            # Some runs flatten directly onto root — accept that too
            if not briefing and any(k in data for k in ("headline", "bottom_line")):
                briefing = data
        except Exception as exc:
            logger.warning("briefing content_json parse failed: %s", exc)
    if not briefing:
        raise HTTPException(
            status_code=409,
            detail=f"No synthesis output for {ticker_u} {period} — run analysis first",
        )

    # 3. Active thesis
    thesis: dict | None = None
    tq = await db.execute(
        select(ThesisVersion).where(
            ThesisVersion.company_id == company.id,
            ThesisVersion.active == True,  # noqa: E712
        ).order_by(desc(ThesisVersion.thesis_date)).limit(1)
    )
    t = tq.scalar_one_or_none()
    if t:
        thesis = {
            "core_thesis":          t.core_thesis,
            "key_risks":            t.key_risks,
            "valuation_framework":  getattr(t, "valuation_framework", None),
        }

    # 4. Scenarios (bull/base/bear)
    scenarios: dict = {}
    vq = await db.execute(
        select(ValuationScenario).where(ValuationScenario.company_id == company.id)
    )
    for s in vq.scalars():
        stype = (s.scenario_type or "").lower()
        if stype in ("bull", "base", "bear"):
            scenarios[stype] = {
                "target_price":  float(s.target_price) if s.target_price is not None else None,
                "probability":   float(s.probability) if s.probability is not None else None,
                "currency":      s.currency,
                "methodology":   s.methodology,
            }

    # 5. Tracked KPIs with recent scores
    kpi_rows: list[dict] = []
    kq = await db.execute(
        select(TrackedKPI).where(TrackedKPI.company_id == company.id)
    )
    tracked = kq.scalars().all()
    if tracked:
        # Pull last 8 periods of scores + extracted metric values
        score_rows_q = await db.execute(
            select(KPIScore).where(
                KPIScore.kpi_id.in_([k.id for k in tracked])
            )
        )
        by_kpi_period: dict = {}
        for r in score_rows_q.scalars():
            by_kpi_period.setdefault(str(r.kpi_id), {})[r.period_label] = {
                "value":  float(r.metric_value) if r.metric_value is not None else None,
                "score":  float(r.score) if r.score is not None else None,
            }
        for k in tracked:
            kpi_rows.append({
                "name":    k.name,
                "periods": by_kpi_period.get(str(k.id), {}),
            })

    # 6. Decision log
    decisions: list[dict] = []
    dq = await db.execute(
        select(DecisionLog).where(DecisionLog.company_id == company.id)
        .order_by(desc(DecisionLog.created_at)).limit(10)
    )
    for d in dq.scalars():
        decisions.append({
            "created_at":  d.created_at.isoformat() if d.created_at else None,
            "action":      getattr(d, "action", "") or getattr(d, "decision", ""),
            "rationale":   getattr(d, "rationale", "") or getattr(d, "note", ""),
        })

    # 7. Render
    from services.briefing_pdf import render_briefing_pdf
    try:
        pdf_bytes = render_briefing_pdf(
            company_name=company.name or ticker_u,
            ticker=ticker_u,
            period=period,
            briefing=briefing,
            thesis=thesis,
            scenarios=scenarios,
            kpi_rows=kpi_rows,
            decisions=decisions,
            generated_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.exception("briefing pdf render failed for %s %s", ticker_u, period)
        raise HTTPException(status_code=500, detail=f"PDF render failed: {str(exc)[:200]}")

    safe_ticker = ticker_u.replace(" ", "_").replace("/", "_")
    filename = f"{safe_ticker}_{period}_briefing.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control":       "no-store",
        },
    )
