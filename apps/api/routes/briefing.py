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
    TrackedKPI, KPIScore, ExtractedMetric, DecisionLog, AgentOutput,
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

    # 2. Latest synthesis + raw agent outputs.
    # Always pull agent_outputs — the renderer now mirrors the UI cards
    # (financial_analyst / transcript / presentation / bear / bull /
    # debate / QC / competitive_positioning / guidance_tracker) so the
    # PDF and the UI surface the same structured content. The legacy
    # ResearchOutput "full_analysis" stitched briefing is kept only as
    # a fallback for older runs that pre-date the agent pipeline.
    briefing: dict = {}
    agent_outs: dict[str, dict] = {}

    aoq = await db.execute(
        select(AgentOutput).where(
            AgentOutput.company_id == company.id,
            AgentOutput.period_label == period,
            AgentOutput.status == "completed",
        ).order_by(desc(AgentOutput.created_at))
    )
    for ao in aoq.scalars():
        # Keep only the most recent output per agent (order is desc)
        if ao.agent_id not in agent_outs and isinstance(ao.output_json, dict):
            agent_outs[ao.agent_id] = ao.output_json

    out_q = await db.execute(
        select(ResearchOutput).where(
            ResearchOutput.company_id == company.id,
            ResearchOutput.period_label == period,
            ResearchOutput.output_type.in_(["full_analysis", "batch_synthesis"]),
        ).order_by(desc(ResearchOutput.created_at)).limit(1)
    )
    analysis = out_q.scalar_one_or_none()
    if analysis and analysis.content_json:
        try:
            data = json.loads(analysis.content_json)
            briefing = data.get("synthesis") or data.get("briefing") or {}
            if not briefing and any(k in data for k in ("headline", "bottom_line")):
                briefing = data
        except Exception as exc:
            logger.warning("briefing content_json parse failed: %s", exc)

    # If nothing stitched but agents are there, fall back to the flat shim
    # so the renderer's legacy "Bottom line" block isn't empty.
    if not briefing and agent_outs:
        briefing = _briefing_from_agent_outputs(agent_outs)

    if not briefing and not agent_outs:
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

    # 4. Scenarios — prefer analyst-set ValuationScenario rows, fall back
    # to probabilities the debate_agent / bear_case / bull_case produced.
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
    if not scenarios and agent_outs:
        scenarios = _scenarios_from_agent_outputs(agent_outs)

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

    # 7. Financial Summary feed — same shape as /results-summary so the
    #    PDF can render the analyst-template table that sits at the top
    #    of the Results tab in the UI.
    fin_summary: dict = {}
    try:
        from apps.api.routes.consensus import results_summary as _results_summary_route
        fin_summary = await _results_summary_route(ticker_u, period, db)
    except Exception as exc:
        logger.warning("results_summary lookup failed in briefing PDF: %s", str(exc)[:200])

    # 8. Render
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
            agent_outs=agent_outs,
            fin_summary=fin_summary,
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


def _briefing_from_agent_outputs(agents: dict[str, dict]) -> dict:
    """Stitch a synthesis-shaped briefing dict out of the Phase-B agent
    outputs (financial_analyst, debate_agent, bear_case, bull_case,
    guidance_tracker, quality_control). Returns {} when nothing useful is
    present.

    The renderer (services/briefing_pdf.py) expects keys:
      headline / bottom_line / what_happened / management_message /
      thesis_impact / what_changed / thesis_status
    Map agent fields onto these so the existing renderer works unchanged.
    """
    fa    = agents.get("financial_analyst", {}) or {}
    dbt   = agents.get("debate_agent", {}) or {}
    bear  = agents.get("bear_case", {}) or {}
    bull  = agents.get("bull_case", {}) or {}
    gt    = agents.get("guidance_tracker", {}) or {}
    qc    = agents.get("quality_control", {}) or {}

    out: dict = {}

    # Headline: prefer debate verdict; fall back to financial_analyst grade
    verdict = dbt.get("verdict") or dbt.get("recommendation") or ""
    grade   = fa.get("overall_grade") or ""
    thesis_dir = fa.get("thesis_direction") or ""
    if verdict:
        out["headline"] = f"Debate verdict: {verdict}" + (f" (grade: {grade})" if grade else "")
    elif grade:
        out["headline"] = f"Overall grade: {grade}" + (f" — thesis direction: {thesis_dir}" if thesis_dir else "")

    # Bottom line: debate_summary is the strongest single paragraph
    if dbt.get("debate_summary"):
        out["bottom_line"] = _compact_str(dbt["debate_summary"])

    # What happened: financial_analyst key surprises + revenue/margin read
    bits = []
    if fa.get("revenue_assessment"):
        bits.append("Revenue: " + _compact_str(fa["revenue_assessment"]))
    if fa.get("margin_analysis"):
        bits.append("Margins: " + _compact_str(fa["margin_analysis"]))
    if fa.get("key_surprises"):
        surprises = fa["key_surprises"]
        if isinstance(surprises, list):
            bits.append("Surprises: " + "; ".join(_compact_str(s) for s in surprises[:5]))
        elif isinstance(surprises, str):
            bits.append("Surprises: " + _compact_str(surprises))
    if bits:
        out["what_happened"] = "\n".join(bits)

    # Management message: financial_analyst.management_signals +
    # guidance_tracker.overall_signal
    mbits = []
    if fa.get("management_signals"):
        mbits.append(_compact_str(fa["management_signals"]))
    if gt.get("overall_signal"):
        mbits.append("Guidance signal: " + _compact_str(gt["overall_signal"]))
    if gt.get("track_record_signal"):
        mbits.append("Track record: " + _compact_str(gt["track_record_signal"]))
    if mbits:
        out["management_message"] = "\n".join(mbits)

    # Thesis impact: fa.thesis_direction + debate's base-case narrative
    tbits = []
    if thesis_dir:
        tbits.append(f"Thesis direction: {thesis_dir}")
    if dbt.get("base_scenario"):
        base = dbt["base_scenario"]
        if isinstance(base, dict) and base.get("description"):
            tbits.append("Base case: " + _compact_str(base["description"]))
        elif isinstance(base, str):
            tbits.append("Base case: " + _compact_str(base))
    if tbits:
        out["thesis_impact"] = "\n".join(tbits)

    # What changed: guidance_tracker.methodology_changes or .notable_walkbacks
    cbits = []
    if gt.get("methodology_changes"):
        mc = gt["methodology_changes"]
        if isinstance(mc, list) and mc:
            cbits.append("Methodology: " + "; ".join(_compact_str(x) for x in mc[:3]))
    if gt.get("notable_walkbacks"):
        nw = gt["notable_walkbacks"]
        if isinstance(nw, list) and nw:
            cbits.append("Walkbacks: " + "; ".join(_compact_str(x) for x in nw[:3]))
    if cbits:
        out["what_changed"] = "\n".join(cbits)

    # Thesis status: QC's recommendation gives a clean one-liner
    qc_rec = qc.get("recommendation")
    if qc_rec:
        out["thesis_status"] = _compact_str(qc_rec)

    # Stash bear/bull theses on top-level so the PDF has narrative context
    # even if not in the usual renderer keys.
    if bear.get("bear_thesis"):
        out.setdefault("what_happened", "")
        out["what_happened"] = (out.get("what_happened","") + "\n\nBear thesis: " + _compact_str(bear["bear_thesis"])).strip()
    if bull.get("bull_thesis"):
        out["what_happened"] = (out.get("what_happened","") + "\n\nBull thesis: " + _compact_str(bull["bull_thesis"])).strip()

    return out


def _scenarios_from_agent_outputs(agents: dict[str, dict]) -> dict:
    """Build scenarios block from debate / bear / bull output when DB
    ValuationScenario rows are empty."""
    out: dict = {}
    dbt  = agents.get("debate_agent", {}) or {}
    bear = agents.get("bear_case", {}) or {}
    bull = agents.get("bull_case", {}) or {}

    # debate_agent exposes base_scenario + per-side probabilities
    base = dbt.get("base_scenario")
    if isinstance(base, dict):
        out["base"] = {
            "target_price":  base.get("target_price") or base.get("price"),
            "probability":   dbt.get("base_probability"),
            "methodology":   base.get("description") or base.get("methodology"),
        }

    bull_s = bull.get("upside_scenario") if isinstance(bull.get("upside_scenario"), dict) else None
    if bull_s:
        out["bull"] = {
            "target_price":  bull_s.get("target_price") or bull_s.get("implied_price"),
            "probability":   dbt.get("bull_probability"),
            "methodology":   bull_s.get("description") or bull_s.get("methodology"),
        }

    bear_s = bear.get("downside_scenario") if isinstance(bear.get("downside_scenario"), dict) else None
    if bear_s:
        out["bear"] = {
            "target_price":  bear_s.get("target_price") or bear_s.get("implied_price"),
            "probability":   dbt.get("bear_probability"),
            "methodology":   bear_s.get("description") or bear_s.get("methodology"),
        }

    return out


def _compact_str(v, max_len: int = 20000) -> str:
    """Coerce anything reasonable to a readable single string for the PDF.

    Default cap was 1200 chars (with nested calls at 300) which clipped
    bull/bear theses mid-sentence ("The bull case was stro..."). Bumped
    to 20K so analyst narratives flow through unchanged. Nested dict/
    list serialisation still uses 800 per sub-value so a single field
    doesn't dominate."""
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
    elif isinstance(v, dict):
        parts = []
        for k, val in v.items():
            if val is None or val == "":
                continue
            parts.append(f"{k}: {_compact_str(val, max_len=800)}")
        s = " | ".join(parts)
    elif isinstance(v, list):
        s = "; ".join(_compact_str(x, max_len=800) for x in v if x not in (None, ""))
    else:
        s = str(v)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s
