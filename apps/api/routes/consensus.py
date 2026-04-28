"""
Consensus expectations — analyst-curated street estimates per (company,
period, metric). Used to render Actual-vs-Consensus beat/miss in the
Results tab and feed agent prompts the right benchmark for "what did
the market expect" framing.

Endpoints:
  GET  /companies/{ticker}/consensus?period=2026_Q1
  POST /companies/{ticker}/consensus            (single upsert)
  POST /companies/{ticker}/consensus/bulk       (CSV paste)
  DELETE /consensus/{id}
"""

from __future__ import annotations

import csv
import io
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import Company, ConsensusExpectation
from services.consensus_storage import consensus_to_dict, upsert_consensus_row

logger = logging.getLogger(__name__)
router = APIRouter(tags=["consensus"])


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class ConsensusItem(BaseModel):
    metric_name:     str
    consensus_value: Optional[float] = None
    unit:            Optional[str] = None
    source:          Optional[str] = None
    notes:           Optional[str] = None


class ConsensusUpsertSingle(BaseModel):
    period_label:    str
    metric_name:     str
    consensus_value: Optional[float] = None
    unit:            Optional[str] = None
    source:          Optional[str] = None
    notes:           Optional[str] = None
    uploaded_by:     Optional[str] = None


class ConsensusBulkUpload(BaseModel):
    period_label: str
    csv_text:     str    # paste from a spreadsheet — first row = header
    source:       Optional[str] = None
    uploaded_by:  Optional[str] = None


class ConsensusOut(BaseModel):
    id:              str
    period_label:    str
    metric_name:     str
    consensus_value: Optional[float]
    unit:            Optional[str]
    source:          Optional[str]
    notes:           Optional[str]
    uploaded_by:     Optional[str]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

async def _get_company_or_404(db: AsyncSession, ticker: str) -> Company:
    q = await db.execute(select(Company).where(Company.ticker == ticker.upper().strip()))
    co = q.scalar_one_or_none()
    if not co:
        raise HTTPException(404, f"Company {ticker} not found")
    return co


# Storage helpers live in services/consensus_storage.py so the
# document-pipeline extractor can reuse them.
_row_to_dict = consensus_to_dict
_upsert_one = upsert_consensus_row


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get("/companies/{ticker}/consensus")
async def get_consensus(
    ticker: str,
    period: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List consensus expectations for a company, optionally scoped to
    a period. Returns rows ordered by metric_name."""
    co = await _get_company_or_404(db, ticker)
    q = select(ConsensusExpectation).where(ConsensusExpectation.company_id == co.id)
    if period:
        q = q.where(ConsensusExpectation.period_label == period)
    q = q.order_by(ConsensusExpectation.period_label.desc(), ConsensusExpectation.metric_name)
    res = await db.execute(q)
    rows = res.scalars().all()
    return {
        "ticker":       co.ticker,
        "period":       period,
        "items":        [_row_to_dict(r) for r in rows],
        "count":        len(rows),
    }


@router.post("/companies/{ticker}/consensus")
async def upsert_consensus(
    ticker: str,
    body: ConsensusUpsertSingle,
    db: AsyncSession = Depends(get_db),
):
    """Insert or update a single consensus row. Conflict on
    (company_id, period_label, metric_name) updates in place."""
    co = await _get_company_or_404(db, ticker)
    if not body.metric_name.strip():
        raise HTTPException(400, "metric_name required")
    out = await _upsert_one(
        db,
        company_id=co.id,
        period_label=body.period_label.strip(),
        metric_name=body.metric_name.strip(),
        consensus_value=body.consensus_value,
        unit=(body.unit or "").strip() or None,
        source=(body.source or "").strip() or None,
        notes=(body.notes or "").strip() or None,
        uploaded_by=(body.uploaded_by or "").strip() or None,
    )
    await db.commit()
    return out


@router.post("/companies/{ticker}/consensus/bulk")
async def bulk_upload_consensus(
    ticker: str,
    body: ConsensusBulkUpload,
    db: AsyncSession = Depends(get_db),
):
    """Paste a CSV (or TSV) of consensus expectations. First row is the
    header. Required column: metric_name. Optional columns:
    consensus_value (or value), unit, source, notes.

    Examples that parse cleanly:

        metric_name,consensus_value,unit
        Total Income,14336,SEK_M
        NII,9976,SEK_M
        EPS,2.98,SEK
        ROE,12.35,%

    Tab-separated also works (paste from Excel).
    """
    co = await _get_company_or_404(db, ticker)
    raw = (body.csv_text or "").strip()
    if not raw:
        raise HTTPException(400, "csv_text empty")

    # Detect delimiter: tab if any row has one, else comma.
    delim = "\t" if "\t" in raw.splitlines()[0] else ","
    reader = csv.DictReader(io.StringIO(raw), delimiter=delim)
    if not reader.fieldnames:
        raise HTTPException(400, "Could not parse header row")
    headers = {h.strip().lower(): h for h in reader.fieldnames}
    if "metric_name" not in headers and "metric" not in headers:
        raise HTTPException(400, "First row must contain 'metric_name' (or 'metric') header")

    metric_col = headers.get("metric_name", headers.get("metric"))
    value_col = headers.get("consensus_value") or headers.get("value") or headers.get("estimate")
    unit_col = headers.get("unit")
    source_col = headers.get("source")
    notes_col = headers.get("notes") or headers.get("comment")

    inserted: list[dict] = []
    skipped: list[dict] = []
    for raw_row in reader:
        name = (raw_row.get(metric_col) or "").strip() if metric_col else ""
        if not name:
            continue
        v_raw = (raw_row.get(value_col) or "").strip() if value_col else ""
        try:
            # Strip currency symbols + commas commonly pasted from spreadsheets.
            v_clean = v_raw.replace(",", "").replace(" ", "")
            v_clean = v_clean.lstrip("$£€¥")
            value = float(v_clean) if v_clean else None
        except ValueError:
            skipped.append({"metric": name, "reason": f"non-numeric value '{v_raw}'"})
            continue
        out = await _upsert_one(
            db,
            company_id=co.id,
            period_label=body.period_label.strip(),
            metric_name=name,
            consensus_value=value,
            unit=((raw_row.get(unit_col) or "").strip() if unit_col else None) or None,
            source=((raw_row.get(source_col) or "").strip() if source_col else None) or body.source,
            notes=((raw_row.get(notes_col) or "").strip() if notes_col else None) or None,
            uploaded_by=body.uploaded_by,
        )
        inserted.append(out)
    await db.commit()
    return {
        "ticker":   co.ticker,
        "period":   body.period_label,
        "inserted": len(inserted),
        "skipped":  len(skipped),
        "items":    inserted,
        "errors":   skipped,
    }


@router.delete("/consensus/{row_id}", status_code=204)
async def delete_consensus(row_id: str, db: AsyncSession = Depends(get_db)):
    try:
        rid = uuid.UUID(row_id)
    except ValueError:
        raise HTTPException(400, "invalid id")
    await db.execute(delete(ConsensusExpectation).where(ConsensusExpectation.id == rid))
    await db.commit()


# ─────────────────────────────────────────────────────────────────
# Results summary — joins Actuals × Consensus × Prior-year YoY
# ─────────────────────────────────────────────────────────────────

@router.get("/companies/{ticker}/results-summary")
async def results_summary(
    ticker: str,
    period: str,
    db: AsyncSession = Depends(get_db),
):
    """Quarterly Summary feed for the Results tab.

    Joins three sources for one period:
      • extracted_metrics for the period          → actual
      • extracted_metrics for the prior-year period → YoY anchor
      • consensus_expectations for the period     → vs-consensus benchmark

    Match key is lowercase, whitespace-stripped metric_name. Returns
    rows ordered as: rows-with-consensus first (sorted by absolute
    beat/miss magnitude), then actuals-only rows. Each row includes
    the raw fields plus pre-computed yoy_pct and beat_miss_pct so
    the UI can render directly.
    """
    from apps.api.models import ExtractedMetric
    from services.metric_normaliser import _previous_period

    co_q = await db.execute(select(Company).where(Company.ticker == ticker.upper().strip()))
    co = co_q.scalar_one_or_none()
    if not co:
        raise HTTPException(404, f"Company {ticker} not found")

    # 1. Actuals for this period
    cur_q = await db.execute(
        select(ExtractedMetric).where(
            ExtractedMetric.company_id == co.id,
            ExtractedMetric.period_label == period,
        )
    )
    cur_rows = cur_q.scalars().all()

    # 2. Prior-year period for YoY (use the canonical _previous_period
    #    semantics — Q[1-4] / H2 sequential, FY / H1 / L3Q / LTM YoY)
    prior_period = _previous_period(period)
    prior_rows: list = []
    if prior_period:
        prior_q = await db.execute(
            select(ExtractedMetric).where(
                ExtractedMetric.company_id == co.id,
                ExtractedMetric.period_label == prior_period,
            )
        )
        prior_rows = prior_q.scalars().all()

    # 3. Consensus expectations for this period
    cons_q = await db.execute(
        select(ConsensusExpectation).where(
            ConsensusExpectation.company_id == co.id,
            ConsensusExpectation.period_label == period,
        )
    )
    cons_rows = cons_q.scalars().all()

    import re as _re

    # Adjectives stripped from both sides before comparing — "adjusted
    # operating income" + "operating income" should join, but "operating
    # income growth" should NOT join "operating income" (different metric).
    _STRIP_PREFIX = {
        "adjusted", "underlying", "reported", "company", "broker",
        "consensus", "basis", "gaap", "non-gaap", "diluted", "basic",
        "total", "net", "core",
    }
    # Trailing tokens that change the SUBSTANCE of a metric — never strip.
    # We don't strip these but we use them to reject sloppy substring
    # matches (operating_income_growth ≠ operating_income).
    _SUBSTANCE_TOKENS = {
        "growth", "yoy", "qoq", "margin", "ratio", "rate", "pct",
        "change", "delta", "bridge", "gap", "abs", "effect",
    }

    def _key(name: str) -> str:
        s = (name or "").lower().replace("_", " ").replace("-", " ")
        # Collapse parentheticals to spaces so "(GAAP)" doesn't trip token logic.
        s = _re.sub(r"\(.*?\)", " ", s)
        s = _re.sub(r"\s+", " ", s).strip()
        return s

    def _core_tokens(name: str) -> tuple[str, ...]:
        """Tokenise + drop generic adjectives. The remaining ordered
        tokens are the metric's *substance*. Two metrics join iff their
        core token tuples are equal AND neither carries a substance
        token the other lacks."""
        toks = [t for t in _key(name).split() if t and t not in _STRIP_PREFIX]
        return tuple(toks)

    def _is_derivative(name: str) -> bool:
        """Skip rows that compute from a base metric — they share the
        base name as a substring but represent a different number
        (e.g. operating_income_growth, BRIDGE_GAP_*, tax_effect_*).
        Substring fallback would otherwise match these to the absolute
        consensus and produce nonsense beat/miss values like +250%."""
        n = (name or "").lower()
        if any(s in n for s in ("bridge_", "bridge:", "_growth", " growth",
                                "tax_effect", "_pct", "_yoy", "_qoq",
                                "_change", "_delta")):
            return True
        return False

    # Index prior + consensus by normalised metric name.
    prior_by_name: dict[str, ExtractedMetric] = {}
    for r in prior_rows:
        k = _key(r.metric_name)
        if k not in prior_by_name or (
            float(r.confidence or 0) > float(prior_by_name[k].confidence or 0)
        ):
            prior_by_name[k] = r

    # Build TWO consensus indexes: full key + core-token tuple. The
    # core-tuple lookup catches "Adjusted Operating Income" ↔
    # "operating_income" while NOT matching "Operating Income Growth"
    # (different core-token tuple).
    cons_by_key: dict[str, ConsensusExpectation] = {}
    cons_by_core: dict[tuple, ConsensusExpectation] = {}
    for r in cons_rows:
        cons_by_key.setdefault(_key(r.metric_name), r)
        core = _core_tokens(r.metric_name)
        if core:
            cons_by_core.setdefault(core, r)

    def _lookup_consensus(name: str) -> ConsensusExpectation | None:
        if _is_derivative(name):
            return None
        # Exact key match
        k = _key(name)
        if k in cons_by_key:
            return cons_by_key[k]
        # Core-token tuple match (drops "adjusted" / "diluted" / etc.)
        core = _core_tokens(name)
        if core and core in cons_by_core:
            return cons_by_core[core]
        return None

    # Build the joined rows. Drive off actuals, then add consensus-only
    # rows (consensus uploaded but no matching extracted metric — usually
    # means the metric name doesn't align; surface so the analyst sees it).
    seen_keys: set[str] = set()
    matched_cons_keys: set[str] = set()
    rows: list[dict] = []

    for r in cur_rows:
        k = _key(r.metric_name)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        actual = float(r.metric_value) if r.metric_value is not None else None
        prior = prior_by_name.get(k)
        prior_v = float(prior.metric_value) if (prior and prior.metric_value is not None) else None
        cons = _lookup_consensus(r.metric_name)
        cons_v = float(cons.consensus_value) if (cons and cons.consensus_value is not None) else None
        if cons is not None:
            matched_cons_keys.add(_key(cons.metric_name))

        yoy_pct = None
        if actual is not None and prior_v not in (None, 0):
            yoy_pct = round((actual - prior_v) / abs(prior_v) * 100, 2)

        beat_miss_pct = None
        beat_miss_abs = None
        if actual is not None and cons_v not in (None, 0):
            beat_miss_abs = round(actual - cons_v, 2)
            beat_miss_pct = round((actual - cons_v) / abs(cons_v) * 100, 2)

        rows.append({
            "metric_name":     r.metric_name,
            "unit":            r.unit,
            "segment":         r.segment,
            "period_label":    r.period_label,
            "period_frequency": r.period_frequency,
            "actual":          actual,
            "actual_text":     r.metric_text,
            "prior_period":    prior_period if prior else None,
            "prior_value":     prior_v,
            "yoy_pct":         yoy_pct,
            "consensus":       cons_v,
            "consensus_unit":  cons.unit if cons else None,
            "consensus_source": cons.source if cons else None,
            "consensus_notes": cons.notes if cons else None,
            "beat_miss_abs":   beat_miss_abs,
            "beat_miss_pct":   beat_miss_pct,
            "has_consensus":   cons_v is not None,
            "confidence":      float(r.confidence) if r.confidence is not None else None,
        })

    # Consensus-only rows (no matching actual). Helpful for the analyst:
    # if they uploaded "Total income" but the extractor wrote "Revenue",
    # the mismatch is now visible. Skip rows already matched via
    # _lookup_consensus to avoid duplicating them as "consensus_only".
    for k, cons in cons_by_key.items():
        if k in matched_cons_keys:
            continue
        if k in seen_keys:
            continue
        cons_v = float(cons.consensus_value) if cons.consensus_value is not None else None
        rows.append({
            "metric_name":     cons.metric_name,
            "unit":            cons.unit,
            "segment":         None,
            "period_label":    period,
            "period_frequency": None,
            "actual":          None,
            "actual_text":     None,
            "prior_period":    None,
            "prior_value":     None,
            "yoy_pct":         None,
            "consensus":       cons_v,
            "consensus_unit":  cons.unit,
            "consensus_source": cons.source,
            "consensus_notes": cons.notes,
            "beat_miss_abs":   None,
            "beat_miss_pct":   None,
            "has_consensus":   True,
            "confidence":      None,
            "consensus_only":  True,
        })

    # Sort: consensus-bearing rows first (by abs beat/miss magnitude
    # descending so the most-surprising lines lead), then actuals-only.
    def _sort_key(r):
        has_cons = bool(r.get("has_consensus"))
        bm = abs(r.get("beat_miss_pct") or 0.0)
        return (not has_cons, -bm)
    rows.sort(key=_sort_key)

    return {
        "ticker":              co.ticker,
        "company_name":        co.name,
        "period":              period,
        "prior_period":        prior_period,
        "rows":                rows,
        "rows_with_consensus": sum(1 for r in rows if r.get("has_consensus") and r.get("actual") is not None),
        "rows_total":          len(rows),
        "consensus_uploaded":  len(cons_rows),
        "consensus_only":      sum(1 for r in rows if r.get("consensus_only")),
    }
