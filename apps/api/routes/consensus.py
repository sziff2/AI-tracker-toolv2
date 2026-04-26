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
