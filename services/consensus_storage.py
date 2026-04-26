"""
Consensus storage helpers — used by both the API CRUD route
(apps/api/routes/consensus.py) and the document-pipeline extractor
(services/consensus_extractor.py). Upsert keyed on
(company_id, period_label, metric_name) — the unique index defined
in the lifespan migration.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ConsensusExpectation

logger = logging.getLogger(__name__)


def consensus_to_dict(r: ConsensusExpectation) -> dict:
    return {
        "id":              str(r.id),
        "period_label":    r.period_label,
        "metric_name":     r.metric_name,
        "consensus_value": float(r.consensus_value) if r.consensus_value is not None else None,
        "unit":            r.unit,
        "source":          r.source,
        "notes":           r.notes,
        "uploaded_by":     r.uploaded_by,
    }


async def upsert_consensus_row(
    db: AsyncSession,
    *,
    company_id,
    period_label: str,
    metric_name: str,
    consensus_value: Optional[float],
    unit: Optional[str] = None,
    source: Optional[str] = None,
    notes: Optional[str] = None,
    uploaded_by: Optional[str] = None,
) -> dict:
    """Postgres ON CONFLICT upsert keyed on the
    ux_consensus_unique(company_id, period_label, metric_name) index.

    Returns the row as a dict (post-upsert state). Caller is responsible
    for awaiting db.commit() — kept transactional so bulk loads in the
    extractor can commit once at the end."""
    stmt = pg_insert(ConsensusExpectation).values(
        id=uuid.uuid4(),
        company_id=company_id,
        period_label=period_label,
        metric_name=metric_name,
        consensus_value=consensus_value,
        unit=unit,
        source=source,
        notes=notes,
        uploaded_by=uploaded_by,
    ).on_conflict_do_update(
        index_elements=["company_id", "period_label", "metric_name"],
        set_={
            "consensus_value": consensus_value,
            "unit":            unit,
            "source":          source,
            "notes":           notes,
            "uploaded_by":     uploaded_by,
        },
    ).returning(ConsensusExpectation)
    res = await db.execute(stmt)
    row = res.scalar_one()
    return consensus_to_dict(row)
