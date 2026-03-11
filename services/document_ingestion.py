"""
Document Ingestion Service (§7)

Responsibilities:
  - Accept uploaded files (manual)
  - Store raw documents in the file-system layout
  - Create document DB records
  - Deduplicate by checksum
"""

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Company, Document
from configs.settings import settings

logger = logging.getLogger(__name__)


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _storage_dir(ticker: str, period_label: str) -> Path:
    base = Path(settings.storage_base_path) / "raw" / ticker / period_label
    base.mkdir(parents=True, exist_ok=True)
    return base


async def ingest_document(
    db: AsyncSession,
    company_id: uuid.UUID,
    ticker: str,
    file_path: str,
    filename: str,
    document_type: str,
    period_label: str,
    title: str | None = None,
    source: str = "manual",
    source_url: str | None = None,
    published_at: datetime | None = None,
) -> Document:
    """
    Copy uploaded file to storage, create a Document record, and return it.
    Also stores file bytes in DB so they survive Railway redeploys.
    Raises ValueError on duplicate checksum.
    """
    cs = _checksum(file_path)

    # ── Deduplication ────────────────────────────────────────────
    existing = await db.execute(select(Document).where(Document.checksum == cs))
    if existing.scalar_one_or_none():
        raise ValueError(f"Duplicate document (checksum {cs[:12]}…)")

    # ── Read file bytes ──────────────────────────────────────────
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # ── Copy to structured storage ───────────────────────────────
    dest_dir = _storage_dir(ticker, period_label)
    dest_path = dest_dir / filename
    shutil.copy2(file_path, dest_path)
    logger.info("Stored %s → %s", filename, dest_path)

    # ── Create DB record ─────────────────────────────────────────
    doc = Document(
        id=uuid.uuid4(),
        company_id=company_id,
        document_type=document_type,
        title=title or filename,
        period_label=period_label,
        source=source,
        source_url=source_url,
        published_at=published_at or datetime.now(timezone.utc),
        file_path=str(dest_path),
        file_content=file_bytes,
        checksum=cs,
        parsing_status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc
