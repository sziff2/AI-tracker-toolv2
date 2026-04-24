"""
File repair — re-download raw files for Document rows whose file_path no
longer exists on disk.

Background: the container's local filesystem was ephemeral before the
Railway persistent volume was attached (2026-04-23). Every file under
storage/raw/... written before that date was wiped on the next deploy,
even though the Document rows pointing at them remained in Postgres.
The net effect: the pipeline appears to have documents, but /file
returns 404/500, extraction hands empty or stale parse leftovers to the
LLM, and every downstream agent skips or produces zero metrics.

This module iterates Document rows, checks whether file_path exists on
disk, and for anything missing with a usable source_url, re-downloads
from source and writes back to the same path. The Document row itself
is NOT moved — we keep the existing file_path so downstream code that
reads it (parser, /file endpoint, /admin/period-diagnostic) stays
coherent. Checksum is recomputed and parsing_status reset to "pending"
so the repaired doc flows through the parse pipeline again.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Company, Document

logger = logging.getLogger(__name__)


async def repair_missing_files(
    db: AsyncSession,
    *,
    ticker: Optional[str] = None,
    dry_run: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """Re-download raw files for Document rows whose file_path is missing.

    Args:
        db:       AsyncSession — caller owns the lifecycle.
        ticker:   If set, only repair this company's docs (Bloomberg format
                  with exchange suffix, e.g. "ARW US").
        dry_run:  When True (default), count what would be repaired but
                  don't download or touch disk. Use this first to confirm
                  scope before spending network + disk on the real run.
        limit:    Cap the number of rows inspected. Useful for smoke tests.

    Returns:
        {
          "dry_run":          bool,
          "ticker":           str | None,
          "checked":          int,   # Document rows inspected
          "present":          int,   # file on disk, nothing to do
          "missing":          int,   # file absent from disk
          "no_url":           int,   # missing AND no source_url — unrecoverable
          "fixed":            int,   # re-downloaded + written to disk
          "download_failed":  int,
          "failures_sample":  list,  # first 10 failures for triage
        }
    """
    from services.harvester.dispatcher import _download
    from sqlalchemy import update

    # Select plain columns (not ORM objects) so the loop body never
    # touches the SQLAlchemy session after a rollback. An ENOSPC mid-loop
    # used to trigger db.rollback(), which expired every ORM attribute
    # in the session's identity map — the next iteration's `doc.id`
    # access then tried an async ping from a non-greenlet context and
    # killed the whole task. Tuples are immune to that.
    q = select(
        Document.id,
        Document.file_path,
        Document.source_url,
    )
    if ticker:
        ticker_u = ticker.upper().strip()
        q = q.join(Company, Document.company_id == Company.id).where(
            Company.ticker == ticker_u
        )
    if limit and limit > 0:
        q = q.limit(limit)

    result = await db.execute(q)
    rows = list(result.all())

    stats = {
        "dry_run":         dry_run,
        "ticker":          ticker,
        "checked":         0,
        "present":         0,
        "missing":         0,
        "no_url":          0,
        "fixed":           0,
        "download_failed": 0,
    }
    failures: list[dict] = []

    for row in rows:
        stats["checked"] += 1

        # Plain tuple from the SELECT — no ORM attributes to expire.
        doc_id, file_path, source_url = row
        doc_id_str = str(doc_id)

        if not file_path:
            stats["no_url"] += 1
            failures.append({"doc_id": doc_id_str, "reason": "no_file_path"})
            continue

        p = Path(file_path)
        try:
            if p.exists() and p.stat().st_size > 0:
                stats["present"] += 1
                continue
        except OSError:
            # Treat stat errors the same as missing — we'll try to recover.
            pass

        stats["missing"] += 1

        if not source_url:
            stats["no_url"] += 1
            failures.append({
                "doc_id":    doc_id_str,
                "file_path": file_path,
                "reason":    "no_source_url",
            })
            continue

        if dry_run:
            continue

        try:
            content, _suffix = await _download(source_url)
            if not content:
                raise ValueError("download returned empty body")

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

            # Update via the table instead of ORM attribute writes —
            # keeps the session's identity map empty so rollbacks on a
            # subsequent iteration can't expire anything.
            await db.execute(
                update(Document)
                .where(Document.id == doc_id)
                .values(
                    checksum=hashlib.sha256(content).hexdigest(),
                    parsing_status="pending",
                )
            )
            await db.commit()
            stats["fixed"] += 1
            logger.info(
                "[FILE_REPAIR] Restored %s (%d bytes) from %s",
                file_path, len(content), source_url,
            )
        except Exception as exc:
            try:
                await db.rollback()
            except Exception:
                pass
            stats["download_failed"] += 1
            failures.append({
                "doc_id":     doc_id_str,
                "file_path":  file_path,
                "source_url": source_url,
                "reason":     str(exc)[:300],
            })
            logger.warning(
                "[FILE_REPAIR] Download failed for %s (%s): %s",
                doc_id_str, source_url, str(exc)[:200],
            )

    stats["failures_sample"] = failures[:10]
    return stats
