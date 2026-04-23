"""
Backfill DocumentSection.embedding for rows ingested before Tier 3.4
shipped.

Strategy:
  - Page through document_sections where embedding IS NULL and
    text_content IS NOT NULL (skip empty shells)
  - Encode in batches of N via services.vector_search.embed_texts
    (batch encoding is ~10x faster than one-at-a-time)
  - UPDATE each row's .embedding column
  - Commit every chunk so a crash only loses the in-flight chunk

Dry-run by default — prints how many rows would be embedded and the
first chunk's keys so you can eyeball before committing. Pass --apply
to actually write.

Idempotent: re-running after success is a no-op because the WHERE
clause excludes rows that already have embeddings.

Usage (against production Railway PG):
  python scripts/backfill_embeddings.py                 # dry-run, full
  python scripts/backfill_embeddings.py --ticker BNZL LN  # one ticker
  python scripts/backfill_embeddings.py --apply         # commit, all
  python scripts/backfill_embeddings.py --apply --limit 200  # cap
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make the repo root importable when this script is run directly
# (e.g. via `railway run python scripts/backfill_embeddings.py`).
# Without this, the top-level `apps` / `services` / `configs` packages
# aren't on sys.path and imports fail with ModuleNotFoundError.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _backfill(
    db: AsyncSession,
    ticker: str | None,
    chunk_size: int,
    limit: int | None,
    apply: bool,
) -> dict:
    from apps.api.models import Company, Document, DocumentSection
    from services.vector_search import embed_texts

    # Build the filter once. We UPDATE by id inside the loop so there's no
    # risk of the chunk windowing drifting when rows get an embedding.
    base_q = (
        select(DocumentSection.id, DocumentSection.text_content)
        .where(DocumentSection.embedding.is_(None))
        .where(DocumentSection.text_content.is_not(None))
    )
    if ticker:
        # Scope to a single company via the Document→Company join.
        cq = await db.execute(select(Company).where(Company.ticker == ticker))
        company = cq.scalar_one_or_none()
        if not company:
            logger.error("Ticker %s not found", ticker)
            return {"status": "error", "reason": f"ticker {ticker} not found"}
        base_q = (
            base_q.join(Document, Document.id == DocumentSection.document_id)
                  .where(Document.company_id == company.id)
        )
        logger.info("Scoping to ticker %s (company_id=%s)", ticker, company.id)

    # Count total candidates first so the log shows progress.
    from sqlalchemy import func as sa_func
    count_q = (
        select(sa_func.count(DocumentSection.id))
        .where(DocumentSection.embedding.is_(None))
        .where(DocumentSection.text_content.is_not(None))
    )
    if ticker:
        count_q = (
            count_q.join(Document, Document.id == DocumentSection.document_id)
                   .where(Document.company_id == company.id)
        )
    total_candidates = (await db.execute(count_q)).scalar() or 0
    logger.info("Candidate sections (embedding IS NULL): %d", total_candidates)
    if total_candidates == 0:
        return {"status": "complete", "embedded": 0, "candidates": 0}

    if limit is not None:
        total_candidates = min(total_candidates, limit)
        base_q = base_q.limit(limit)

    # Pull all ids first (cheap — text_content not needed yet in the id list).
    # Iterate in chunks; fetch text just for each chunk to keep memory bounded.
    id_rows = (await db.execute(base_q)).all()
    all_ids = [r[0] for r in id_rows]
    logger.info("Will process %d rows in chunks of %d", len(all_ids), chunk_size)

    embedded = 0
    failed = 0
    for i in range(0, len(all_ids), chunk_size):
        chunk_ids = all_ids[i:i+chunk_size]
        chunk_q = await db.execute(
            select(DocumentSection.id, DocumentSection.text_content)
            .where(DocumentSection.id.in_(chunk_ids))
        )
        rows = chunk_q.all()
        texts = [r.text_content or "" for r in rows]
        vecs = await embed_texts(texts)
        if not apply:
            non_empty = sum(1 for t in texts if t.strip())
            got = sum(1 for v in vecs if v is not None)
            logger.info("[DRY] chunk %d-%d: %d non-empty, would embed %d",
                        i, i+len(rows), non_empty, got)
            embedded += got
            continue

        # Update each row's embedding column in this chunk
        chunk_ok = 0
        for row, vec in zip(rows, vecs):
            if vec is None:
                failed += 1
                continue
            await db.execute(
                update(DocumentSection)
                .where(DocumentSection.id == row.id)
                .values(embedding=vec)
            )
            chunk_ok += 1
        await db.commit()
        embedded += chunk_ok
        logger.info("Committed chunk %d-%d: %d embedded (running total %d/%d)",
                    i, i+len(rows), chunk_ok, embedded, len(all_ids))

    return {
        "status":     "complete" if apply else "dry_run",
        "candidates": total_candidates,
        "embedded":   embedded,
        "failed":     failed,
    }


async def _main(args):
    from apps.api.database import AsyncSessionLocal
    from configs.settings import settings

    if not settings.use_pgvector_search:
        logger.warning(
            "USE_PGVECTOR_SEARCH is False. The backfill will still load the "
            "embedding model and write rows, which is what you probably want "
            "as prep for flipping the flag. Continuing."
        )

    async with AsyncSessionLocal() as db:
        result = await _backfill(
            db,
            ticker=args.ticker,
            chunk_size=args.chunk_size,
            limit=args.limit,
            apply=args.apply,
        )
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=None, help="Restrict to one company (e.g. 'BNZL LN')")
    p.add_argument("--chunk-size", type=int, default=64, help="Rows per encode + commit batch")
    p.add_argument("--limit", type=int, default=None, help="Cap total rows processed")
    p.add_argument("--apply", action="store_true", help="Commit changes (default is dry-run)")
    args = p.parse_args()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        sys.exit(130)
