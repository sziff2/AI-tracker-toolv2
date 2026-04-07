"""
Harvester Dispatcher

Orchestrates the end-to-end flow for each HarvestCandidate:
  1. Deduplication against harvested_documents (source_url key)
  2. Download PDF/document to temp file
  3. Call ingest_document() (existing pipeline)
  4. Fire start_background_job() → full extraction
  5. Record harvest result in DB

No Slack or external notifications — results visible in the platform UI.
"""

import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AsyncSessionLocal
from apps.api.models import Company, Document, HarvestedDocument
from services.document_ingestion import ingest_document

logger = logging.getLogger(__name__)

_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "application/pdf,*/*",
}
_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# ─────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────

async def _download(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=_DOWNLOAD_HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "").lower()
        suffix = ".pdf" if ("pdf" in ct or url.lower().endswith(".pdf")) else (
            ".html" if "html" in ct else (Path(url).suffix or ".pdf")
        )
        content = resp.content
        if len(content) > _MAX_FILE_BYTES:
            raise ValueError(f"File too large ({len(content) / 1024 / 1024:.1f} MB)")
        return content, suffix


# ─────────────────────────────────────────────────────────────────
# Deduplication + record
# ─────────────────────────────────────────────────────────────────

async def _seen(db: AsyncSession, source_url: str) -> bool:
    from services.doc_utils import normalise_url
    norm = normalise_url(source_url)
    # Check both exact URL and normalised URL
    r = await db.execute(select(HarvestedDocument).where(HarvestedDocument.source_url == source_url))
    if r.scalar_one_or_none():
        return True
    # Also check normalised version in case query params differ
    if norm != source_url:
        r2 = await db.execute(select(HarvestedDocument).where(HarvestedDocument.source_url == norm))
        if r2.scalar_one_or_none():
            return True
    return False


async def _record(
    db: AsyncSession,
    candidate: dict,
    company_id: uuid.UUID,
    document_id: Optional[uuid.UUID],
    ingested: bool,
    error: Optional[str] = None,
) -> None:
    """Persist a harvest attempt. period_label is stored so the
    Documents tab can group entries by period without a DB join."""
    db.add(HarvestedDocument(
        id=uuid.uuid4(),
        company_id=company_id,
        source=candidate["source"],
        source_url=candidate.get("pdf_url") or candidate["source_url"],
        headline=candidate.get("headline", "")[:500],
        period_label=candidate.get("period_label"),   # required by Documents tab
        discovered_at=datetime.now(timezone.utc),
        ingested=ingested,
        document_id=document_id,
        error=error,
    ))
    await db.commit()


# ─────────────────────────────────────────────────────────────────
# Period fallback
# ─────────────────────────────────────────────────────────────────

def _fallback_period(published_at: Optional[datetime]) -> str:
    dt = published_at or datetime.now(timezone.utc)
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}_Q{q}"


# ─────────────────────────────────────────────────────────────────
# Core dispatch
# ─────────────────────────────────────────────────────────────────

async def dispatch_candidates(candidates: list[dict]) -> dict:
    """
    Process HarvestCandidates: dedup → download → ingest → pipeline.

    Returns {"new": int, "skipped": int, "failed": int, "tickers": list}
    """
    summary = {"new": 0, "skipped": 0, "failed": 0, "tickers": set()}
    if not candidates:
        return summary

    async with AsyncSessionLocal() as db:
        for c in candidates:
            ticker     = c["ticker"]
            source_url = c["source_url"]
            headline   = c.get("headline", "")

            # Look up company
            co_r = await db.execute(select(Company).where(Company.ticker == ticker))
            company = co_r.scalar_one_or_none()
            if not company:
                logger.warning("[DISPATCH] Company not found: %s", ticker)
                summary["failed"] += 1
                continue

            # Resolve actual document URL (pdf_url) vs index page (source_url)
            download_url  = c.get("pdf_url") or source_url

            # Deduplicate — check both index URL and actual doc URL
            if await _seen(db, source_url) or (download_url != source_url and await _seen(db, download_url)):
                summary["skipped"] += 1
                continue

            period_label  = c.get("period_label") or _fallback_period(c.get("published_at"))
            c["period_label"] = period_label  # Ensure period is stored in harvested_documents
            document_type = c.get("document_type", "earnings_release")

            # Skip if manual document already exists for this company + period
            # Manual uploads should never be overwritten by harvester
            existing_manual = await db.execute(
                select(Document).where(
                    Document.company_id == company.id,
                    Document.period_label == period_label,
                    Document.source == "manual",
                )
            )
            if existing_manual.scalars().first():
                logger.info(
                    "[DISPATCH] Skipping %s %s — manual document already exists for this period",
                    ticker, period_label
                )
                await _record(db, c, company.id, None, False, "skipped: manual document exists")
                summary["skipped"] += 1
                continue

            # Download
            try:
                file_bytes, suffix = await _download(download_url)
            except Exception as exc:
                logger.warning("[DISPATCH] Download failed %s — %s: %s", ticker, headline[:60], exc)
                await _record(db, c, company.id, None, False, str(exc))
                summary["failed"] += 1
                continue

            # Ingest — clean title before saving
            from services.doc_utils import clean_title
            headline = clean_title(headline) or headline
            safe_ticker = ticker.replace(" ", "_").replace("/", "_")
            filename = f"{safe_ticker}_{period_label}_{document_type}{suffix}"

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                doc = await ingest_document(
                    db=db,
                    company_id=company.id,
                    ticker=ticker,
                    file_path=tmp_path,
                    filename=filename,
                    document_type=document_type,
                    period_label=period_label,
                    title=headline or filename,
                    source=c["source"],
                    source_url=download_url,  # actual document URL, not EDGAR index page
                    published_at=c.get("published_at"),
                )
            except ValueError as exc:
                # Duplicate checksum — already ingested via another route
                logger.info("[DISPATCH] Duplicate checksum %s — %s", ticker, headline[:60])
                await _record(db, c, company.id, None, False, f"duplicate: {exc}")
                summary["skipped"] += 1
                continue
            except Exception as exc:
                logger.error("[DISPATCH] Ingest failed %s — %s: %s", ticker, headline[:60], exc, exc_info=True)
                await _record(db, c, company.id, None, False, str(exc))
                summary["failed"] += 1
                continue

            # Document ingested — do NOT auto-trigger analysis pipeline.
            # User triggers analysis manually from the Results tab to control LLM costs.
            await _record(db, c, company.id, doc.id, True)

            logger.info(
                "[DISPATCH] ✓ %s — %s (%s) → doc %s",
                ticker, headline[:60], period_label, str(doc.id)[:8]
            )
            summary["new"] += 1
            summary["tickers"].add(ticker)

    summary["tickers"] = list(summary["tickers"])
    return summary
