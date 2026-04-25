"""
Self-healing raw-file fetch + eviction for the storage/raw/ cache.

Design principle: the raw file is a CACHE, not the source of truth.
Document.source_url is always authoritative. Any consumer of a raw file
should call ensure_local_file() first — it returns a path that is
guaranteed to exist on disk (or raises) by lazy-downloading from
source_url when the local copy is missing.

This lets a small persistent volume host an arbitrary corpus: old files
get evicted when disk pressure rises, and the next time they're accessed
they're re-fetched on demand. No user-visible 404s, no manual repair.

Replaces the inline re-download block previously in document_parser.py
(lines 296-334 of the old version) so the /file endpoint and any future
consumers share one code path.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


_DOWNLOAD_HEADERS = {
    "User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com",
}
_DOWNLOAD_TIMEOUT = 60.0


async def ensure_local_file(document) -> Path:
    """Return a Path to the document's raw file, lazy-downloading from
    Document.source_url if the local cache copy is missing.

    Raises FileNotFoundError when no file_path is set, no source_url is
    available, or the download fails. Callers can choose to catch this
    and degrade gracefully (e.g. skip the doc) rather than crash.
    """
    file_path = getattr(document, "file_path", None)
    if not file_path:
        raise FileNotFoundError("Document has no file_path on record")

    p = Path(file_path)
    if p.exists() and p.stat().st_size > 0:
        return p

    source_url = getattr(document, "source_url", None)
    if not source_url:
        raise FileNotFoundError(
            f"File missing at {file_path} and document has no source_url — "
            "cannot lazy-restore. Re-upload manually."
        )

    download_url = _resolve_edgar_index(source_url)
    logger.info("[DOC_FETCH] Restoring %s from %s", file_path, download_url[:80])

    try:
        async with httpx.AsyncClient(
            timeout=_DOWNLOAD_TIMEOUT,
            follow_redirects=True,
            headers=_DOWNLOAD_HEADERS,
        ) as client:
            resp = await client.get(download_url)
            resp.raise_for_status()
            content = resp.content
    except Exception as exc:
        raise FileNotFoundError(
            f"File missing at {file_path} and lazy download failed: {exc}"
        ) from exc

    if not content:
        raise FileNotFoundError(f"File missing at {file_path} and source returned empty body")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    logger.info("[DOC_FETCH] Restored %d bytes to %s", len(content), file_path)
    return p


def _resolve_edgar_index(source_url: str) -> str:
    """EDGAR stores some filings under a -index.htm page that lists the
    actual documents. When we see one, follow it once to the primary
    document. Keeps the old inline behaviour from document_parser.py.
    """
    if "-index.htm" not in source_url or "sec.gov" not in source_url:
        return source_url
    try:
        resp = httpx.get(source_url, headers=_DOWNLOAD_HEADERS, timeout=15.0)
        base = source_url.rsplit("/", 1)[0]
        links = re.findall(r'href="([^"]+\.htm)"', resp.text)
        primary = [l for l in links if "-index" not in l and l.endswith(".htm")]
        if primary:
            return primary[0] if primary[0].startswith("http") else f"{base}/{primary[0]}"
    except Exception as exc:
        logger.warning("[DOC_FETCH] EDGAR index resolve failed: %s", str(exc)[:150])
    return source_url


# ─────────────────────────────────────────────────────────────────
# Eviction — keep disk usage bounded
# ─────────────────────────────────────────────────────────────────

async def ensure_pdf_for_native(document) -> Optional[Path]:
    """Return a PDF path suitable for Claude's native PDF document block.

    For PDFs: returns the existing file_path (lazy-restored if missing).
    For HTML: converts via WeasyPrint and caches as
    {file_path}.pdf alongside the source. The cached PDF survives until
    the eviction loop reclaims it (same lifecycle as the source HTML).

    Returns None when:
      - The source format isn't supported (.docx, .txt, etc.)
      - The conversion fails (bad HTML, WeasyPrint error)
      - ensure_local_file fails to fetch the source

    Used by services/metric_extractor.py to give Claude's native PDF
    block path SEC iXBRL 10-Ks. Observed: text-path native extraction
    on 240KB iXBRL inputs times out at 300s; native PDF on the same
    content typically completes in 30-90s because Claude's document
    API processes layout-aware PDFs more efficiently than equivalent
    text payloads.
    """
    import asyncio as _aio

    try:
        src = await ensure_local_file(document)
    except FileNotFoundError as exc:
        logger.warning("[DOC_FETCH] ensure_pdf_for_native: source unavailable: %s", str(exc)[:200])
        return None

    suffix = src.suffix.lower()
    if suffix == ".pdf":
        return src
    if suffix not in (".htm", ".html", ".xhtml"):
        return None

    # Cache PDF alongside source. .name + ".pdf" preserves the original
    # extension in the filename for easier debugging (foo.html → foo.html.pdf).
    pdf_path = src.with_name(src.name + ".pdf")
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    # xhtml2pdf is pure-Python (built on reportlab); no Cairo/Pango.
    # CPU-bound, so offload to a worker thread.
    def _convert():
        # Lazy import — xhtml2pdf pulls reportlab + html5lib at import time.
        from xhtml2pdf import pisa

        html = src.read_text(errors="replace")
        # iXBRL filings often start with an <?xml declaration that pisa
        # struggles with — stripping any leading XML processing
        # instruction lets pisa fall through to the html5lib branch.
        html = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", html, count=1)

        # Anthropic's native PDF block has a HARD 100-page-per-doc cap.
        # SEC 10-Ks render to 120-180 pages at default xhtml2pdf layout
        # — fall back to text path when that hits, which then times out.
        # Inject tight @page + body rules so the same content fits in
        # 60-90 pages, well under the cap. This is purely a layout
        # density tweak; the content/text Claude reads is unchanged.
        DENSITY_CSS = """
        <style>
          @page { size: letter; margin: 0.3in; }
          body, p, div, td, th, li { font-size: 7pt; line-height: 1.05; }
          table { font-size: 6.5pt; }
          h1, h2, h3, h4 { font-size: 9pt; margin: 4pt 0 2pt 0; }
          td, th { padding: 1pt 2pt; }
        </style>
        """
        # Inject before </head> if present, else prepend.
        if "</head>" in html.lower():
            html = re.sub(r"</head>", DENSITY_CSS + "</head>", html, count=1, flags=re.IGNORECASE)
        else:
            html = DENSITY_CSS + html

        with open(pdf_path, "wb") as out:
            result = pisa.CreatePDF(html, dest=out, encoding="utf-8")
        if result.err:
            raise RuntimeError(f"pisa returned {result.err} errors")

    try:
        await _aio.to_thread(_convert)
        size = pdf_path.stat().st_size
        logger.info(
            "[DOC_FETCH] HTML→PDF: %s → %s (%.1f KB)",
            src.name, pdf_path.name, size / 1024,
        )
        return pdf_path
    except Exception as exc:
        logger.warning(
            "[DOC_FETCH] HTML→PDF conversion failed for %s: %s",
            src.name, str(exc)[:200],
        )
        # Clean up partial output so the next call retries cleanly.
        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError:
                pass
        return None


async def evict_until_free(
    target_free_bytes: int,
    *,
    raw_root: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """Delete oldest raw files (by mtime) until the volume has at least
    target_free_bytes free, or nothing is left to delete.

    Only touches files under raw_root (defaults to settings.storage_base_path
    + "/raw"). Leaves the Document DB rows alone — they'll lazy-restore
    from source_url on next access.

    Returns {freed_bytes, freed_files, remaining_files, stopped_reason}.
    """
    import shutil
    from configs.settings import settings as _settings

    base = raw_root or (Path(_settings.storage_base_path) / "raw")
    if not base.exists():
        return {"freed_bytes": 0, "freed_files": 0, "remaining_files": 0,
                "stopped_reason": "raw_dir_absent"}

    # Collect (mtime, size, path) sorted oldest-first.
    entries: list[tuple[float, int, Path]] = []
    for p in base.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                entries.append((st.st_mtime, st.st_size, p))
            except OSError:
                continue
    entries.sort(key=lambda e: e[0])

    def _free_bytes() -> int:
        try:
            return shutil.disk_usage(str(base)).free
        except Exception:
            return 0

    freed_bytes = 0
    freed_files = 0
    stopped_reason = "target_met"

    if _free_bytes() >= target_free_bytes:
        return {
            "freed_bytes":      0,
            "freed_files":      0,
            "remaining_files":  len(entries),
            "stopped_reason":   "already_above_target",
        }

    for _mtime, size, path in entries:
        if _free_bytes() >= target_free_bytes:
            break
        if dry_run:
            freed_bytes += size
            freed_files += 1
            continue
        try:
            path.unlink()
            freed_bytes += size
            freed_files += 1
        except OSError as exc:
            logger.warning("[DOC_FETCH] Evict failed for %s: %s", path, exc)
            continue
    else:
        stopped_reason = "ran_out_of_candidates"

    logger.info(
        "[DOC_FETCH] Eviction: freed %d files (%d bytes), reason=%s",
        freed_files, freed_bytes, stopped_reason,
    )
    return {
        "freed_bytes":      freed_bytes,
        "freed_files":      freed_files,
        "remaining_files":  len(entries) - freed_files,
        "stopped_reason":   stopped_reason,
        "dry_run":          dry_run,
    }


async def evict_if_pressure(
    threshold_ratio: float = 0.80,
    target_free_ratio: float = 0.30,
) -> Optional[dict]:
    """Opportunistic eviction: if the volume is more than threshold_ratio
    full, evict oldest files until target_free_ratio is free again.

    Called after successful downloads so the cache never runs right up
    to ENOSPC. Returns None if no eviction needed, otherwise the stats
    dict from evict_until_free.
    """
    import shutil
    from configs.settings import settings as _settings

    base = Path(_settings.storage_base_path)
    try:
        usage = shutil.disk_usage(str(base) if base.exists() else "/")
    except Exception:
        return None

    used_ratio = usage.used / usage.total if usage.total else 0
    if used_ratio < threshold_ratio:
        return None

    target_free_bytes = int(usage.total * target_free_ratio)
    return await evict_until_free(target_free_bytes)
