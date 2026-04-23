"""
Vector search over DocumentSection embeddings — Tier 3.4 Part 1.

Part 1 ships the service skeleton + feature flag so the module is
importable and unit-testable without touching the hot ingestion path.
Part 2 wires embedding on parse and RAG context assembly.

Design:
- Local sentence-transformers (BAAI/bge-small-en-v1.5, 384 dims). Runs
  in-process via torch CPU. No third-party API, no per-token cost, data
  never leaves the container.
- Cosine distance via pgvector's <=> operator, indexed by HNSW.
- search_sections returns (section, distance) pairs; callers decide a
  threshold. Distance is 0 (identical) → 2 (opposite); typical relevant
  matches sit ≤ 0.5.

Model is loaded lazily once per process (~130MB weights on first use,
then cached). Inference is CPU-bound — we offload to a thread via
asyncio.to_thread so we don't block the event loop for 20-50ms per call.

Feature flag:
    settings.use_pgvector_search: True enables the path.
    When False, search_sections returns [] — callers should fall back
    to the existing keyword search (services/search.py).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One section-level search result."""
    section_id:    str
    document_id:   str
    document_title: str
    section_title: Optional[str]
    page_number:   Optional[int]
    snippet:       str           # first 500 chars of text_content
    distance:      float         # 0 (identical) → 2 (opposite)
    company_id:    Optional[str] = None
    period_label:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "section_id":     self.section_id,
            "document_id":    self.document_id,
            "document_title": self.document_title,
            "section_title":  self.section_title,
            "page_number":    self.page_number,
            "snippet":        self.snippet,
            "distance":       round(self.distance, 4),
            "company_id":     self.company_id,
            "period_label":   self.period_label,
        }


# ─────────────────────────────────────────────────────────────────
# Lazy model loader — one instance per process
# ─────────────────────────────────────────────────────────────────

_model = None  # type: ignore[var-annotated]


def _get_model():
    """Lazy-load the sentence-transformers model. Cached for the life
    of the process. Returns None on failure so callers fall back to
    keyword search rather than crashing."""
    global _model
    if _model is not None:
        return _model
    try:
        # Lazy import so the module loads even when sentence-transformers
        # isn't installed (minimal test environments).
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s (first call — ~130MB)", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        return _model
    except Exception as exc:
        logger.warning("Embedding model load failed: %s", str(exc)[:200])
        return None


async def embed_text(text: str) -> list[float] | None:
    """Embed a single piece of text with the local sentence-transformers
    model. Returns None on empty input or model-load failure. Offloaded
    to a worker thread so the CPU-bound encode doesn't block the event
    loop.
    """
    if not text or not text.strip():
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        # bge-* models recommend normalize_embeddings=True so cosine
        # distance in pgvector matches the paper's benchmark setup.
        vec = await asyncio.to_thread(
            model.encode, text[:8000],   # ~2k tokens — plenty per section
            normalize_embeddings=True,
        )
        return list(map(float, vec))
    except Exception as exc:
        logger.warning("embed_text failed: %s", str(exc)[:200])
        return None


async def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Batch-embed many pieces of text in a single encode call.

    ~10x faster than calling embed_text in a loop because
    sentence-transformers batches inference internally. Returns a list
    aligned to the input — None entries for empty strings or failure.

    Called from the parser (one call per document, ~30-100 sections)
    and the backfill script (batched by N rows).
    """
    if not texts:
        return []
    # Short-circuit for empties while preserving index alignment.
    cleaned = [t[:8000] if (t and t.strip()) else "" for t in texts]
    if not any(cleaned):
        return [None] * len(texts)

    model = _get_model()
    if model is None:
        return [None] * len(texts)
    try:
        vecs = await asyncio.to_thread(
            model.encode, cleaned,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out: list[list[float] | None] = []
        for raw, vec in zip(cleaned, vecs):
            if not raw:
                out.append(None)
            else:
                out.append(list(map(float, vec)))
        return out
    except Exception as exc:
        logger.warning("embed_texts failed: %s", str(exc)[:200])
        return [None] * len(texts)


async def search_sections(
    db: AsyncSession,
    query: str,
    company_id: Optional[str] = None,
    period_label: Optional[str] = None,
    k: int = 5,
) -> list[SearchHit]:
    """Top-k semantic search over DocumentSection.embedding.

    Returns [] when:
      - Feature flag is off
      - Query is empty
      - embed_text fails (model-load issue, encode error, etc.)

    Callers should fall back to the existing keyword search in those
    cases so search is never "silently broken" — the UI still returns
    results, just via the less-clever path.
    """
    if not settings.use_pgvector_search:
        return []
    if not query or not query.strip():
        return []

    vec = await embed_text(query)
    if vec is None:
        return []

    # Parameterised SQL. Joining Document gives us title + optional
    # filters on company_id / period_label. The <=> operator returns
    # cosine distance when the column uses vector_cosine_ops.
    sql = """
        SELECT
            ds.id::text AS section_id,
            ds.document_id::text AS document_id,
            COALESCE(d.title, '') AS document_title,
            ds.section_title,
            ds.page_number,
            LEFT(COALESCE(ds.text_content, ''), 500) AS snippet,
            (ds.embedding <=> :qvec) AS distance,
            d.company_id::text AS company_id,
            d.period_label AS period_label
        FROM document_sections ds
        JOIN documents d ON d.id = ds.document_id
        WHERE ds.embedding IS NOT NULL
    """
    params: dict = {"qvec": _format_vector(vec), "k": k}
    if company_id:
        sql += " AND d.company_id = :company_id"
        params["company_id"] = str(company_id)
    if period_label:
        sql += " AND d.period_label = :period_label"
        params["period_label"] = period_label
    sql += " ORDER BY ds.embedding <=> :qvec LIMIT :k"

    try:
        result = await db.execute(sa_text(sql), params)
        rows = result.fetchall()
    except Exception as exc:
        logger.warning("search_sections query failed: %s", str(exc)[:200])
        return []

    hits: list[SearchHit] = []
    for row in rows:
        hits.append(SearchHit(
            section_id=row.section_id,
            document_id=row.document_id,
            document_title=row.document_title,
            section_title=row.section_title,
            page_number=row.page_number,
            snippet=row.snippet,
            distance=float(row.distance),
            company_id=row.company_id,
            period_label=row.period_label,
        ))
    return hits


def _format_vector(vec: list[float]) -> str:
    """pgvector accepts bracket-delimited comma-separated floats as the
    canonical text form. SQLAlchemy passes the string; Postgres casts."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
