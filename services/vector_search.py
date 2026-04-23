"""
Vector search over DocumentSection embeddings — Tier 3.4 Part 1.

Part 1 ships the service skeleton + feature flag so the module is
importable and unit-testable without touching the hot ingestion path.
Part 2 wires embedding on parse and RAG context assembly.

Design:
- OpenAI text-embedding-3-small (1536 dims, $0.02/M tokens).
- Cosine distance via pgvector's <=> operator, indexed by HNSW.
- search_sections returns (section, distance) pairs; callers decide a
  threshold. Distance is 0 (identical) → 2 (opposite); typical relevant
  matches sit ≤ 0.5.

Feature flag:
    settings.use_pgvector_search: True enables the path.
    When False, search_sections returns []  — callers should fall back
    to the existing keyword search (services/search.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, DocumentSection
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


async def embed_text(text: str) -> list[float] | None:
    """Embed a single piece of text via the OpenAI API.

    Returns None on any failure — callers fall back to keyword search.
    Safe when OPENAI_API_KEY is unset (returns None with a warning).
    """
    if not text or not text.strip():
        return None
    api_key = settings.openai_api_key
    if not api_key:
        logger.warning("embed_text called but OPENAI_API_KEY is not configured")
        return None
    try:
        # Lazy import so the module loads even when openai isn't installed
        # (e.g. minimal test environments).
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=text[:8000],  # ~2k tokens — more than enough per section
        )
        return list(resp.data[0].embedding)
    except Exception as exc:
        logger.warning("embed_text failed: %s", str(exc)[:200])
        return None


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
      - embed_text fails (no API key, rate limit, etc.)

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

    # Build the parameterised SQL. Joining Document gives us title +
    # optional filters on company_id / period_label. The <=> operator
    # returns cosine distance when the column uses vector_cosine_ops.
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
