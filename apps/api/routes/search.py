"""
Global semantic search — LLM-powered keyword search across:
  - document_sections   (parsed paragraphs)
  - extracted_metrics   (KPI data points)
  - management_statements (forward guidance & claims)
  - analyst_notes       (research notes & call notes)

POST /api/v1/search
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.models import (
    Company,
    DocumentSection,
    Document,
    ExtractedMetric,
    ManagementStatement,
    AnalystNote,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

# ── Max hits to pull from DB before LLM ranking ──────────────────
_DB_LIMIT = 40   # rows per corpus
_LLM_CONTEXT_CHARS = 18000  # total chars fed to LLM


class SearchRequest(BaseModel):
    query: str
    ticker: Optional[str] = None   # if set, scope to one company
    limit: int = 8                  # results to return in response


class SearchResult(BaseModel):
    type: str           # section | metric | statement | note
    ticker: str
    company_name: str
    period: Optional[str]
    title: str
    snippet: str
    relevance_comment: str


class SearchResponse(BaseModel):
    query: str
    answer: str
    results: list[SearchResult]
    total_hits: int


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _build_ilike(col, terms: list[str]):
    """Return OR of ILIKE clauses for each search term."""
    return or_(*[col.ilike(f"%{t}%") for t in terms])


def _terms(query: str) -> list[str]:
    """Split query into meaningful tokens (3+ chars)."""
    raw = query.strip().split()
    tokens = [t for t in raw if len(t) >= 3]
    # Always include the full query as a phrase too
    if len(raw) > 1:
        tokens.append(query.strip())
    return tokens[:8]  # cap to avoid huge SQL


# ─────────────────────────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def global_search(body: SearchRequest, db: AsyncSession = Depends(get_db)):
    if not body.query or len(body.query.strip()) < 2:
        raise HTTPException(400, "Query must be at least 2 characters")

    terms = _terms(body.query)
    ticker_filter = body.ticker.upper() if body.ticker else None

    # Build company filter if scoped
    company_ids = None
    company_map: dict[str, tuple[str, str]] = {}  # company_id -> (ticker, name)

    if ticker_filter:
        co_q = await db.execute(select(Company).where(Company.ticker == ticker_filter))
        co = co_q.scalar_one_or_none()
        if not co:
            raise HTTPException(404, f"Company {ticker_filter} not found")
        company_ids = [co.id]
        company_map[str(co.id)] = (co.ticker, co.name)
    else:
        co_q = await db.execute(select(Company))
        for co in co_q.scalars().all():
            company_map[str(co.id)] = (co.ticker, co.name)

    # Helper to get ticker/name from company_id
    def co_info(cid) -> tuple[str, str]:
        return company_map.get(str(cid), ("??", "Unknown"))

    hits: list[dict] = []

    # ── 1. Document Sections ────────────────────────────────────
    sec_q = select(DocumentSection, Document).join(
        Document, DocumentSection.document_id == Document.id
    ).where(_build_ilike(DocumentSection.text_content, terms))

    if company_ids:
        sec_q = sec_q.where(Document.company_id.in_(company_ids))

    sec_q = sec_q.order_by(
        func.length(DocumentSection.text_content)
    ).limit(_DB_LIMIT)

    sec_rows = await db.execute(sec_q)
    for sec, doc in sec_rows.all():
        ticker, name = co_info(doc.company_id)
        snippet = (sec.text_content or "")[:400].strip()
        hits.append({
            "type": "section",
            "ticker": ticker,
            "company_name": name,
            "period": doc.period_label,
            "title": f"{doc.title or doc.document_type} — {sec.section_title or sec.section_type or 'paragraph'}",
            "snippet": snippet,
        })

    # ── 2. Extracted Metrics ────────────────────────────────────
    met_q = select(ExtractedMetric, Company).join(
        Company, ExtractedMetric.company_id == Company.id
    ).where(
        or_(
            _build_ilike(ExtractedMetric.metric_name, terms),
            _build_ilike(ExtractedMetric.metric_text, terms),
        )
    )

    if company_ids:
        met_q = met_q.where(ExtractedMetric.company_id.in_(company_ids))

    met_q = met_q.order_by(ExtractedMetric.created_at.desc()).limit(_DB_LIMIT)

    met_rows = await db.execute(met_q)
    for met, co in met_rows.all():
        val = met.metric_text or (str(float(met.metric_value)) if met.metric_value else "")
        unit = met.unit or ""
        snippet = f"{met.metric_name}: {val} {unit}".strip()
        if met.source_snippet:
            snippet += f" | Source: {met.source_snippet[:200]}"
        hits.append({
            "type": "metric",
            "ticker": co.ticker,
            "company_name": co.name,
            "period": met.period_label,
            "title": f"KPI: {met.metric_name}",
            "snippet": snippet,
        })

    # ── 3. Management Statements ────────────────────────────────
    stmt_q = select(ManagementStatement, Company).join(
        Company, ManagementStatement.company_id == Company.id
    ).where(
        or_(
            _build_ilike(ManagementStatement.statement_text, terms),
            _build_ilike(ManagementStatement.target_metric, terms),
            _build_ilike(ManagementStatement.source_snippet, terms),
        )
    )

    if company_ids:
        stmt_q = stmt_q.where(ManagementStatement.company_id.in_(company_ids))

    stmt_q = stmt_q.order_by(ManagementStatement.created_at.desc()).limit(_DB_LIMIT)

    stmt_rows = await db.execute(stmt_q)
    for stmt, co in stmt_rows.all():
        snippet = stmt.statement_text[:400]
        if stmt.target_metric:
            snippet = f"[{stmt.speaker or 'Mgmt'} on {stmt.target_metric}] " + snippet
        status_tag = f" | Status: {stmt.status}" if stmt.status else ""
        hits.append({
            "type": "statement",
            "ticker": co.ticker,
            "company_name": co.name,
            "period": stmt.statement_date,
            "title": f"Mgmt Statement: {stmt.category} — {stmt.speaker or 'Speaker unknown'}",
            "snippet": snippet + status_tag,
        })

    # ── 4. Analyst Notes ────────────────────────────────────────
    note_q = select(AnalystNote, Company).join(
        Company, AnalystNote.company_id == Company.id
    ).where(
        or_(
            _build_ilike(AnalystNote.content, terms),
            _build_ilike(AnalystNote.title, terms),
        )
    )

    if company_ids:
        note_q = note_q.where(AnalystNote.company_id.in_(company_ids))

    note_q = note_q.order_by(AnalystNote.created_at.desc()).limit(_DB_LIMIT)

    note_rows = await db.execute(note_q)
    for note, co in note_rows.all():
        hits.append({
            "type": "note",
            "ticker": co.ticker,
            "company_name": co.name,
            "period": None,
            "title": f"Analyst Note: {note.title or note.note_type}",
            "snippet": (note.content or "")[:400],
        })

    total_hits = len(hits)

    if not hits:
        return SearchResponse(
            query=body.query,
            answer="No matches found across documents, metrics, statements, or notes.",
            results=[],
            total_hits=0,
        )

    # ── LLM: rank and synthesise ────────────────────────────────
    # Build compact context block
    context_parts = []
    char_budget = _LLM_CONTEXT_CHARS
    for i, h in enumerate(hits):
        block = (
            f"[{i+1}] TYPE={h['type']} TICKER={h['ticker']} PERIOD={h['period'] or '—'}\n"
            f"TITLE: {h['title']}\n"
            f"TEXT: {h['snippet']}\n"
        )
        if len(block) > char_budget:
            break
        context_parts.append(block)
        char_budget -= len(block)

    context_str = "\n---\n".join(context_parts)

    prompt = f"""You are an investment research assistant at Oldfield Partners.
A user has searched for: "{body.query}"

Below are up to {len(context_parts)} matched passages from the research database, covering document extracts, KPI metrics, management statements, and analyst notes.

{context_str}

Your task:
1. Write a concise 2-4 sentence ANSWER to the user's query based strictly on the data above. Do not invent figures.
2. Then return the {min(body.limit, len(hits))} most relevant items from the list above as a JSON array.

Respond ONLY with valid JSON in this exact structure:
{{
  "answer": "<2-4 sentence answer>",
  "top_indices": [<1-based indices of top {min(body.limit, len(hits))} results, most relevant first>]
}}

If the query is ambiguous or the data is insufficient, say so in the answer field.
Do not include any text outside the JSON object."""

    try:
        from services.llm_client import call_llm
        import json as _json

        raw = call_llm(prompt, max_tokens=1000, feature="search_rerank")
        # Strip markdown fences if present
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = _json.loads(clean)
        answer = parsed.get("answer", "")
        top_indices = parsed.get("top_indices", list(range(1, min(body.limit, len(hits)) + 1)))
    except Exception as e:
        logger.warning("LLM ranking failed, falling back to raw order: %s", e)
        answer = f"Found {total_hits} matches for '{body.query}'."
        top_indices = list(range(1, min(body.limit, len(hits)) + 1))

    # Build ranked results
    results: list[SearchResult] = []
    seen = set()
    for idx in top_indices:
        i = idx - 1  # convert to 0-based
        if i < 0 or i >= len(hits):
            continue
        h = hits[i]
        key = h["type"] + h["ticker"] + (h["period"] or "") + h["snippet"][:80]
        if key in seen:
            continue
        seen.add(key)
        results.append(SearchResult(
            type=h["type"],
            ticker=h["ticker"],
            company_name=h["company_name"],
            period=h["period"],
            title=h["title"],
            snippet=h["snippet"],
            relevance_comment="",
        ))

    return SearchResponse(
        query=body.query,
        answer=answer,
        results=results,
        total_hits=total_hits,
    )
