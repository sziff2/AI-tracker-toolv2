# Code Review: AI-Tracker-Tool v2

**Date:** 2026-03-19
**Reviewer:** Claude (automated)

## Overview

FastAPI-based investment research pipeline with Celery workers, PostgreSQL, and Anthropic LLM integration. Solid architectural foundations — clean service separation, proper async patterns, good job tracking. Significant issues across security, error handling, database design, and testing.

---

## CRITICAL — Security

### 1. CORS wildcard (`apps/api/main.py:51-57`)
```python
allow_origins=["*"]
```
Any origin can call the API. Whitelist specific origins.

### 2. Hardcoded database credentials (`alembic.ini:3`, `docker-compose.yml:10`)
```ini
sqlalchemy.url = postgresql+psycopg2://postgres:postgres@localhost:5432/research_agent
```
Move to environment variables. Never commit defaults.

### 3. No authentication/authorization
All routes are publicly accessible. No user context, roles, or API keys.

### 4. Unbounded file uploads (`apps/api/routes/documents.py:56-61`)
```python
content = await file.read()  # No size limit
```
Add `MAX_UPLOAD_SIZE` validation before reading.

---

## CRITICAL — Bugs

### 5. Undefined function reference (`metric_extractor.py:456`, `documents.py:145`)
Code calls `smart_chunk` but only `_smart_chunk` is defined (leading underscore mismatch).

---

## HIGH — Architecture

### 6. Mega-functions need splitting
- `run_single_pipeline` (background_processor.py) — 175 lines orchestrating parse → extract → compare → surprises → briefing → save
- `extract_by_document_type` (metric_extractor.py) — 88 lines handling routing, extraction, validation, persistence

### 7. Schema changes in app startup instead of migrations (`main.py:33-40`)
```python
for col in ["recommendation", "catalyst", "conviction", ...]:
    await conn.execute(sa_text(f"ALTER TABLE thesis_versions ADD COLUMN IF NOT EXISTS {col} TEXT"))
```
Runs on every startup. Use Alembic migrations properly.

### 8. Global mutable LLM client (`llm_client.py:17-18`)
```python
_client: anthropic.Anthropic | None = None
_executor = ThreadPoolExecutor(max_workers=6)
```
Global state + hardcoded thread count. Use dependency injection.

### 9. Sync-in-async Celery bridge (`apps/worker/tasks.py:75`)
```python
asyncio.run(_async_process(document_id))  # New event loop per task
```
Inefficient. Consider async Celery setup or keep logic sync.

---

## HIGH — Database

### 10. Eager loading everything (`apps/api/models.py:30-34`)
```python
documents = relationship("Document", back_populates="company", lazy="selectin")
```
Listing 50 companies loads ALL their metrics/documents. Use `lazy="select"` and explicit `selectinload()` where needed.

### 11. Missing compound indexes
`ExtractedMetric(company_id, period_label)`, `Document(company_id, period_label)`, `ThesisVersion(company_id, active)` are queried together everywhere but not indexed together.

### 12. Python-side deduplication (`context_builder.py:49-79`)
Fetches extra rows then dedupes in Python. Use SQL window functions instead.

### 13. No connection pool tuning
No `pool_size`, `max_overflow`, `pool_pre_ping` on the async engine.

---

## MEDIUM — Code Quality

### 14. Duplicated metric persistence
Same create-ExtractedMetric-and-add pattern in 3 places across `metric_extractor.py`. Extract to a shared utility.

### 15. Duplicated period normalization
Same Q1/Q2/FY parsing logic in 3+ places. Should be one function in `metric_normaliser.py`.

### 16. Duplicated company lookup
`get_company_or_404` pattern repeated across multiple routes.

### 17. Broad exception swallowing
Many `except Exception` blocks log truncated messages and continue silently:
```python
except Exception as e:
    logger.warning("Failed: %s", str(e)[:100])  # Truncated, then silently continues
```

### 18. No token counting / cost tracking
LLM calls have no usage tracking. A bad batch could cost hundreds of dollars with no visibility.

### 19. Parallel LLM failures silently return empty (`llm_client.py:82-93`)
```python
if isinstance(r, Exception):
    out.append([])  # Chunk lost silently
```

---

## MEDIUM — Observability

### 20. No structured logging
String interpolation, not JSON. No correlation IDs across pipeline steps.

### 21. No health checks for worker/beat
Only the API has a health endpoint.

### 22. No environment validation at startup
Empty API key (`anthropic_api_key: str = ""`) only fails at first LLM call, not at boot.

---

## LOW — Cleanup

### 23. Dead `.bak` files in version control
- `apps/api/routes/portfolio.py.bak`
- `apps/ui/index.html.bak`

### 24. Unused/stub functions
- `_chunk_text` (replaced by `_smart_chunk`)
- `extract_guidance` (returns empty list)
- `scan_sources` Celery task (does nothing, just logs)

---

## Testing Gaps

### 25. Minimal coverage
`test_api.py` only tests health check and company CRUD. No tests for metric extraction, validation, background pipeline, or error cases.

### 26. No test fixtures
Tests use real Postgres, create/drop all tables each run. No factory fixtures.

---

## Priority Action Plan

| Priority | Action | Files |
|----------|--------|-------|
| P0 | Fix CORS whitelist | `main.py` |
| P0 | Remove hardcoded credentials | `alembic.ini`, `docker-compose.yml` |
| P0 | Fix `smart_chunk` → `_smart_chunk` bug | `metric_extractor.py`, `documents.py` |
| P0 | Add file upload size limit | `documents.py` |
| P1 | Add API authentication | `main.py`, routes |
| P1 | Move schema changes to Alembic migrations | `main.py`, `migrations/` |
| P1 | Split mega-functions | `background_processor.py`, `metric_extractor.py` |
| P1 | Add compound DB indexes | models/migrations |
| P1 | Fix eager loading to lazy | `models.py` |
| P2 | Extract duplicated code (persistence, normalization, lookups) | services, routes |
| P2 | Add structured logging + correlation IDs | all services |
| P2 | Add LLM token tracking | `llm_client.py` |
| P2 | Improve error handling (stop swallowing) | all services |
| P3 | Delete .bak files and dead code | various |
| P3 | Add comprehensive test suite | `tests/` |
| P3 | Add startup config validation | `configs/settings.py` |
