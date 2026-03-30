# CLAUDE.md — Project Context for Claude Code

## What This Is
Investment research platform for Oldfield Partners (buy-side value fund). Tracks ~39 companies across multiple portfolios. Deployed on Railway.

## Key Architecture Decisions
- **Single HTML file UI** (`apps/ui/index.html`) — no framework, no build step. All JS inline.
- **Background jobs run in-process** via `asyncio.create_task`, not Celery (Celery is wired but Redis not deployed yet).
- **No file_content in DB** — raw PDFs are NOT stored in PostgreSQL to save space. Parsed text lives in `document_sections`.
- **Tables auto-created on startup** via `Base.metadata.create_all` + manual ALTER TABLE migrations in `apps/api/main.py` lifespan.

## Ticker Format
Tickers use Bloomberg format with exchange suffix: `LKQ US`, `BNZL LN`, `3679 JP`, `005930 KS`.
**BP LN** was renamed from `BP/ LN` — slashes break FastAPI path routing. The harvester routes use `{ticker:path}` but other routes use `{ticker}`.

## Document Sourcing Pipeline
Priority: SEC EDGAR → Regex scraper → LLM scraper
- EDGAR: CIK stored on Company model, auto-looked up from SEC company_tickers.json
- IR scraper: regex-based, fast but breaks on SPA pages
- LLM scraper: sends page HTML to Claude, handles complex sites, costs ~$0.01-0.05/scan

## Analysis Pipeline
`Document → parse (PDF/HTML/DOCX) → extract metrics → compare thesis → detect surprises → synthesise`
- Already-parsed documents are skipped on re-run (checks sections_count + metrics_count)
- `resynthesise` endpoint skips parsing/extraction, just re-runs synthesis with updated thesis
- FY is treated as equivalent to Q4 for period comparisons

## Common Patterns
- `enc(ticker)` in JS = `encodeURIComponent`
- `get_company_or_404(db, ticker)` in Python routes
- `_clean_ticker(raw)` = strip + uppercase
- Period format: `2025_Q1`, `2025_FY`, `2025_H1`
- `suggestPeriod()` / `suggestPeriodFromDate()` in JS for auto-period from filing date

## Testing
```bash
DATABASE_URL="postgresql+asyncpg://x:x@localhost/x" pytest tests/test_services.py -v
```
52 tests, all passing. Tests don't need a real DB — the dummy URL satisfies import-time engine creation.

## Deployment
Push to main → Railway auto-deploys. UI served with no-cache headers to prevent stale JS.
Production URL: https://ai-tracker-tool-production.up.railway.app

## Things to Watch
- Railway PostgreSQL has limited storage — `file_content` column should stay NULL
- LLM timeout is 120s on the HTTP client, 90s on async calls
- The `parseCSVLine` function handles quoted fields with commas (e.g. "Consumer, Cyclical")
- Scenario snapshots are saved on every PUT to `/companies/{ticker}/scenarios`
- Feedback is auto-promoted to Prompt Lab on save (no manual promote step)
