# CLAUDE.md — Project Context for Claude Code

## What This Is
Investment research platform for Oldfield Partners (buy-side value fund). Tracks ~39 companies across multiple portfolios. Deployed on Railway.

## Key Architecture Decisions
- **Single HTML file UI** (`apps/ui/index.html`) — no framework, no build step. All JS inline.
- **Background jobs**: Celery + Redis for scheduled tasks (weekly harvest via Celery Beat). One-off background work uses `asyncio.create_task` in-process.
- **No file_content in DB** — raw PDFs are NOT stored in PostgreSQL to save space. Parsed text lives in `document_sections`.
- **Tables auto-created on startup** via `Base.metadata.create_all` + manual ALTER TABLE migrations in `apps/api/main.py` lifespan.

## Ticker Format
Tickers use Bloomberg format with exchange suffix: `LKQ US`, `BNZL LN`, `3679 JP`, `005930 KS`.
**BP LN** was renamed from `BP/ LN` — slashes break FastAPI path routing. The harvester routes use `{ticker:path}` but other routes use `{ticker}`.

## Document Sourcing Pipeline
Priority: SEC EDGAR → Investegate RNS → IR regex scraper → LLM scraper
- EDGAR: CIK configured per company in `EDGAR_SOURCES` dict (`services/harvester/sources/sec_edgar.py`)
- Investegate: UK RNS announcements, configured in `INVESTEGATE_SOURCES` dict (`services/harvester/sources/investegate.py`)
- IR scraper: regex-based, fast but breaks on SPA pages. Supports multiple URLs per company.
- LLM scraper: sends page HTML to Claude, handles complex sites, costs ~$0.01-0.05/scan
- ScrapingBee: optional JS rendering for Cloudflare-blocked IR pages (API key in env)
- BP LN and SHEL LN run both EDGAR and Investegate (dual source)
- IR scraper runs for all companies with `ir_docs_url` set, even if EDGAR/Investegate also runs

## Weekly Auto-Harvest
- Celery Beat runs every Monday 06:00 UTC
- Skips LLM scraper to contain costs (only EDGAR + Investegate + IR regex)
- Saves a `HarvestReport` to DB with per-company breakdown
- Posts summary to Microsoft Teams via webhook (`TEAMS_WEBHOOK_URL` env var)
- Manual trigger: `POST /harvester/run-weekly`
- Reports: `GET /harvester/reports`, `GET /harvester/reports/latest`

## Agent Architecture
The platform is transitioning to a modular agent architecture where each analysis step is an independent agent.

### Directory Structure
- `agents/` — agent package
  - `base.py` — `BaseAgent` abstract class, `AgentResult` dataclass, `AgentTier` enum
  - `registry.py` — `AgentRegistry` with auto-discovery and dependency-ordered execution

### Key Concepts
- **Agents are stateless** — receive inputs dict, call LLM, return `AgentResult`. No DB access inside agents.
- **Orchestrator handles DB** — reads inputs, calls agents, persists results to `agent_outputs` table.
- **Agent tiers**: EXTRACTION (parse docs), ANALYSIS (compare/score), SYNTHESIS (generate outputs), META (QC/calibration).
- **Model routing**: Extraction/Meta agents use Haiku (fast/cheap), Analysis/Synthesis agents use Sonnet (quality). Override per-agent via `model_override`.
- **Dependency ordering**: Agents declare `depends_on` list; `AgentRegistry.get_execution_order()` does topological sort.
- **Predictions**: Agents can return trackable predictions via `extract_predictions()`, resolved later for calibration.

### DB Tables
- `agent_outputs` — one row per agent execution with output_json, confidence, qc_score
- `agent_calibration` — per-agent accuracy tracking
- `pipeline_runs` — audit trail: one row per "Run Analysis" click with full per-agent breakdown
- `context_contracts` — macro assumptions shared across agents
- `sector_theses` — per-sector thesis linked to context contracts

### Clean Break Design (Planned)
Existing plumbing (parser, extractor, normaliser, LLM client) stays unchanged. Everything above it — analysis, comparison, synthesis, output — is rebuilt as agents. Current services to be replaced:
- `thesis_comparator.py` → absorbed into Financial Analyst agent
- `surprise_detector.py` → absorbed into Financial Analyst agent
- `output_generator.py` → replaced by Research Agenda + PM Agent
- `background_processor.py` → replaced by Orchestrator

Agent tiers in the full architecture: TASK → DOCUMENT → SPECIALIST → INDUSTRY → SECTOR → MACRO → PORTFOLIO → META. Each agent declares a `layer` (0-9) for execution ordering within a pipeline run.

### Context Contract (Planned)
Every agent receives a shared `ContextContract` containing macro assumptions (regime, rates, credit, growth, FX, commodities, inflation, geopolitical risks) that no agent may contradict. Set by macro agents with analyst overrides. Stored in `context_contracts` table.

This ensures consistency: if the analyst sets "higher for longer rates", both the REIT bear case and bank bull case respect that assumption.

### Thesis Generation Cascade (Planned)
Thesis is generated as a structured document with pillars (not free text). Each pillar links to macro dependencies and sector views from the context contract. When macro assumptions change, affected theses are flagged for re-evaluation.

### Reference Docs
- `Dev plans/_agent-architecture-clean-break.md` — full agent design, tiers, orchestration
- `Dev plans/_thesis-architecture.md` — context contracts, thesis generation, consistency
- `Dev plans/_0.8_financial-extraction-architecture.md` — pre-segmented extraction pipeline

### Settings
- `agent_default_model` — Sonnet (quality agents)
- `agent_fast_model` — Haiku (extraction/QC agents)
- `agent_max_parallel` — concurrency limit (default 8)
- `agent_pipeline_budget_usd` — per-pipeline spending cap (default $2)

## Financial Extraction Architecture (Planned — 0.8)
The extraction pipeline is being redesigned to pre-segment documents before LLM calls.

### Current Approach (one big prompt)
- Sends full document text to LLM, asks "extract all metrics"
- ~15-25% period misattribution, ~5-10% BS/P&L confusion, ~10% segment errors

### New Approach (pre-segment → parallel targeted extraction → reconciliation)
```
Document → structural parser (no LLM) → FinancialDocumentStructure
  → classify tables (P&L / BS / CF / Segment / KPI / Guidance)
  → parse column headers into periods
  → split multi-period tables into single-period
  → detect currency + unit scale
Then → parallel LLM calls (one per statement × period, focused short prompts)
Then → reconciliation (Q sum vs FY, segment sum vs consolidated, BS equation)
```

### New Files (when built)
- `services/financial_statement_segmenter.py` — structural parsing (no LLM)
- `services/statement_extractors.py` — per-statement type LLM prompts
- `services/extraction_reconciler.py` — cross-checks

### Key Principle
Agents should never see raw financial statements. By the time analysis agents run, data is classified by statement type, tagged with correct period/currency/scale, and reconciled.

### Reference
Full architecture: `Dev plans/_0.8_financial-extraction-architecture.md`

## Analysis Pipeline
`Document → parse (PDF/HTML/DOCX) → extract metrics → compare thesis → detect surprises → synthesise`
- Already-parsed documents are skipped on re-run (checks sections_count + metrics_count)
- `resynthesise` endpoint skips parsing/extraction, just re-runs synthesis with updated thesis
- FY is treated as equivalent to Q4 for period comparisons

## Common Patterns
- `enc(ticker)` in JS = `encodeURIComponent`
- `get_company_or_404(db, ticker)` in Python routes
- `_clean_ticker(raw)` = strip + uppercase
- Period format: `2025_Q1`, `2025_Q2`, `2025_Q3`, `2025_Q4` (FY→Q4, H1→Q2, H2→Q4 mapped everywhere)
- `suggestPeriod()` / `suggestPeriodFromDate()` in JS for auto-period from filing date

## Testing
```bash
DATABASE_URL="postgresql+asyncpg://x:x@localhost/x" pytest tests/test_services.py -v
```
52 tests, all passing. Tests don't need a real DB — the dummy URL satisfies import-time engine creation.

## Deployment
Push to main → Railway auto-deploys. UI served with no-cache headers to prevent stale JS.
Production URL: https://ai-tracker-tool-production.up.railway.app
Redis is deployed on Railway (auto-connected to Celery).

## Things to Watch
- Railway PostgreSQL has limited storage — `file_content` column should stay NULL
- LLM timeout is 120s on the HTTP client, 90s on async calls
- The `parseCSVLine` function handles quoted fields with commas (e.g. "Consumer, Cyclical")
- Scenario snapshots are saved on every PUT to `/companies/{ticker}/scenarios`
- Feedback is auto-promoted to Prompt Lab on save (no manual promote step)
