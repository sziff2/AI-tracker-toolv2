# CLAUDE.md — Project Context for Claude Code

## What This Is
Investment research platform for Oldfield Partners (buy-side value fund). Tracks ~39 companies across multiple portfolios. Deployed on Railway.

## Key Architecture Decisions
- **Single HTML file UI** (`apps/ui/index.html`) — no framework, no build step. All JS inline.
- **Background jobs**: Celery + Redis for scheduled tasks (weekly harvest, daily prices via Celery Beat). One-off background work uses `asyncio.create_task` in-process.
- **No file_content in DB** — raw PDFs are NOT stored in PostgreSQL. The `Document.file_content` column does not exist in the model. Parsed text lives in `document_sections`.
- **Tables auto-created on startup** via `Base.metadata.create_all` + manual ALTER TABLE / CREATE TABLE migrations in `apps/api/main.py` lifespan.

## Ticker Format
Tickers use Bloomberg format with exchange suffix: `LKQ US`, `BNZL LN`, `3679 JP`, `005930 KS`.
**BP LN** was renamed from `BP/ LN` — slashes break FastAPI path routing. The harvester routes use `{ticker:path}` but other routes use `{ticker}`.

## Document Sourcing Pipeline
Priority: SEC EDGAR → Investegate RNS → IR regex scraper → LLM scraper
- EDGAR: CIK configured per company in `EDGAR_SOURCES` dict (`services/harvester/sources/sec_edgar.py`)
- Investegate: UK RNS announcements, configured in `INVESTEGATE_SOURCES` dict (`services/harvester/sources/investegate.py`)
- IR scraper: regex-based, fast but breaks on SPA pages. Supports multiple URLs per company. 90s per-company timeout.
- LLM scraper: sends page HTML to Claude, handles complex sites, costs ~$0.01-0.05/scan. Skipped in auto-runs.
- ScrapingBee: optional JS rendering for Cloudflare-blocked IR pages (API key in env)
- BP LN and SHEL LN run both EDGAR and Investegate (dual source)
- IR scraper runs for all companies with `ir_docs_url` set, even if EDGAR/Investegate also runs
- robots.txt compliance: `services/harvester/sources/robots_check.py` checks before scraping
- Retry logic: `services/harvester/http_retry.py` with exponential backoff on 429/502/503/504

## Weekly Auto-Harvest
- Celery Beat: Monday + Tuesday 00:00 UTC (1 AM BST). Tuesday is temporary for testing.
- Skips LLM scraper to contain costs (only EDGAR + Investegate + IR regex)
- 90-second per-company timeout prevents slow scrapers blocking the run
- Saves a `HarvestReport` to DB with per-company breakdown
- Posts summary to Microsoft Teams via webhook (`TEAMS_WEBHOOK_URL` env var, Power Automate Workflows format)
- Manual trigger: `POST /harvester/run-weekly` (dispatches to Celery worker)
- Reports: `GET /harvester/reports`, `GET /harvester/reports/latest`

## Daily Price Feed
- Celery Beat: daily at 18:00 UTC (7 PM BST, after US market close)
- Yahoo Finance as primary source, EODHD as fallback
- Bloomberg→Yahoo ticker mapping in `services/price_feed.py`
- After price update, auto-snapshots current price against valuation scenarios for chart tracking
- Manual trigger: `POST /prices/refresh`

## LLM Client — Two Async Paths
**Read this before writing any agent or orchestrator code.**

There are two async paths in `services/llm_client.py`. Use the right one:

| Function | When to use | Returns |
|----------|-------------|---------|
| `call_llm_native_async()` | **Agents and orchestrator** | `{"text": str, "input_tokens": int, "output_tokens": int}` |
| `call_llm_async()` | **Legacy non-agent callers only** | `str` |

`call_llm_native_async` uses `AsyncAnthropic` (true async, no ThreadPoolExecutor), has retry on `RateLimitError`/`APIConnectionError` via tenacity, and returns token counts so `BaseAgent.run()` can track cost per agent. This is what the orchestrator and all agents use.

`call_llm_async` wraps `call_llm()` in a ThreadPoolExecutor. It has no retry and returns a plain string. Keep for existing non-agent callers only — don't use it in new agent code.

Both paths share the global concurrency semaphore (`settings.agent_max_parallel`).

```python
# Correct — for agents and orchestrator:
from services.llm_client import call_llm_native_async
result = await call_llm_native_async(prompt, model=model, max_tokens=4096, feature="agent_xyz")
text = result["text"]
input_tokens = result["input_tokens"]

# Legacy only — for existing non-agent code:
from services.llm_client import call_llm_async
text = await call_llm_async(prompt)
```

## Agent Architecture
The platform is transitioning to a modular agent architecture where each analysis step is an independent agent.

### Status
- `agents/base.py`, `agents/registry.py`, `agents/__init__.py` — **built and reviewed**
- Agent DB tables (`agent_outputs`, `pipeline_runs`, etc.) — **in models.py**
- Individual agents (Financial Analyst, Bear, Bull, Debate, QC) — **not yet built (Phase 1)**

### Directory Structure
- `agents/` — agent package
  - `__init__.py` — exports `BaseAgent`, `AgentResult`, `AgentTier`, `AgentRegistry`
  - `base.py` — `BaseAgent` abstract class, `AgentResult` dataclass, `AgentTier` enum
  - `registry.py` — `AgentRegistry` with auto-discovery and dependency-ordered execution

### Agent Tiers (8 tiers, layered execution)
| Tier | Layer | Model | Purpose |
|------|-------|-------|---------|
| TASK | 0 | Haiku | Pre-processing, triage, chunking |
| DOCUMENT | 1 | Haiku | Per-document extraction (P&L, BS, CF, segments) |
| SPECIALIST | 2 | Sonnet | Company-level analysis (financial analyst, thesis comparison) |
| INDUSTRY | 3 | Sonnet | Cross-company within an industry |
| SECTOR | 4 | Sonnet | Sector-level views |
| MACRO | 5 | Sonnet | Macro regime, rates, credit cycle |
| PORTFOLIO | 6 | Sonnet | Portfolio-level risk, allocation |
| META | 7 | Haiku | QC, calibration, orchestration |

### Writing an Agent — Required Pattern
```python
from agents.base import BaseAgent, AgentTier, AgentResult
from agents.registry import AgentRegistry
from typing import Any

@AgentRegistry.register
class FinancialAnalystAgent(BaseAgent):
    agent_id   = "financial_analyst"
    agent_name = "Financial Analyst"
    tier       = AgentTier.SPECIALIST

    # depends_on MUST be a class-level declaration (not set in __init__).
    # AgentRegistry reads this at class level for topological sort.
    depends_on = []                    # no upstream dependencies for this agent
    feeds_into = ["bear_case", "bull_case", "debate_agent"]

    # Optional: inline prompt string (overrides prompts/agents/financial_analyst.txt)
    # prompt_template = "Analyse {ticker} for period {period_label}..."

    # Optional: output structure for QC agent validation
    output_schema = {
        "revenue_growth": float,
        "margin_direction": str,
        "thesis_direction": str,
        "confidence": float,
    }

    def validate_output(self, raw: str) -> Any:
        import json
        return json.loads(raw)          # or use a Pydantic model
```

Key rules:
- Agents are **stateless** — no DB access inside an agent. Orchestrator handles all DB I/O.
- `depends_on` is a **ClassVar** — set at class level, not in `__init__`. The registry reads it on the class, not an instance.
- `validate_output` **must** parse and validate `raw` — it receives the raw LLM string. Raise if invalid.
- Place agent files under `agents/` in an appropriate subdirectory. The registry auto-discovers them.

### Agent Startup — `autodiscover()` Must Be Called in App Lifespan
`AgentRegistry.autodiscover()` is NOT called on import (that would cause package scan side effects in tests). It must be called once at startup in `apps/api/main.py` after the DB is ready:

```python
from agents.registry import AgentRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... DB init ...
    AgentRegistry.autodiscover()
    for warning in AgentRegistry.validate_dependencies():
        logger.warning("Agent wiring: %s", warning)
    yield
```

### Key Concepts
- **Agents are stateless** — receive inputs dict, call LLM, return `AgentResult`. No DB access inside agents.
- **Orchestrator handles DB** — reads inputs, calls agents, persists results to `agent_outputs` table.
- **Dependency ordering**: Agents declare `depends_on` (ClassVar list); `AgentRegistry.get_execution_order()` does topological sort with cycle detection.
- **Predictions**: Agents can return trackable predictions via `extract_predictions()`, resolved later for calibration.
- **Model routing**: TASK/DOCUMENT/META → Haiku. All others → Sonnet. Override per-agent via `model_override` class attribute or `settings.agent_model_overrides` dict.
- **Prompt cache**: File-based prompts are cached. Call `prompts.loader.clear_prompt_cache()` when a Prompt Lab variant is promoted or a `.txt` file is updated at runtime.

### Clean Break Design (Planned — Phase 1+)
Existing plumbing (parser, extractor, normaliser, LLM client) stays unchanged. Everything above it — analysis, comparison, synthesis, output — is rebuilt as agents. Current services to be replaced:
- `thesis_comparator.py` → absorbed into Financial Analyst agent
- `surprise_detector.py` → absorbed into Financial Analyst agent
- `output_generator.py` → replaced by Research Agenda + PM Agent
- `background_processor.py` → replaced by Orchestrator

### Context Contract (Planned — Phase 1+)
Every agent receives a shared `ContextContract` containing macro assumptions (regime, rates, credit, growth, FX, commodities, inflation, geopolitical risks) that no agent may contradict. Set by macro agents with analyst overrides. Stored in `context_contracts` table.

### Thesis Generation Cascade (Planned — Phase 2+)
Thesis is generated as a structured document with pillars (not free text). Each pillar links to macro dependencies and sector views from the context contract. When macro assumptions change, affected theses are flagged for re-evaluation.

### DB Tables (Agent Infrastructure)
- `agent_outputs` — one row per agent × pipeline run; output_json (JSONB), confidence, qc_score, cost_usd
- `agent_calibration` — per-agent accuracy tracking (unique on agent_id)
- `pipeline_runs` — audit trail: one row per "Run Analysis" click with full per-agent breakdown
- `context_contracts` — macro assumptions (JSONB) shared across agents; one `is_active=True` row at a time
- `sector_theses` — per-sector thesis linked to a context contract
- `thesis_macro_dependencies` — links sector theses to specific macro assumption keys
- `harvest_reports` — weekly harvest summary with per-company JSONB breakdown

### Reference Docs
- `Dev plans/_agent-architecture-clean-break.md` — full agent design, tiers, orchestration
- `Dev plans/_thesis-architecture.md` — context contracts, thesis generation, consistency
- `Dev plans/_0.8_financial-extraction-architecture.md` — pre-segmented extraction pipeline

### Settings (Agent-Related)
- `agent_default_model` — `claude-sonnet-4-6` (quality agents: SPECIALIST, INDUSTRY, SECTOR, MACRO, PORTFOLIO)
- `agent_fast_model` — `claude-haiku-4-5-20251001` (TASK, DOCUMENT, META agents)
- `agent_max_parallel` — global concurrency semaphore limit (default 8)
- `agent_pipeline_budget_usd` — per-pipeline spending cap (default $2); enforced by `BudgetGuard`
- `agent_pipeline_timeout_seconds` — wall-clock timeout per pipeline run (default 300s)
- `agent_model_overrides` — dict of `{agent_id: model_name}` for per-agent model routing at pipeline config time (set via env as JSON: `AGENT_MODEL_OVERRIDES='{"financial_analyst": "claude-opus-4-6"}'`)

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

## Analysis Pipeline (Current)
`Document → parse (PDF/HTML/DOCX) → extract metrics → compare thesis → detect surprises → synthesise`
- Already-parsed documents are skipped on re-run (checks sections_count + metrics_count)
- `resynthesise` endpoint skips parsing/extraction, just re-runs synthesis with updated thesis
- FY is treated as equivalent to Q4 for period comparisons

## Common Patterns
- `enc(ticker)` in JS = `encodeURIComponent`
- `get_company_or_404(db, ticker)` in Python routes
- `_clean_ticker(raw)` = strip + uppercase
- Period format: `2025_Q1`, `2025_Q2`, `2025_Q3`, `2025_Q4` (FY→Q4, H1→Q2, H2→Q4 mapped everywhere)
- `currentQuarter()` / `currentYear()` in JS for earnings-lag-aware default period

## Testing
```bash
DATABASE_URL="postgresql+asyncpg://x:x@localhost/x" pytest tests/ -v
```
146 tests across 5 test files. Tests don't need a real DB — the dummy URL satisfies import-time engine creation.

## Deployment
Push to main → Railway auto-deploys (web, worker, beat services). UI served with no-cache headers to prevent stale JS.
Production URL: https://ai-tracker-tool-production.up.railway.app
Redis deployed on Railway (Celery broker + backend).
Three Railway services: web (uvicorn), worker (celery worker), beat (celery beat).

## Things to Watch
- Railway PostgreSQL has limited storage — `Document.file_content` does NOT exist in the model. Do not add it.
- LLM client has two async paths — **use `call_llm_native_async` in agents/orchestrator**, not `call_llm_async` (see LLM Client section above)
- `AgentRegistry.autodiscover()` must be called in the FastAPI lifespan, not on import
- `prompts.loader.clear_prompt_cache()` must be called when a Prompt Lab variant is promoted
- Agent `depends_on` / `feeds_into` / `trigger_conditions` must be **ClassVar declarations** on the subclass, not set in `__init__`
- Concurrency semaphore limits parallel LLM calls (default 8, `settings.agent_max_parallel`)
- The `parseCSVLine` function handles quoted fields with commas (e.g. "Consumer, Cyclical")
- Scenario snapshots: saved on manual PUT + auto-created daily by price feed
- Feedback is auto-promoted to Prompt Lab on save (no manual promote step)
- Deploys kill running background tasks on the web service — long-running jobs should go via Celery worker
