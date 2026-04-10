# Investment Research CoWork Agent

An event-driven research workflow engine for buy-side investment teams. Automates document ingestion, KPI extraction, thesis comparison, valuation tracking, and analyst-ready output generation across a multi-company portfolio.

**Live deployment:** Railway (API + PostgreSQL)

## What It Does

1. **Document Sourcing** — Discovers and downloads financial documents from SEC EDGAR (US companies) and company IR websites (global). LLM-powered scraper handles JavaScript-rendered pages.
2. **Extraction** — Parses PDFs, HTML, and DOCX files. Extracts structured metrics, guidance, and management statements using tailored prompts per document type.
3. **Analysis** — Compares results against your investment thesis. Detects surprises, generates probabilistic scenario assessments, and produces opinionated synthesis.
4. **Portfolio Management** — Multi-portfolio support with bear/base/bull valuation scenarios, expected return calculations, and historical scenario tracking.
5. **Prompt Optimisation** — Autonomous overnight prompt improvement using evidence-grounded evals (snippet recall, hallucination detection, Oldfield rubric scoring).

## Architecture

```
Harvester ─→ Parser ─→ Section Splitter ─→ Metric Extractor ─→ Enrichment ─→ Agent Pipeline
   │              │         (FilingSection)      + Qualifiers       │              │
   ├─ SEC EDGAR   │                              + Segments         │    Financial Analyst
   ├─ Investegate │                              + Confidence       │    Bear / Bull Case
   ├─ IR Scraper  │                              + MD&A narrative   │    Debate Agent
   ├─ LLM Scraper │                                                 │    QC / Consistency
   └─ Manual      │                                                 │    PM Agent
                  │                                                 │
   Coverage ◄─────┘    Daily Prices ─→ Scenario Snapshots ─→ Valuation Chart
   Monitor              (Yahoo Finance)
```

## Features

### Cockpit (Per-Company Research Hub)
- **Thesis** — Editable investment thesis with IC Summary fields, auto-generate from LLM
- **Results** — Period-based analysis output with synthesis, probabilistic scenarios, Bayesian signals
- **KPIs** — Tracked metrics with per-period scores and analyst comments
- **Guidance** — Management statement tracking and execution scoring
- **Competitive Analysis** — Moat analysis with LLM assessment
- **Valuation** — Bear/base/bull/Buffett formula inputs with historical tracking chart
- **ESG** — PAI indicator tracking with AI-powered gap analysis
- **Documents** — EDGAR browser, IR page harvester, LLM scan, per-document pipeline status

### Document Sourcing
- **SEC EDGAR** — Auto-lookup CIK by ticker, browse all filings, per-row period/type selection, bulk ingest
- **IR Page Scraper** — Regex-based PDF link detection with sibling page discovery
- **LLM Scraper** — Sends page HTML to Claude for intelligent document identification. Handles SPAs, JSON blobs, unicode-escaped content
- **Harvested Documents Table** — Date filtering, search, duplicate detection, language detection, format badges, pipeline status

### Portfolio
- Multi-portfolio management (OverGlob, OverGac, etc.)
- CSV import with smart header detection (handles quoted fields, Bloomberg tickers)
- Bear/base/bull valuation scenarios with probability weighting
- Scenario history tracking with SVG chart
- Risk/reward scatter map and expected return bar chart
- Clickable company names link to research cockpit

### Extraction Pipeline v2 (Section-Aware)
- **Document Parser** — PDF (PyMuPDF + pdfplumber), HTML (BeautifulSoup for SEC filings), DOCX
- **Section Splitter** — Splits documents into semantic sections (financial statements, MD&A, notes, risk factors, guidance) using pattern-based heading detection with SEC Item numbers, banking/insurance patterns
- **Metric Extraction** — Parallel per-section extraction routed by model tier (financial → Haiku, narrative → Sonnet), with post-processing, normalisation, deduplication, validation
- **Enrichment** — Qualifier analysis (hedge terms, one-off detection), segment decomposition, period validation, disappearance flags, non-GAAP bridge detection, confidence profiling
- **Persistence** — ExtractedMetric rows + ExtractionProfile + ResearchOutput (extraction_context for agents)

### Analysis Pipeline
- **Thesis Comparison** — Prior period context, FY treated as Q4 equivalent
- **Surprise Detection** — Flags material deviations from thesis expectations
- **Synthesis** — Opinionated briefing with management scrutiny, probabilistic scenarios, Bayesian belief updates
- **IR Questions** — Thesis-linked questions for investor relations calls

### Agent Pipeline (Phase 1 — In Progress)
- **Financial Analyst** — Full-spectrum earnings analysis, overall grade (A–F), thesis direction
- **Bear Case** — Steelman negative case with ranked risks and probability × impact scores
- **Bull Case** — Identify underappreciated positives, optionality, and catalysts
- **Debate Agent** — Adjudicates Bear vs Bull, produces probability-weighted verdict
- **QC Agent** — Scores all agent outputs, checks macro consistency against Context Contract

### Prompt Lab
- A/B testing of prompt variants across extraction and output pipelines
- Auto-promotion after 5 wins
- LLM-driven refinement using analyst feedback
- Inline feedback (thumbs up/down) auto-promoted to refinement pipeline

### AutoResearch (Overnight Optimisation)
- **Extraction Pipeline** — Snippet recall (40%), schema compliance (30%), hallucination rate (30%)
- **Output Pipeline** — Specificity, thesis linkage, management scrutiny, actionability, conciseness
- Evidence-grounded evals using real documents and verified metrics
- Cost-guarded with hard spending caps

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (Python) |
| Database | PostgreSQL (Railway) |
| LLM | Anthropic Claude (Sonnet/Opus/Haiku) |
| UI | Single-page HTML/JS (no framework) |
| PDF Parsing | PyMuPDF + pdfplumber |
| HTML Parsing | BeautifulSoup + lxml |
| Background Jobs | Celery + Redis (weekly harvest, daily prices) |
| Task Scheduling | Celery Beat (Monday harvest, daily 18:00 UTC prices) |
| Deployment | Railway |

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Anthropic API key

### Setup

```bash
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/dbname"
export ANTHROPIC_API_KEY="sk-ant-..."

# Start the API
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Tables are auto-created on startup. Open the UI at `http://localhost:8000`.

### Railway Deployment
Push to GitHub — Railway auto-deploys three services from main branch:
- **web** — FastAPI (uvicorn)
- **worker** — Celery worker (processes harvest, price refresh, analysis tasks)
- **beat** — Celery Beat scheduler (triggers weekly harvest Monday 1AM BST, daily prices 7PM BST)

## Project Structure

```
AI-tracker-toolv2/
├── agents/
│   ├── __init__.py              # Exports BaseAgent, AgentResult, AgentTier, AgentRegistry
│   ├── base.py                  # BaseAgent abstract class, AgentResult dataclass, AgentTier enum
│   └── registry.py              # AgentRegistry with auto-discovery and dependency ordering
├── apps/
│   ├── api/
│   │   ├── main.py              # FastAPI app, lifespan, DDL migrations, autodiscover
│   │   ├── models.py            # SQLAlchemy ORM (35+ models incl. agent tables)
│   │   ├── database.py          # Engine, session, helpers
│   │   └── routes/
│   │       ├── cockpit.py       # Per-company research hub
│   │       ├── companies.py     # CRUD, thesis, merge, rename
│   │       ├── documents.py     # Upload, ingest, EDGAR browser
│   │       ├── portfolio.py     # Portfolios, holdings, scenarios, pricing, backload
│   │       ├── harvester.py     # Document discovery, coverage monitor, harvest reports
│   │       ├── experiments.py   # Prompt Lab A/B testing
│   │       ├── feedback.py      # Inline analyst feedback
│   │       ├── esg.py           # ESG/PAI data
│   │       ├── execution.py     # Management execution tracking
│   │       └── outputs.py       # Research output history
│   ├── worker/
│   │   └── tasks.py             # Celery tasks (weekly harvest, daily prices, pipeline)
│   └── ui/
│       └── index.html           # Single-file UI
├── configs/
│   └── settings.py              # Pydantic settings (incl. agent budget, timeouts)
├── prompts/
│   └── __init__.py              # All LLM prompt templates
├── services/
│   ├── llm_client.py            # Anthropic wrapper (sync + native async + parallel)
│   ├── budget_guard.py          # Pipeline cost tracking and enforcement
│   ├── section_splitter.py      # FilingSection dataclass, semantic section detection
│   ├── metric_extractor.py      # v2 section-aware extraction orchestrator
│   ├── qualifier_extractor.py   # Hedge/one-off language detection
│   ├── segment_extractor.py     # Segment decomposition
│   ├── metric_normaliser.py     # Name/unit normalisation, dedup
│   ├── metric_validator.py      # Plausibility checks, cross-validation
│   ├── context_builder.py       # Compressed context for agents (thesis, KPIs, enrichment)
│   ├── background_processor.py  # Batch pipeline + extraction profile persistence
│   ├── price_feed.py            # Yahoo Finance + EODHD daily prices
│   └── harvester/
│       ├── __init__.py          # Harvest orchestrator (39 companies, 90s timeout)
│       ├── dispatcher.py        # Candidate dedup + DB save
│       ├── scheduler.py         # Weekly report generation + Teams notification
│       ├── coverage.py          # Coverage monitor (expected period vs actual)
│       └── sources/
│           ├── sec_edgar.py     # SEC EDGAR API (15 companies)
│           ├── investegate.py   # UK RNS announcements (8 companies)
│           ├── ir_scraper.py    # Regex-based IR page scraper
│           ├── ir_llm_scraper.py # LLM-powered document finder
│           └── robots_check.py  # robots.txt compliance
├── tests/
│   ├── test_services.py         # 64 unit tests (incl. coverage monitor)
│   ├── test_agent_base.py       # Agent base class tests
│   ├── test_segmenter.py        # Financial statement segmenter tests
│   ├── test_llm_client.py       # LLM client tests
│   ├── test_extraction_evals.py # Extraction evaluation tests
│   └── test_api.py              # API integration tests (needs DB)
├── Dockerfile                   # Python 3.12, PYTHONPATH=/app
├── requirements.txt
└── README.md
```

## Data Model (Key Tables)

| Table | Purpose |
|-------|---------|
| `companies` | Ticker, name, sector, country, CIK |
| `documents` | Uploaded/ingested files with period and type |
| `document_sections` | Parsed text per page |
| `extracted_metrics` | Structured KPIs with source provenance |
| `thesis_versions` | Investment thesis with IC summary fields |
| `research_outputs` | Analysis results (synthesis, comparisons) |
| `portfolios` / `portfolio_holdings` | Multi-portfolio with weights |
| `valuation_scenarios` | Bear/base/bull targets and probabilities |
| `scenario_snapshots` | Historical scenario tracking |
| `price_records` | Price history |
| `extraction_profiles` | Per-document enrichment (confidence, segments, MD&A) |
| `harvester_sources` | IR page URLs per company |
| `harvested_documents` | Discovered documents awaiting ingestion |
| `harvest_reports` | Weekly harvest summary with per-company breakdown |
| `agent_outputs` | Per-agent output per pipeline run |
| `pipeline_runs` | Audit trail per analysis run |
| `context_contracts` | Shared macro assumptions for agent consistency |
| `prompt_variants` / `ab_experiments` | Prompt optimisation |
| `extraction_feedback` | Inline analyst annotations |

## Workflow

### Document Ingestion
1. **Documents tab** — Browse EDGAR or scan IR pages to discover documents
2. Review documents with per-row period and type controls
3. Click **Ingest Selected** to download and save to database

### Analysis
1. **Results tab** — Periods auto-appear when documents are ingested
2. Add manual files (transcripts, broker notes) if needed
3. Click **Run Analysis** to process: parse → extract → compare → synthesise
4. Review output with inline feedback (thumbs up/down)

### Valuation
1. **Valuation tab** — Enter bear/base/bull targets and probabilities
2. Buffett formula auto-calculates
3. Scenario history tracks changes over time
4. Portfolio page shows weighted expected returns

## Tests

```bash
DATABASE_URL="postgresql+asyncpg://x:x@localhost/x" pytest tests/test_services.py -v
```

200 tests covering: period logic, metric normalisation, deduplication, plausibility checks, confidence filtering, smart chunking, segment validation, settings, LLM JSON parsing, coverage monitor, agent base class. 13 API tests require a running PostgreSQL instance.

## Design Principles

- **Facts before interpretation** — raw extraction precedes any narrative
- **Traceability** — every metric links to source document, snippet, and page
- **Human-in-the-loop** — document selection, thesis changes, and valuation updates require analyst input
- **Scepticism** — management claims assessed against actual numbers
- **Cost awareness** — model toggle (Haiku/Sonnet/Opus), free regex scraper alongside LLM scan
