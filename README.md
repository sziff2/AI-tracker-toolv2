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
Document Sources → Ingestion → Parsing → Extraction → Comparison → Synthesis → Output
     │                                                                           │
     ├─ SEC EDGAR (US/Canadian)                                    Briefings ────┤
     ├─ IR Page Scraper (regex)                                    IR Questions ─┤
     ├─ LLM Scraper (complex sites)                                Thesis Drift ─┤
     └─ Manual Upload                                              Scenarios ────┘
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

### Analysis Pipeline
- **Document Parser** — PDF (PyMuPDF + pdfplumber), HTML (BeautifulSoup for SEC filings), DOCX
- **Metric Extraction** — Combined extractor with post-processing, normalisation, deduplication, validation
- **Thesis Comparison** — Prior period context, FY treated as Q4 equivalent
- **Surprise Detection** — Flags material deviations from thesis expectations
- **Synthesis** — Opinionated briefing with management scrutiny, probabilistic scenarios, Bayesian belief updates
- **IR Questions** — Thesis-linked questions for investor relations calls

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
| Background Jobs | asyncio tasks (Celery-ready) |
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
Push to GitHub — Railway auto-deploys from main branch.

## Project Structure

```
AI-tracker-toolv2/
├── apps/
│   ├── api/
│   │   ├── main.py              # FastAPI app, lifespan, migrations
│   │   ├── models.py            # SQLAlchemy ORM (30+ models)
│   │   ├── database.py          # Engine, session, helpers
│   │   └── routes/
│   │       ├── cockpit.py       # Per-company research hub
│   │       ├── companies.py     # CRUD, thesis, merge, rename
│   │       ├── documents.py     # Upload, ingest, EDGAR browser, reprocess
│   │       ├── portfolio.py     # Portfolios, holdings, scenarios, pricing
│   │       ├── harvester.py     # Document discovery sources
│   │       ├── experiments.py   # Prompt Lab A/B testing
│   │       ├── feedback.py      # Inline analyst feedback
│   │       ├── esg.py           # ESG/PAI data
│   │       ├── execution.py     # Management execution tracking
│   │       └── outputs.py       # Research output history
│   ├── worker/
│   │   └── tasks.py             # Celery tasks (harvest, process)
│   └── ui/
│       └── index.html           # Single-file UI
├── configs/
│   └── settings.py              # Pydantic settings
├── prompts/
│   └── __init__.py              # All LLM prompt templates
├── schemas/
│   └── __init__.py              # Request/response models
├── services/
│   ├── llm_client.py            # Anthropic wrapper (sync + async + parallel)
│   ├── document_ingestion.py    # File storage + DB record
│   ├── document_parser.py       # PDF, HTML, DOCX text extraction
│   ├── metric_extractor.py      # Combined + per-type extraction
│   ├── metric_normaliser.py     # Name/unit normalisation, dedup
│   ├── metric_validator.py      # Plausibility checks, cross-validation
│   ├── thesis_comparator.py     # Prior period comparison
│   ├── surprise_detector.py     # Material deviation detection
│   ├── output_generator.py      # Briefings, IR questions
│   ├── background_processor.py  # Batch pipeline + resynthesis
│   ├── context_builder.py       # Thesis + KPI + prior period context
│   ├── prompt_registry.py       # DB-backed prompt variant lookup
│   └── harvester/
│       ├── __init__.py          # Harvest orchestrator
│       ├── dispatcher.py        # Candidate dedup + DB save
│       ├── discovery.py         # IR page auto-discovery
│       └── sources/
│           ├── sec_edgar.py     # SEC EDGAR API
│           ├── ir_scraper.py    # Regex-based IR page scraper
│           ├── ir_llm_scraper.py # LLM-powered document finder
│           └── ir_rss.py        # RSS feed harvester
├── scripts/
│   └── autorun.py               # Overnight prompt optimisation
├── tests/
│   ├── test_services.py         # 52 unit tests
│   └── test_api.py              # API integration tests
├── storage/                     # Raw + processed files
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
| `harvester_sources` | IR page URLs per company |
| `harvested_documents` | Discovered documents awaiting ingestion |
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

52 tests covering: period logic, metric normalisation, deduplication, plausibility checks, confidence filtering, smart chunking, segment validation, settings, LLM JSON parsing.

## Design Principles

- **Facts before interpretation** — raw extraction precedes any narrative
- **Traceability** — every metric links to source document, snippet, and page
- **Human-in-the-loop** — document selection, thesis changes, and valuation updates require analyst input
- **Scepticism** — management claims assessed against actual numbers
- **Cost awareness** — model toggle (Haiku/Sonnet/Opus), free regex scraper alongside LLM scan
