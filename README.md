# Investment Research CoWork Agent

An event-driven research workflow engine that automates document ingestion, KPI extraction, thesis comparison, and analyst-ready output generation for buy-side investment teams.

## Architecture

```
External Sources → Ingestion → Processing → Research Intelligence → Storage → Application
```

The system follows a strict pipeline: **Document → Parsed Text → Structured Data → Comparison → Interpretation → Output**. Every extracted data point carries source provenance, confidence scores, and traceability metadata.

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- An Anthropic API key

### 2. Setup

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

docker compose up --build
```

This starts:
| Service     | URL                    |
|-------------|------------------------|
| API         | http://localhost:8000  |
| Streamlit UI| http://localhost:8501  |
| PostgreSQL  | localhost:5432         |
| Redis       | localhost:6379         |

### 3. Seed Pilot Company

```bash
docker compose exec api python scripts/seed_pilot.py
```

This creates **Heineken (HEIA)** with an initial investment thesis.

### 4. Usage

Open the **Streamlit UI** at `http://localhost:8501` to:

1. **Upload** earnings releases, transcripts, or presentations
2. **Process** documents (parse → extract → compare)
3. **Generate** briefings, IR questions, and thesis drift reports
4. **Review** flagged items in the review queue

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/companies` | List all companies |
| POST | `/api/v1/companies` | Add a company |
| POST | `/api/v1/companies/{ticker}/documents/upload` | Upload a document |
| POST | `/api/v1/documents/{id}/process` | Parse a document |
| POST | `/api/v1/documents/{id}/extract` | Extract KPIs |
| POST | `/api/v1/documents/{id}/compare` | Compare vs thesis |
| POST | `/api/v1/companies/{ticker}/generate-briefing` | Generate briefing |
| POST | `/api/v1/companies/{ticker}/generate-ir-questions` | Generate IR questions |
| GET | `/api/v1/review-queue` | View review queue |

Full API docs at `http://localhost:8000/docs` (Swagger UI).

## Project Structure

```
research-cowork-agent/
├── apps/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   ├── models.py     # SQLAlchemy ORM models
│   │   ├── database.py   # Engine & session setup
│   │   └── routes/       # API endpoint handlers
│   ├── worker/           # Celery background tasks
│   │   └── tasks.py      # Scheduled + event-driven tasks
│   └── ui/               # Streamlit dashboard
│       └── dashboard.py
├── configs/
│   └── settings.py       # Pydantic settings (env vars)
├── prompts/
│   └── __init__.py       # All LLM prompt templates
├── schemas/
│   └── __init__.py       # Pydantic request/response models
├── services/
│   ├── llm_client.py     # Anthropic API wrapper
│   ├── document_ingestion.py
│   ├── document_parser.py
│   ├── metric_extractor.py
│   ├── thesis_comparator.py
│   ├── surprise_detector.py
│   └── output_generator.py
├── storage/              # File storage (raw/processed/outputs)
├── migrations/           # Alembic DB migrations
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Development

### Run without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis locally, then:
uvicorn apps.api.main:app --reload          # API
celery -A apps.worker.tasks worker -l info   # Worker
celery -A apps.worker.tasks beat -l info     # Scheduler
streamlit run apps/ui/dashboard.py           # UI
```

### Run Tests

```bash
pytest tests/ -v
```

## Design Principles

- **Facts before interpretation** — raw extraction precedes any narrative
- **Traceability** — every metric links to source document, snippet, and page
- **Human-in-the-loop** — thesis changes and valuation updates require analyst approval
- **Modular agents** — each service is independent and replaceable

## MVP Scope

Included: document upload, parsing, KPI extraction, thesis comparison, briefing generation, review queue.

Not yet built: portfolio optimisation, complex forecasting, multi-fund analytics.
