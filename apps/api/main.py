"""
Investment Research CoWork Agent — FastAPI application entry point.

Start with:
    uvicorn apps.api.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from apps.api.database import async_engine, Base
from apps.api.routes import (
    companies_router,
    documents_router,
    outputs_router,
    review_router,
    kpi_tracker_router,
    cockpit_router,
    experiments_router,
    esg_router,
    portfolio_router,
    execution_router,
    autorun_router,
    harvester_router,
)
from apps.api.routes.feedback import router as feedback_router
from configs.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup and run migrations."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        sa_text = __import__("sqlalchemy").text
        await conn.execute(sa_text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_content BYTEA"))
        await conn.execute(sa_text("ALTER TABLE research_outputs ADD COLUMN IF NOT EXISTS content_json TEXT"))
        # IC Summary fields on thesis_versions
        for col in ["recommendation", "catalyst", "conviction", "what_would_make_us_wrong", "disconfirming_evidence", "positive_surprises", "negative_surprises"]:
            await conn.execute(sa_text(f"ALTER TABLE thesis_versions ADD COLUMN IF NOT EXISTS {col} TEXT"))
        # Log entries for processing jobs (Option C - detailed log stream)
        await conn.execute(sa_text("ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS log_entries TEXT DEFAULT '[]'"))
        # Model selection for processing jobs (cost control)
        await conn.execute(sa_text("ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'standard'"))
        # Harvester tables
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS harvester_sources (
                id UUID PRIMARY KEY,
                company_id UUID UNIQUE REFERENCES companies(id),
                ir_docs_url TEXT,
                ir_url TEXT,
                rss_url TEXT,
                ir_reachable BOOLEAN DEFAULT FALSE,
                discovery_method TEXT,
                last_checked_at TIMESTAMPTZ,
                override BOOLEAN DEFAULT FALSE,
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS harvested_documents (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id),
                source TEXT,
                source_url TEXT UNIQUE NOT NULL,
                headline TEXT,
                period_label TEXT,
                discovered_at TIMESTAMPTZ,
                ingested BOOLEAN DEFAULT FALSE,
                document_id UUID REFERENCES documents(id),
                error TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
    yield
    await async_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# ── CORS (adjust in production) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ──────────────────────────────────────────
PREFIX = settings.api_prefix

app.include_router(companies_router, prefix=PREFIX)
app.include_router(documents_router, prefix=PREFIX)
app.include_router(outputs_router, prefix=PREFIX)
app.include_router(review_router, prefix=PREFIX)
app.include_router(kpi_tracker_router, prefix=PREFIX)
app.include_router(cockpit_router, prefix=PREFIX)
app.include_router(experiments_router, prefix=PREFIX)
app.include_router(esg_router, prefix=PREFIX)
app.include_router(portfolio_router, prefix=PREFIX)
app.include_router(execution_router, prefix=PREFIX)
app.include_router(autorun_router, prefix=PREFIX)
app.include_router(harvester_router, prefix=PREFIX)
app.include_router(feedback_router, prefix=PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the research analyst UI."""
    html_path = Path(__file__).parent.parent / "ui" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
