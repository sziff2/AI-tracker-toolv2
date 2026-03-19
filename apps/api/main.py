"""
Investment Research CoWork Agent — FastAPI application entry point.
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
    search_router,
    feedback_router,
)
from configs.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        sa_text = __import__("sqlalchemy").text
        await conn.execute(sa_text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_content BYTEA"))
        await conn.execute(sa_text("ALTER TABLE research_outputs ADD COLUMN IF NOT EXISTS content_json TEXT"))
        for col in ["recommendation", "catalyst", "conviction", "what_would_make_us_wrong",
                    "disconfirming_evidence", "positive_surprises", "negative_surprises"]:
            await conn.execute(sa_text(f"ALTER TABLE thesis_versions ADD COLUMN IF NOT EXISTS {col} TEXT"))
    yield
    await async_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(search_router, prefix=PREFIX)
app.include_router(feedback_router, prefix=PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/", response_class=HTMLResponse)
async def ui():
    html_path = Path(__file__).parent.parent / "ui" / "index.html"
    return HTMLResponse(html_path.read_text())
