"""
Investment Research CoWork Agent — FastAPI application entry point.

Start with:
    uvicorn apps.api.main:app --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import PlainTextResponse

from apps.api.database import async_engine, Base
from apps.api.rate_limit import limiter
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
    analytics_router,
)
from apps.api.routes.feedback import router as feedback_router
from apps.api.routes.pipeline import router as pipeline_router
from configs.settings import settings

logger = logging.getLogger(__name__)

START_TIME = time.time()


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
        # CIK on companies
        await conn.execute(sa_text("ALTER TABLE companies ADD COLUMN IF NOT EXISTS cik TEXT"))
        # Tier 5.1 — analyst-curated peer set for competitive positioning agent
        await conn.execute(sa_text("ALTER TABLE companies ADD COLUMN IF NOT EXISTS peer_tickers JSONB DEFAULT '[]'::jsonb"))
        # Tier 3.4 — enable pgvector + embedding column on document_sections.
        # Wrapped so a DB lacking the extension logs a warning but doesn't
        # halt startup. We target vector(384) (BAAI/bge-small-en-v1.5).
        # A prior deploy briefly shipped vector(1536) (OpenAI default); this
        # detects + drops that before recreating at the right dim. Only
        # safe because no rows have embeddings yet — the column was added
        # empty in the same deploy and nothing wrote to it before we flipped.
        try:
            await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
            target_dim = 384
            # Probe existing column dim via pg_attribute.atttypmod (pgvector
            # stores dim in typmod). Returns None if the column is missing.
            dim_q = await conn.execute(sa_text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'document_sections'::regclass "
                "AND attname = 'embedding' AND NOT attisdropped"
            ))
            existing_dim = dim_q.scalar()
            if existing_dim is not None and existing_dim != target_dim:
                logger.info(
                    "pgvector: embedding dim mismatch (existing=%s, target=%s) — dropping + recreating",
                    existing_dim, target_dim,
                )
                await conn.execute(sa_text("DROP INDEX IF EXISTS ix_document_sections_embedding"))
                await conn.execute(sa_text("ALTER TABLE document_sections DROP COLUMN IF EXISTS embedding"))
            await conn.execute(sa_text(
                f"ALTER TABLE document_sections ADD COLUMN IF NOT EXISTS embedding vector({target_dim})"
            ))
            # HNSW index on cosine distance — best quality/speed trade-off
            # for top-k similarity search on ≥1k rows.
            await conn.execute(sa_text(
                "CREATE INDEX IF NOT EXISTS ix_document_sections_embedding "
                "ON document_sections USING hnsw (embedding vector_cosine_ops)"
            ))
            # One-line startup log of current embedding coverage. Makes
            # Tier 3.4 rollout observable via Railway deploy logs without
            # needing a session cookie for the /admin/embedding-stats
            # route. Query is cheap — COUNT(*) on document_sections is
            # ~instant at our scale.
            try:
                stats_q = await conn.execute(sa_text("""
                    SELECT COUNT(*) AS total,
                           COUNT(embedding) AS with_emb,
                           COUNT(*) FILTER (WHERE embedding IS NULL
                                            AND text_content IS NOT NULL
                                            AND text_content <> '') AS candidates
                    FROM document_sections
                """))
                _s = stats_q.one()
                logger.info(
                    "EMBEDDING_STATS total=%d with_embedding=%d backfill_candidates=%d dim=%d",
                    _s.total, _s.with_emb, _s.candidates, target_dim,
                )
            except Exception as _stats_exc:
                logger.warning("Embedding stats query skipped: %s", str(_stats_exc)[:120])
        except Exception as exc:
            logger.warning("pgvector setup skipped: %s", str(exc)[:200])
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
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS harvest_reports (
                id UUID PRIMARY KEY,
                run_at TIMESTAMPTZ NOT NULL,
                "trigger" TEXT NOT NULL,
                summary_json TEXT,
                details_json TEXT,
                teams_sent BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        # KPI actuals — extracted KPIs from briefings
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS kpi_actuals (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id),
                period TEXT NOT NULL,
                year INTEGER NOT NULL,
                kpi_name TEXT NOT NULL,
                value DOUBLE PRECISION,
                value_bool BOOLEAN,
                value_text TEXT,
                source_doc_id UUID REFERENCES documents(id),
                extracted_at TIMESTAMPTZ,
                UNIQUE(company_id, period, year, kpi_name)
            )
        """))
        # LLM usage log — ensure cost-attribution columns exist
        try:
            for col in ["ticker TEXT", "period_label TEXT", "duration_ms INTEGER"]:
                col_name = col.split()[0]
                await conn.execute(sa_text(f"ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS {col}"))
        except Exception:
            pass  # table may not exist yet on fresh deploy

        # ── Agent architecture tables ─────────────────────────────
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS agent_outputs (
                id UUID PRIMARY KEY,
                agent_id TEXT NOT NULL,
                company_id UUID REFERENCES companies(id),
                period_label TEXT,
                document_id UUID REFERENCES documents(id),
                portfolio_id UUID,
                status TEXT DEFAULT 'completed',
                output_json TEXT,
                confidence FLOAT,
                qc_score FLOAT,
                duration_ms INTEGER,
                prompt_variant_id UUID,
                inputs_used TEXT,
                pipeline_run_id UUID,
                predictions_json TEXT,
                predictions_resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS agent_calibration (
                id UUID PRIMARY KEY,
                agent_id TEXT UNIQUE NOT NULL,
                total_runs INTEGER DEFAULT 0,
                avg_confidence FLOAT,
                avg_qc_score FLOAT,
                prediction_accuracy FLOAT,
                last_calibrated_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS context_contracts (
                id UUID PRIMARY KEY,
                version INTEGER NOT NULL,
                macro_assumptions TEXT,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS sector_theses (
                id UUID PRIMARY KEY,
                sector TEXT NOT NULL,
                contract_id UUID REFERENCES context_contracts(id),
                thesis_json TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id),
                period_label TEXT,
                trigger TEXT,
                status TEXT DEFAULT 'running',
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                duration_ms INTEGER,
                total_cost_usd FLOAT,
                total_input_tokens INTEGER,
                total_output_tokens INTEGER,
                total_llm_calls INTEGER,
                agents_planned INTEGER,
                agents_completed INTEGER,
                agents_failed INTEGER,
                agents_skipped INTEGER,
                overall_qc_score FLOAT,
                error_message TEXT,
                agent_execution_log TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))

        # ── Agent columns on thesis_versions ──────────────────────
        for col in ["pillars", "macro_dependencies", "sector_dependencies", "generated_by"]:
            await conn.execute(sa_text(f"ALTER TABLE thesis_versions ADD COLUMN IF NOT EXISTS {col} TEXT"))
        await conn.execute(sa_text("ALTER TABLE thesis_versions ADD COLUMN IF NOT EXISTS contract_version INTEGER"))

        # ── Agent columns on processing_jobs ──────────────────────
        for col in ["agent_results", "agents_completed", "agents_failed"]:
            await conn.execute(sa_text(f"ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS {col} TEXT"))

        # ── Enriched extraction storage ──────────────────────────
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS extraction_profiles (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id),
                document_id UUID REFERENCES documents(id),
                period_label TEXT,
                extraction_method TEXT,
                sections_found INTEGER,
                section_types TEXT,
                items_extracted INTEGER,
                confidence_profile TEXT,
                segment_data TEXT,
                disappearance_flags TEXT,
                non_gaap_bridges TEXT,
                non_gaap_comparison TEXT,
                mda_narrative TEXT,
                detected_period TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        # Per-metric qualifier columns + period_frequency for FY-vs-Q
        # disambiguation. Default 'Q' on backfill matches the historical
        # quarterly-only assumption; native-extraction's IS/BS/CF pass
        # writes 'FY' for 10-K full-year line items so they no longer
        # collide with same-period quarterly metrics.
        for col in [
            "is_one_off BOOLEAN DEFAULT FALSE",
            "qualifier_json TEXT",
            "period_frequency TEXT DEFAULT 'Q'",
        ]:
            col_name = col.split()[0]
            await conn.execute(sa_text(f"ALTER TABLE extracted_metrics ADD COLUMN IF NOT EXISTS {col}"))
        # Backfill: ensure all existing rows have the new default.
        await conn.execute(sa_text(
            "UPDATE extracted_metrics SET period_frequency = 'Q' WHERE period_frequency IS NULL"
        ))

        # 2026-04-26: re-canonicalise period_label on rows where the old
        # FY→Q4 / H1→Q2 fold was applied. Now that normalise_period
        # preserves the shape natively, those rows should sit under
        # 2025_FY / 2025_H1 etc. rather than collide with quarterly
        # buckets. period_frequency was already correctly tagged so
        # we use it as the source of truth.
        await conn.execute(sa_text("""
            UPDATE extracted_metrics
            SET period_label = REGEXP_REPLACE(period_label, '_Q4$', '_FY')
            WHERE period_frequency = 'FY' AND period_label ~ '_Q4$'
        """))
        await conn.execute(sa_text("""
            UPDATE extracted_metrics
            SET period_label = REGEXP_REPLACE(period_label, '_Q2$', '_H1')
            WHERE period_frequency = 'H1' AND period_label ~ '_Q2$'
        """))
        # Normalise frequency values written as the suffix (Q1/Q2/Q3/Q4)
        # — pick the right Q[1-4] from the period_label.
        await conn.execute(sa_text("""
            UPDATE extracted_metrics
            SET period_frequency = SUBSTRING(period_label FROM '_(Q[1-4])$')
            WHERE period_frequency = 'Q' AND period_label ~ '_Q[1-4]$'
        """))

        # Reconciliation report column on extraction_profiles
        await conn.execute(sa_text(
            "ALTER TABLE extraction_profiles ADD COLUMN IF NOT EXISTS reconciliation JSONB"
        ))

        # ── Ingestion triage ─────────────────────────────────────
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS ingestion_triage (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id),
                source_url TEXT NOT NULL,
                candidate_title TEXT,
                source_type TEXT,
                document_type TEXT,
                period_label TEXT,
                priority TEXT,
                relevance_score INTEGER,
                auto_ingest BOOLEAN DEFAULT TRUE,
                needs_review BOOLEAN DEFAULT FALSE,
                rationale TEXT,
                was_ingested BOOLEAN DEFAULT FALSE,
                analyst_override TEXT,
                was_useful BOOLEAN,
                document_id UUID REFERENCES documents(id),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_ingestion_triage_company ON ingestion_triage(company_id)"
        ))
        await conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_ingestion_triage_source_url ON ingestion_triage(source_url)"
        ))

        # ── Pipeline runs: warnings column (gates output goes here) ──
        await conn.execute(sa_text(
            "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS warnings JSONB"
        ))

        # ── Harvest reports schema drift ─────────────────────────
        # Scheduler writes run_at/summary_json/details_json/teams_sent but
        # the ORM model (HarvestReport) was refactored to use started_at /
        # per_company etc. Railway DB was created from the older schema, so
        # ensure the columns the scheduler expects actually exist.
        for col_def in [
            "run_at TIMESTAMPTZ",
            "summary_json TEXT",
            "details_json TEXT",
            "teams_sent BOOLEAN DEFAULT FALSE",
        ]:
            await conn.execute(sa_text(
                f"ALTER TABLE harvest_reports ADD COLUMN IF NOT EXISTS {col_def}"
            ))

        # ── Coverage rescan log ──────────────────────────────────
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS coverage_rescan_log (
                id UUID PRIMARY KEY,
                company_id UUID REFERENCES companies(id) NOT NULL,
                ticker TEXT,
                doc_type TEXT,
                expected_period TEXT,
                triggered_by TEXT NOT NULL DEFAULT 'auto',
                triggered_at TIMESTAMPTZ NOT NULL,
                sources_tried JSONB,
                candidates_found INTEGER,
                result TEXT,
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_coverage_rescan_company ON coverage_rescan_log(company_id)"
        ))
        await conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_coverage_rescan_triggered_at ON coverage_rescan_log(triggered_at)"
        ))

        # ── Holding factor exposures (Phase 2b: factor shock stress tests) ─
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS holding_factor_exposures (
                company_id UUID REFERENCES companies(id) ON DELETE CASCADE NOT NULL,
                window_months INT NOT NULL,
                betas JSONB NOT NULL,
                alpha DOUBLE PRECISION,
                tstats JSONB,
                r_squared DOUBLE PRECISION,
                n_months INT,
                as_of_month TEXT,
                computed_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (company_id, window_months)
            )
        """))
        await conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_hfe_window ON holding_factor_exposures(window_months)"
        ))

        # ── FX rates (Phase A: historical-price backbone) ───────
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS fx_rates (
                id UUID PRIMARY KEY,
                currency TEXT NOT NULL,
                rate_date DATE NOT NULL,
                rate_to_usd NUMERIC(18, 8) NOT NULL,
                source TEXT DEFAULT 'yahoo',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_fx_ccy_date ON fx_rates(currency, rate_date)"
        ))

    # ── Agent registry autodiscovery ─────────────────────────────
    from agents.registry import AgentRegistry
    AgentRegistry.autodiscover()
    warnings = AgentRegistry.validate_dependencies()
    for w in warnings:
        logger.warning("Agent wiring issue: %s", w)

    yield
    await async_engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# ── Rate limiting ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-tracker-tool-production.up.railway.app",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GZip compression ──────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Security headers ──────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://api.anthropic.com"
    )
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ── Authentication ───────────────────────────────────────────────

import hashlib
import hmac
import secrets

_AUTH_OPEN_PATHS = {"/health", "/robots.txt", "/login", "/auth/login"}


def _sign_session(value: str) -> str:
    """Create a signed session token."""
    sig = hmac.new(settings.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{value}.{sig}"


def _verify_session(token: str) -> bool:
    """Verify a signed session token."""
    if not token or "." not in token:
        return False
    value, sig = token.rsplit(".", 1)
    expected = hmac.new(settings.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    return hmac.compare_digest(sig, expected)


if settings.app_password:
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        # Allow open paths
        if path in _AUTH_OPEN_PATHS:
            return await call_next(request)
        # Check session cookie
        session = request.cookies.get("session")
        if session and _verify_session(session):
            return await call_next(request)
        # API calls get 401, browser gets redirect
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        return RedirectResponse("/login")


LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Login — AI Tracker</title>
<style>
body{background:#0c0d10;color:#e2e4ea;font-family:'DM Sans',sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.box{background:#12141a;border:1px solid #2a2d37;border-radius:12px;padding:32px;width:340px;text-align:center}
h1{font-size:18px;font-weight:600;margin-bottom:4px}
.sub{font-size:12px;color:#5a5d68;margin-bottom:24px}
input{width:100%;padding:10px 14px;background:#0c0d10;border:1px solid #2a2d37;border-radius:8px;color:#e2e4ea;font-family:inherit;font-size:14px;margin-bottom:12px;box-sizing:border-box}
input:focus{outline:none;border-color:#c9a960}
button{width:100%;padding:10px;background:linear-gradient(135deg,#c9a960,#a88932);color:#0f1117;border:none;border-radius:8px;font-family:inherit;font-size:13px;font-weight:600;cursor:pointer}
button:hover{opacity:.9}
.err{color:#ef4444;font-size:12px;margin-top:8px;display:none}
</style></head>
<body><div class="box">
<h1>AI Tracker</h1>
<div class="sub">Oldfield Partners</div>
<form onsubmit="return doLogin(event)">
<input type="password" id="pw" placeholder="Password" autofocus>
<button type="submit">Sign in</button>
</form>
<div class="err" id="err">Incorrect password</div>
</div>
<script>
async function doLogin(e){
  e.preventDefault();
  var pw=document.getElementById('pw').value;
  var r=await fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
  if(r.ok){window.location.href='/';}
  else{document.getElementById('err').style.display='block';document.getElementById('pw').value='';document.getElementById('pw').focus();}
  return false;
}
</script></body></html>"""


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return LOGIN_PAGE


@app.post("/auth/login")
async def do_login(request: Request):
    try:
        body = await request.json()
        password = body.get("password", "")
    except Exception:
        return JSONResponse({"detail": "Invalid request"}, status_code=400)

    if not hmac.compare_digest(password, settings.app_password):
        return JSONResponse({"detail": "Incorrect password"}, status_code=401)

    token = _sign_session(secrets.token_hex(16))
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        "session", token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,  # 30 days
    )
    return response


@app.post("/auth/logout")
async def do_logout():
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("session")
    return response


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
app.include_router(analytics_router, prefix=PREFIX)
app.include_router(feedback_router, prefix=PREFIX)
app.include_router(pipeline_router, prefix=PREFIX)

# Tier 3.3 — briefing PDF download
from apps.api.routes.briefing import router as briefing_router
app.include_router(briefing_router, prefix=PREFIX)


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots():
    return "User-agent: *\nDisallow: /"


@app.get("/health")
async def health():
    return {
        "ok": True,
        "status": "ok",
        "version": settings.app_version,
        "uptime": round(time.time() - START_TIME, 1),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the research analyst UI."""
    html_path = Path(__file__).parent.parent / "ui" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
