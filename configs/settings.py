"""
Application configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────
    app_name: str = "Investment Research CoWork Agent"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Authentication ──────────────────────────────────────────
    app_password: str = ""          # Set via APP_PASSWORD env var. Empty = no auth.
    session_secret: str = "change-me-in-production"  # Signs session cookies

    # ── Database ─────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/research_agent"
    database_url_sync: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/research_agent"

    # ── Redis / Celery ───────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # ── Storage ──────────────────────────────────────────────────
    storage_backend: str = "local"  # "local" | "s3"
    storage_base_path: str = "./storage"
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None

    # ── Upload limits ────────────────────────────────────────────
    max_upload_size_mb: int = 50
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB in bytes

    # ── LLM ──────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # FIX: Use current model names that match llm_client._COST_PER_1M
    # and base.py._MODEL_PRICING. Legacy "claude-sonnet-4-20250514" is
    # kept as an alias in llm_client but should not be the default.
    llm_model: str = "claude-sonnet-4-6"
    llm_model_advanced: str = "claude-sonnet-4-6"   # Tier 3: synthesis, thesis comparison
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Agent / Pipeline ─────────────────────────────────────────
    # Default model for quality agents (SPECIALIST, INDUSTRY, SECTOR, MACRO, PORTFOLIO)
    agent_default_model: str = "claude-sonnet-4-6"

    # Fast/cheap model for high-volume agents (TASK, DOCUMENT, META)
    agent_fast_model: str = "claude-haiku-4-5-20251001"

    # Maximum number of concurrent LLM calls across all async agent paths.
    # Drives the global asyncio.Semaphore in llm_client.
    agent_max_parallel: int = 8

    # Hard spending cap per pipeline run (one "Run Analysis" click).
    # BudgetGuard raises BudgetExceeded when this is hit.
    agent_pipeline_budget_usd: float = 2.0

    # Wall-clock timeout for an entire pipeline run in seconds.
    agent_pipeline_timeout_seconds: int = 300

    # Enable agent output caching (24h TTL per agent).
    agent_cache_enabled: bool = False

    # Per-agent model overrides — applied by the orchestrator at pipeline
    # config time, before agents run. Takes precedence over tier defaults
    # but yields to agent.model_override set on the class itself.
    #
    # Set via env var as JSON:
    #   AGENT_MODEL_OVERRIDES='{"financial_analyst": "claude-opus-4-6"}'
    #
    # Keys are agent_id strings; values are model name strings.
    agent_model_overrides: dict = {}

    # Data Completeness + Source Coverage gate mode (per-assessment pre-flight).
    # "warn"  — gates run, reports attached to pipeline_run.warnings, pipeline
    #           still proceeds. Use this for the 2-week validation window.
    # "halt"  — halt_incomplete status terminates the pipeline before any agent
    #           runs. Flip to this once false-positive rate is known.
    # Env: COMPLETENESS_GATE_MODE=halt
    completeness_gate_mode: str = "warn"

    # ── Harvester operational thresholds (env-overridable in Tier 7.8) ──
    # Max download size in bytes; files larger than this are rejected by
    # dispatcher._download(). Default 50 MB — annual reports sometimes
    # exceed 20 MB so don't drop below 30.
    # Env: HARVESTER_MAX_FILE_BYTES=104857600 (100 MB)
    harvester_max_file_bytes: int = 50 * 1024 * 1024

    # Per-company timeout for a single harvest scan (all sources). Protects
    # the weekly cron from a slow IR site blocking the whole run.
    # Env: HARVESTER_COMPANY_TIMEOUT_SECONDS=120
    harvester_company_timeout_seconds: int = 90

    # Max number of IR sub-pages to crawl per company during a scan. The
    # IR scraper finds a main page then follows links to results-archive
    # sub-pages — this caps the crawl width.
    # Env: HARVESTER_MAX_SUBPAGES=20
    harvester_max_subpages: int = 10

    # Coverage Monitor auto-rescan policy.
    # Minimum hours between auto-rescans of the same gap. Protects broken
    # sources from being hammered every day.
    # Env: COVERAGE_MIN_RESCAN_INTERVAL_HOURS=48
    coverage_min_rescan_interval_hours: int = 24

    # Max auto-rescan attempts per gap, lifetime. After this cap the
    # analyst has to trigger manually.
    # Env: COVERAGE_MAX_RESCAN_ATTEMPTS=5
    coverage_max_rescan_attempts: int = 3

    # ── Budget ───────────────────────────────────────────────────
    autorun_budget_usd: float = 10.0
    # Override via env: AUTORUN_BUDGET_USD=20

    # ── Scheduling ───────────────────────────────────────────────
    scan_schedule_cron: str = "0 6 * * *"  # Every day at 06:00

    # ── API ──────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # ── Harvester / Document Agent ────────────────────────────────
    slack_webhook_url: Optional[str] = None
    # Set in Railway: SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

    teams_webhook_url: Optional[str] = None
    # Set in Railway: TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...

    app_base_url: str = "https://ai-tracker-tool-production.up.railway.app"
    # Used to build deep-links in Slack/Teams notifications

    # ── EODHD (optional — earnings calendar, price feeds) ────────
    eodhd_api_key: Optional[str] = None
    # Set in Railway: EODHD_API_KEY=your_key_here

    # ── ScrapingBee (optional — JS rendering for blocked IR pages) ─
    scrapingbee_api_key: Optional[str] = None
    # Set in Railway: SCRAPINGBEE_API_KEY=your_key_here
    # Free tier: 1000 credits/month. JS render = 5 credits per page.

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
