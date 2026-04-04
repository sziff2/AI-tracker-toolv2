"""
Application configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────
    app_name: str = "Investment Research CoWork Agent"
    app_version: str = "1.0.0"
    debug: bool = False

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
    max_upload_size_mb: int = 50  # Maximum file size in MB
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB in bytes

    # ── LLM ──────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Budget ────────────────────────────────────────────────────
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
    # Used to build deep-links in Slack notifications

    # ── EODHD (optional — earnings calendar, price feeds) ─────────
    eodhd_api_key: Optional[str] = None
    # Set in Railway: EODHD_API_KEY=your_key_here

    # ── ScrapingBee (optional — bypasses Cloudflare for blocked IR pages) ──
    scrapingbee_api_key: Optional[str] = None
    # Set in Railway: SCRAPINGBEE_API_KEY=your_key_here
    # Free tier: 1000 credits/month. JS render = 5 credits per page.

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
