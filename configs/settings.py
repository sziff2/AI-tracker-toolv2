"""
Application configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.
"""

import logging
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Storage ──────────────────────────────────────────────────
    storage_backend: str = "local"  # "local" | "s3"
    storage_base_path: str = "./storage"
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None

    # ── LLM ──────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Scheduling ───────────────────────────────────────────────
    scan_schedule_cron: str = "0 6 * * *"  # Every day at 06:00

    # ── API ──────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: str = ""  # Comma-separated origins, e.g. "http://localhost:3000,https://app.example.com"

    # ── Upload limits ────────────────────────────────────────────
    max_upload_size_mb: int = 50  # Maximum file upload size in megabytes

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def validate_at_startup(self) -> None:
        """Validate critical settings at boot time. Call from lifespan."""
        warnings = []
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY is not set — LLM calls will fail")
        if not self.cors_origins:
            warnings.append(
                "CORS_ORIGINS is not set — defaulting to allow all origins. "
                "Set CORS_ORIGINS for production use."
            )
        for w in warnings:
            logger.warning("CONFIG WARNING: %s", w)


settings = Settings()
