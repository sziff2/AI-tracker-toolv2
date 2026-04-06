"""
Database engine, session factory, and base model.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Index, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from configs.settings import settings

# ── Async engine (used by FastAPI) ───────────────────────────────
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# ── Sync engine (used by Celery workers / Alembic) ──────────────
sync_engine = create_engine(
    settings.database_url_sync,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(sync_engine)


class Base(DeclarativeBase):
    """Shared base for all ORM models."""
    pass


class TimestampMixin:
    """Adds created_at to every model."""
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def new_uuid():
    return uuid.uuid4()


# ── Dependency for FastAPI routes ────────────────────────────────
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_company_or_404(db: AsyncSession, ticker: str):
    """Look up a company by ticker or raise HTTPException(404)."""
    from fastapi import HTTPException
    from sqlalchemy import select
    from apps.api.models import Company

    result = await db.execute(select(Company).where(Company.ticker == ticker.upper()))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(404, f"Company {ticker} not found")
    return company
