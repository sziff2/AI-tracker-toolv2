"""
Test suite for core API endpoints.

Run with:
    pytest tests/ -v
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from apps.api.database import Base, async_engine
from apps.api.main import app


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables before each test, drop after."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_version(client):
    resp = await client.get("/health")
    assert "version" in resp.json()


# ─────────────────────────────────────────────────────────────────
# Companies CRUD
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_create_and_get_company(client):
    payload = {"ticker": "HEIA", "name": "Heineken N.V.", "sector": "Consumer Staples"}
    resp = await client.post("/api/v1/companies", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["ticker"] == "HEIA"
    assert data["name"] == "Heineken N.V."

    # GET by ticker
    resp2 = await client.get("/api/v1/companies/HEIA")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == data["id"]


@pytest.mark.asyncio
async def test_list_companies(client):
    await client.post("/api/v1/companies", json={"ticker": "HEIA", "name": "Heineken"})
    await client.post("/api/v1/companies", json={"ticker": "ABI", "name": "AB InBev"})
    resp = await client.get("/api/v1/companies")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_update_company(client):
    await client.post("/api/v1/companies", json={"ticker": "HEIA", "name": "Heineken"})
    resp = await client.patch("/api/v1/companies/HEIA", json={"primary_analyst": "Alice"})
    assert resp.status_code == 200
    assert resp.json()["primary_analyst"] == "Alice"


@pytest.mark.asyncio
async def test_company_not_found(client):
    resp = await client.get("/api/v1/companies/XXXX")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ticker_case_insensitive(client):
    await client.post("/api/v1/companies", json={"ticker": "HEIA", "name": "Heineken"})
    resp = await client.get("/api/v1/companies/heia")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "HEIA"


# ─────────────────────────────────────────────────────────────────
# Documents — error cases
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_upload_document_company_not_found(client):
    resp = await client.post(
        "/api/v1/companies/XXXX/documents/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        data={"document_type": "earnings_release", "period_label": "2025_Q4"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_document_not_found(client):
    import uuid
    resp = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_job_not_found(client):
    import uuid
    resp = await client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────
# Review Queue
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_review_queue_empty(client):
    resp = await client.get("/api/v1/review-queue")
    assert resp.status_code == 200
    assert resp.json() == []


# ─────────────────────────────────────────────────────────────────
# Correlation ID Middleware
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_correlation_id_in_response(client):
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers

@pytest.mark.asyncio
async def test_correlation_id_passed_through(client):
    resp = await client.get("/health", headers={"X-Request-ID": "test-123"})
    assert resp.headers.get("x-request-id") == "test-123"
