"""Tests for services/vector_search — Tier 3.4 Part 1.

Covers feature-flag gating, empty-input short-circuits, embedding client
failure modes, and SearchHit shape. Does NOT exercise Postgres — the SQL
path is behind a conditional that requires pgvector + the extension, so
Part 1 verifies the callable shape + fallbacks only."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.vector_search import (
    SearchHit,
    _format_vector,
    embed_text,
    search_sections,
)


# ─────────────────────────────────────────────────────────────────
# embed_text — OpenAI wrapper
# ─────────────────────────────────────────────────────────────────

def test_embed_text_returns_none_on_empty_input():
    assert asyncio.run(embed_text("")) is None
    assert asyncio.run(embed_text("   ")) is None


def test_embed_text_returns_none_when_no_api_key():
    """Without OPENAI_API_KEY, the function logs a warning and returns
    None. Callers fall back to keyword search — search is never 'broken'
    when the embedding path is unavailable."""
    from configs.settings import settings as real_settings
    with patch.object(real_settings, "openai_api_key", ""):
        result = asyncio.run(embed_text("combined ratio"))
    assert result is None


def _stub_openai_module(client_cls):
    """Install a stub `openai` module in sys.modules exposing AsyncOpenAI
    so embed_text's in-function `from openai import AsyncOpenAI` works in
    environments where the real package isn't installed yet."""
    import sys
    import types

    class _StubCtx:
        def __enter__(self):
            self._prev = sys.modules.get("openai")
            mod = types.ModuleType("openai")
            mod.AsyncOpenAI = client_cls
            sys.modules["openai"] = mod
            return mod

        def __exit__(self, *_exc):
            if self._prev is not None:
                sys.modules["openai"] = self._prev
            else:
                sys.modules.pop("openai", None)
            return False

    return _StubCtx()


def test_embed_text_returns_vector_on_success():
    """When the API responds normally, embed_text returns the raw vector
    list. Mock the OpenAI client so no live key / network is needed."""
    from configs.settings import settings as real_settings

    fake_embedding = [0.1] * 1536
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=fake_embedding)]
    ))
    fake_client_cls = MagicMock(return_value=mock_client)

    with patch.object(real_settings, "openai_api_key", "sk-test"):
        with _stub_openai_module(fake_client_cls):
            result = asyncio.run(embed_text("combined ratio pricing"))

    assert result == fake_embedding
    mock_client.embeddings.create.assert_called_once()


def test_embed_text_returns_none_on_api_failure():
    """Rate limits, network errors, malformed responses — all return None
    (rather than propagate). Logged, not raised."""
    from configs.settings import settings as real_settings

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("rate limit"))
    fake_client_cls = MagicMock(return_value=mock_client)

    with patch.object(real_settings, "openai_api_key", "sk-test"):
        with _stub_openai_module(fake_client_cls):
            result = asyncio.run(embed_text("test"))

    assert result is None


# ─────────────────────────────────────────────────────────────────
# search_sections — feature flag + empty query short-circuits
# ─────────────────────────────────────────────────────────────────

def test_search_sections_returns_empty_when_flag_off():
    """When use_pgvector_search is False, search_sections returns []
    without touching the DB. Callers fall back to keyword search."""
    from configs.settings import settings as real_settings

    fake_db = MagicMock()
    fake_db.execute = AsyncMock()  # should never be called

    with patch.object(real_settings, "use_pgvector_search", False):
        hits = asyncio.run(search_sections(fake_db, "combined ratio"))

    assert hits == []
    fake_db.execute.assert_not_called()


def test_search_sections_returns_empty_when_query_empty():
    """Empty query → [] without DB lookup, regardless of flag."""
    from configs.settings import settings as real_settings

    fake_db = MagicMock()
    fake_db.execute = AsyncMock()

    with patch.object(real_settings, "use_pgvector_search", True):
        assert asyncio.run(search_sections(fake_db, "")) == []
        assert asyncio.run(search_sections(fake_db, "   ")) == []

    fake_db.execute.assert_not_called()


def test_search_sections_returns_empty_when_embed_fails():
    """Flag on + non-empty query, but embed_text returns None (no API
    key or rate limit). search_sections returns [] — never silently runs
    with a zero-vector query."""
    from configs.settings import settings as real_settings

    fake_db = MagicMock()
    fake_db.execute = AsyncMock()

    with patch.object(real_settings, "use_pgvector_search", True):
        with patch("services.vector_search.embed_text",
                   new=AsyncMock(return_value=None)):
            hits = asyncio.run(search_sections(fake_db, "combined ratio"))

    assert hits == []
    fake_db.execute.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# SearchHit dataclass + _format_vector helper
# ─────────────────────────────────────────────────────────────────

def test_search_hit_to_dict_rounds_distance():
    hit = SearchHit(
        section_id="s1", document_id="d1", document_title="10-Q",
        section_title="MD&A", page_number=12,
        snippet="Combined ratio worsened...", distance=0.12345678,
        company_id="c1", period_label="2026_Q1",
    )
    d = hit.to_dict()
    assert d["distance"] == 0.1235   # rounded to 4 places
    assert d["section_title"] == "MD&A"
    assert d["period_label"] == "2026_Q1"


def test_format_vector_matches_pgvector_text_form():
    """pgvector's text input form: [0.1,0.2,...]. Round-trips at 7dp."""
    assert _format_vector([0.1, 0.2]) == "[0.1000000,0.2000000]"
    assert _format_vector([]) == "[]"


def test_search_hit_distance_bounds():
    """Cosine distance on normalised vectors is in [0, 2]. Test just
    the dataclass construction accepts the full range."""
    a = SearchHit("s", "d", "t", None, None, "", 0.0)
    b = SearchHit("s", "d", "t", None, None, "", 2.0)
    assert a.distance == 0.0
    assert b.distance == 2.0
