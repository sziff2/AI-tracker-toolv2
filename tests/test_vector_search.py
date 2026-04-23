"""Tests for services/vector_search — Tier 3.4 Part 1.

Covers feature-flag gating, empty-input short-circuits, model-load
failure fallback, and SearchHit shape. Does NOT exercise Postgres — the
SQL path is behind a conditional that requires pgvector + the extension,
so Part 1 verifies the callable shape + fallbacks only.

Model loading is stubbed so tests run in seconds without downloading
sentence-transformers weights (~130MB)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.vector_search import (
    SearchHit,
    _format_vector,
    embed_text,
    embed_texts,
    search_sections,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _reset_model_cache():
    """The module-level _model singleton is cached across tests. Clear
    it before each test that wants control over the loader."""
    import services.vector_search as vs_mod
    vs_mod._model = None


# ─────────────────────────────────────────────────────────────────
# embed_text — local sentence-transformers wrapper
# ─────────────────────────────────────────────────────────────────

def test_embed_text_returns_none_on_empty_input():
    assert asyncio.run(embed_text("")) is None
    assert asyncio.run(embed_text("   ")) is None


def test_embed_text_returns_none_when_model_load_fails():
    """If sentence-transformers isn't installed, _get_model returns None.
    embed_text must not raise — it returns None so callers fall back to
    keyword search."""
    _reset_model_cache()
    with patch("services.vector_search._get_model", return_value=None):
        result = asyncio.run(embed_text("combined ratio"))
    assert result is None


def test_embed_text_returns_vector_on_success():
    """When the model loads and encodes cleanly, embed_text returns a
    plain list[float]. Mock the model so no weight download happens."""
    _reset_model_cache()

    fake_embedding = [0.1] * 384
    fake_model = MagicMock()
    fake_model.encode = MagicMock(return_value=fake_embedding)

    with patch("services.vector_search._get_model", return_value=fake_model):
        result = asyncio.run(embed_text("combined ratio pricing"))

    assert result == fake_embedding
    fake_model.encode.assert_called_once()
    # bge models require normalize_embeddings=True for cosine
    _args, kwargs = fake_model.encode.call_args
    assert kwargs.get("normalize_embeddings") is True


def test_embed_text_returns_none_on_encode_failure():
    """Any exception inside model.encode is caught and reported as None."""
    _reset_model_cache()
    fake_model = MagicMock()
    fake_model.encode = MagicMock(side_effect=RuntimeError("oom"))

    with patch("services.vector_search._get_model", return_value=fake_model):
        result = asyncio.run(embed_text("test"))

    assert result is None


def test_embed_texts_empty_list_returns_empty_list():
    assert asyncio.run(embed_texts([])) == []


def test_embed_texts_all_empty_returns_all_none():
    """Every input is blank → all-None output, aligned to input length."""
    _reset_model_cache()
    with patch("services.vector_search._get_model", return_value=MagicMock()):
        result = asyncio.run(embed_texts(["", "   ", ""]))
    assert result == [None, None, None]


def test_embed_texts_preserves_index_alignment_with_empties():
    """Blanks become None at the correct position; non-blanks get vectors.
    bge-small returns 384-dim vectors — use a stub that respects that."""
    _reset_model_cache()
    fake_model = MagicMock()
    # sentence-transformers returns a 2D numpy-like; our stub returns a list
    # of lists. cleaned ["a", "", "b"] → encode sees ["a", "", "b"] (blank
    # preserved for alignment); the wrapper sets None on the blank.
    fake_model.encode = MagicMock(return_value=[[0.1] * 384, [0.0] * 384, [0.2] * 384])

    with patch("services.vector_search._get_model", return_value=fake_model):
        result = asyncio.run(embed_texts(["topic A", "", "topic B"]))

    assert len(result) == 3
    assert result[0] is not None and len(result[0]) == 384
    assert result[1] is None           # blank input → None preserved
    assert result[2] is not None and len(result[2]) == 384


def test_embed_texts_returns_all_none_on_encode_failure():
    """A batch encode failure reports None for every input — never raises."""
    _reset_model_cache()
    fake_model = MagicMock()
    fake_model.encode = MagicMock(side_effect=RuntimeError("oom"))

    with patch("services.vector_search._get_model", return_value=fake_model):
        result = asyncio.run(embed_texts(["one", "two", "three"]))

    assert result == [None, None, None]


def test_embed_text_truncates_long_input():
    """Input longer than 8000 chars is truncated before encode so we
    don't hit the tokenizer's 512-token limit from the other side."""
    _reset_model_cache()
    captured = {}
    fake_model = MagicMock()

    def _fake_encode(text, **kwargs):
        captured["text_len"] = len(text)
        return [0.0] * 384
    fake_model.encode = _fake_encode

    with patch("services.vector_search._get_model", return_value=fake_model):
        asyncio.run(embed_text("x" * 20_000))

    assert captured["text_len"] == 8000


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
    """Flag on + non-empty query, but embed_text returns None (model
    load issue or encode error). search_sections returns [] — never
    silently runs with a zero vector."""
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
