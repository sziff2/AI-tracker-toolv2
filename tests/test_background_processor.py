"""Tests for services/background_processor — Sprint J / Tier 7.9.

Exercises the lifted module-level `_process_one_doc` helper (skip-existing
path) and the feature-flag dispatch helpers in routes/documents.py without
touching Celery or the real extraction pipeline."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────
# _process_one_doc — skip path when doc already has sections + metrics
# ─────────────────────────────────────────────────────────────────

class _FakeCountRes:
    """Mimics the .scalar() shape returned by select(func.count(...))."""
    def __init__(self, value: int):
        self._value = value

    def scalar(self):
        return self._value


class _FakeMetricsRes:
    """Mimics .scalars().all() for ExtractedMetric rows."""
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDocRes:
    """Mimics .scalar_one_or_none() for a Document row."""
    def __init__(self, doc):
        self._doc = doc

    def scalar_one_or_none(self):
        return self._doc


class _FakeSession:
    """Session that returns a pre-staged sequence of execute() results.

    _process_one_doc calls execute() in this order when the skip path fires:
      1. select(Document).where(id == did)        → doc row
      2. count(DocumentSection)                   → sections count
      3. count(ExtractedMetric)                   → metrics count
      4. select(ExtractedMetric) ... (if skipping)→ existing metric rows
    """
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, *_args, **_kwargs):
        return self._results.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _make_fake_metric(name: str, value: float, unit: str = "%"):
    m = MagicMock()
    m.metric_name = name
    m.metric_value = value
    m.metric_text = f"{value}{unit}"
    m.unit = unit
    m.segment = "consolidated"
    m.geography = None
    m.source_snippet = ""
    m.confidence = 0.9
    return m


def test_process_one_doc_skip_path_loads_existing_metrics():
    """When the doc has sections + metrics, _process_one_doc skips parse +
    extract and returns the existing metric rows as items. No LLM call."""
    from services.background_processor import _process_one_doc

    fake_doc = MagicMock()
    fake_doc.id = uuid.uuid4()
    fake_doc.title = "already_processed.pdf"

    existing_metrics = [
        _make_fake_metric("revenue", 100.0, "USD_M"),
        _make_fake_metric("ebitda_margin", 18.5, "%"),
    ]

    fake_session = _FakeSession([
        _FakeDocRes(fake_doc),
        _FakeCountRes(12),   # 12 sections — already parsed
        _FakeCountRes(42),   # 42 metrics — already extracted
        _FakeMetricsRes(existing_metrics),
    ])

    with patch(
        "services.background_processor.AsyncSessionLocal",
        return_value=fake_session,
    ):
        result = asyncio.run(_process_one_doc(fake_doc.id, "earnings_release", "TEST US"))

    assert result is not None
    assert result["doc_id"] == fake_doc.id
    assert result["dtype"] == "earnings_release"
    assert result["title"] == "already_processed.pdf"
    steps = result["result"]["steps"]
    assert any(s["step"] == "parse" and s["status"] == "skipped" for s in steps)
    assert any(s["step"] == "extract" and s["status"] == "skipped" for s in steps)
    assert len(result["items"]) == 2
    assert result["items"][0]["metric_name"] == "revenue"


def test_process_one_doc_returns_none_when_doc_missing():
    """_process_one_doc handles the case where the document row is gone
    (e.g. deleted between dispatch and worker pickup). Should return None,
    not raise."""
    from services.background_processor import _process_one_doc

    fake_session = _FakeSession([_FakeDocRes(None)])
    with patch(
        "services.background_processor.AsyncSessionLocal",
        return_value=fake_session,
    ):
        result = asyncio.run(
            _process_one_doc(uuid.uuid4(), "earnings_release", "TEST US")
        )
    assert result is None


# ─────────────────────────────────────────────────────────────────
# Dispatch helpers — feature-flag routing (Sprint J)
# ─────────────────────────────────────────────────────────────────

def _stub_worker_tasks_module(single_task=None, batch_task=None):
    """Install a stub apps.worker.tasks module in sys.modules so the
    dispatch helpers' in-route `from apps.worker.tasks import ...` works
    in environments without Celery installed (CI / local dev).
    Returns a context manager that restores the previous module on exit."""
    import sys
    import types

    class _StubCtx:
        def __init__(self):
            self._prev_worker = sys.modules.get("apps.worker.tasks")

        def __enter__(self):
            module = types.ModuleType("apps.worker.tasks")
            if single_task is not None:
                module.parse_and_extract_single_task = single_task
            if batch_task is not None:
                module.parse_and_extract_batch_task = batch_task
            sys.modules["apps.worker.tasks"] = module
            return module

        def __exit__(self, *_exc):
            if self._prev_worker is not None:
                sys.modules["apps.worker.tasks"] = self._prev_worker
            else:
                sys.modules.pop("apps.worker.tasks", None)
            return False

    return _StubCtx()


def test_dispatch_single_prefers_celery_when_flag_on():
    """When the flag is on, _dispatch_single_pipeline hands off to the
    Celery task via .delay() and does NOT start an in-process task."""
    from apps.api.routes.documents import _dispatch_single_pipeline

    fake_task = MagicMock()
    fake_task.delay = MagicMock(return_value=MagicMock(id="celery-job-1"))

    import apps.api.routes.documents as docs_mod
    with patch.object(docs_mod.settings, "use_celery_for_document_processing", True):
        with _stub_worker_tasks_module(single_task=fake_task):
            with patch("services.background_processor.start_background_job") as spawn_mock:
                _dispatch_single_pipeline(
                    job_id=uuid.uuid4(), company_id=uuid.uuid4(),
                    ticker="TEST US", doc_id=uuid.uuid4(), period_label="2026_Q1",
                )
                assert fake_task.delay.call_count == 1
                assert spawn_mock.call_count == 0


def test_dispatch_single_falls_back_in_process_when_flag_off():
    """With the flag off, the helper spawns the in-process coroutine via
    start_background_job — the legacy path."""
    from apps.api.routes.documents import _dispatch_single_pipeline

    import apps.api.routes.documents as docs_mod
    with patch.object(docs_mod.settings, "use_celery_for_document_processing", False):
        with patch("services.background_processor.start_background_job") as spawn_mock:
            with patch("services.background_processor.run_single_pipeline") as run_mock:
                run_mock.return_value = MagicMock()  # coroutine placeholder
                _dispatch_single_pipeline(
                    job_id=uuid.uuid4(), company_id=uuid.uuid4(),
                    ticker="TEST US", doc_id=uuid.uuid4(), period_label="2026_Q1",
                )
                assert spawn_mock.call_count == 1


def test_dispatch_batch_prefers_celery_when_flag_on():
    from apps.api.routes.documents import _dispatch_batch_pipeline

    fake_task = MagicMock()
    fake_task.delay = MagicMock(return_value=MagicMock(id="celery-job-2"))

    import apps.api.routes.documents as docs_mod
    with patch.object(docs_mod.settings, "use_celery_for_document_processing", True):
        with _stub_worker_tasks_module(batch_task=fake_task):
            with patch("services.background_processor.start_background_job") as spawn_mock:
                _dispatch_batch_pipeline(
                    job_id=uuid.uuid4(), company_id=uuid.uuid4(),
                    ticker="TEST US",
                    doc_ids=[uuid.uuid4(), uuid.uuid4()],
                    doc_types=["earnings_release", "transcript"],
                    period_label="2026_Q1",
                    model="standard",
                )
                assert fake_task.delay.call_count == 1
                assert spawn_mock.call_count == 0


def test_dispatch_batch_falls_back_on_celery_exception():
    """If the Celery dispatch raises (worker unreachable, broker down,
    etc.) the helper falls back to the in-process path — no user-visible
    failure for a degraded worker service."""
    from apps.api.routes.documents import _dispatch_batch_pipeline

    failing_task = MagicMock()
    failing_task.delay = MagicMock(side_effect=RuntimeError("broker down"))

    import apps.api.routes.documents as docs_mod
    with patch.object(docs_mod.settings, "use_celery_for_document_processing", True):
        with _stub_worker_tasks_module(batch_task=failing_task):
            with patch("services.background_processor.start_background_job") as spawn_mock:
                with patch("services.background_processor.run_batch_pipeline") as run_mock:
                    run_mock.return_value = MagicMock()
                    _dispatch_batch_pipeline(
                        job_id=uuid.uuid4(), company_id=uuid.uuid4(),
                        ticker="TEST US",
                        doc_ids=[uuid.uuid4()], doc_types=["earnings_release"],
                        period_label="2026_Q1",
                    )
                    assert failing_task.delay.call_count == 1  # tried Celery
                    assert spawn_mock.call_count == 1          # fell back
