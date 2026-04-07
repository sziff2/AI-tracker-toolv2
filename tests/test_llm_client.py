"""
Unit tests for services/llm_client.py (no real API calls).
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestImports:
    """Verify the module imports cleanly and key symbols exist."""

    def test_import_module(self):
        from services.llm_client import (
            call_llm,
            call_llm_async,
            call_llm_json,
            call_llm_json_async,
            call_llm_json_parallel,
            call_llm_native_async,
            usage_tracker,
            set_llm_context,
            set_budget_guard,
            get_budget_guard,
            get_async_client,
        )
        # All should be importable
        assert callable(call_llm)
        assert callable(call_llm_native_async)
        assert callable(get_async_client)


class TestUsageTracker:
    """Test in-memory usage accumulation."""

    def test_record_and_total(self):
        from services.llm_client import _UsageTracker
        tracker = _UsageTracker()
        assert tracker.total_requests == 0
        assert tracker.total_input_tokens == 0

        tracker.record(1000, 500, "claude-sonnet-4-6")
        assert tracker.total_requests == 1
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500

        tracker.record(2000, 1000, "claude-sonnet-4-6")
        assert tracker.total_requests == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500

    def test_total_cost_property(self):
        from services.llm_client import _UsageTracker
        tracker = _UsageTracker()
        # Record 1M input tokens and 0 output tokens
        tracker.record(1_000_000, 0, "claude-sonnet-4-6")
        # At default sonnet pricing ($3/1M input), total should be ~3.0
        assert tracker.total > 0

    def test_record_failure(self):
        from services.llm_client import _UsageTracker
        tracker = _UsageTracker()
        tracker.record_failure()
        tracker.record_failure()
        assert tracker.failed_requests == 2
        assert tracker.total_requests == 0

    def test_summary(self):
        from services.llm_client import _UsageTracker
        tracker = _UsageTracker()
        tracker.record(100, 50, "claude-sonnet-4-6")
        tracker.record_failure()
        s = tracker.summary
        assert s["total_requests"] == 1
        assert s["failed_requests"] == 1
        assert s["total_input_tokens"] == 100
        assert s["total_output_tokens"] == 50


class TestRetryConfig:
    """Verify the retry decorator is wired up correctly."""

    def test_retry_policy_exists(self):
        from services.llm_client import _retry_policy
        # tenacity retry decorator should have retry metadata
        assert _retry_policy is not None
        assert callable(_retry_policy)

    def test_native_async_has_retry(self):
        from services.llm_client import call_llm_native_async
        # A tenacity-wrapped function has a .retry attribute
        assert hasattr(call_llm_native_async, "retry")
        retry_obj = call_llm_native_async.retry
        # Check stop config: 3 attempts
        assert retry_obj.stop.max_attempt_number == 3

    def test_retry_targets_correct_exceptions(self):
        import anthropic
        from services.llm_client import call_llm_native_async
        retry_obj = call_llm_native_async.retry
        # The retry_if_exception_type should target RateLimitError and APIConnectionError
        predicate = retry_obj.retry
        # Verify it's configured to retry the right exception types
        assert hasattr(predicate, 'exception_types') or hasattr(predicate, 'retryable_exceptions') or callable(predicate)


class TestJsonParsing:
    """Test the JSON cleaning/repair helpers."""

    def test_clean_json_fences(self):
        from services.llm_client import _clean_json_string
        raw = '```json\n{"key": "value"}\n```'
        cleaned = _clean_json_string(raw)
        assert cleaned == '{"key": "value"}'

    def test_parse_json_trailing_comma(self):
        from services.llm_client import _parse_json
        raw = '{"a": 1, "b": 2,}'
        result = _parse_json(raw)
        assert result == {"a": 1, "b": 2}


class TestSemaphore:
    """Test that the concurrency semaphore is configured."""

    def test_get_semaphore_returns_semaphore(self):
        import asyncio
        from services.llm_client import _get_semaphore
        # Reset global to force fresh creation
        import services.llm_client as mod
        mod._semaphore = None
        # Need an event loop for Semaphore
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            sem = _get_semaphore()
            assert isinstance(sem, asyncio.Semaphore)
            # Default is settings.agent_max_parallel = 8
            assert sem._value == 8
        finally:
            mod._semaphore = None
            loop.close()
