"""
LLM client with both sync and async support for parallel calls.
Includes token usage tracking and improved error handling.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic

from configs.settings import settings
from services.budget_guard import BudgetGuard, BudgetExceeded  # noqa: F401

# ── Cost per 1M tokens (approximate, update as pricing changes) ──
_COST_PER_1M = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
}

# Thread-local context for usage attribution (safe across ThreadPoolExecutor)
_call_context = threading.local()


def set_llm_context(feature: str = None, ticker: str = None, period: str = None):
    """Set context for the next LLM call(s) so usage logs know what triggered them.
    Thread-safe: each thread-pool worker gets its own copy."""
    if feature is not None:
        _call_context.feature = feature
    if ticker is not None:
        _call_context.ticker = ticker
    if period is not None:
        _call_context.period = period


def _get_ctx(key: str, default=None):
    return getattr(_call_context, key, default)


# Active budget guard — set by autorun, checked on every call
_active_budget_guard: BudgetGuard | None = None


def set_budget_guard(guard: BudgetGuard | None):
    """Attach (or clear) a BudgetGuard checked on every LLM call."""
    global _active_budget_guard
    _active_budget_guard = guard


def get_budget_guard() -> BudgetGuard | None:
    return _active_budget_guard


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
    return round((input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000, 6)


def _log_usage(model: str, input_tokens: int, output_tokens: int, duration_ms: int = 0,
               *, feature: str = None, ticker: str = None, period: str = None):
    """Persist LLM usage to database (fire-and-forget).
    Explicit kwargs override thread-local context."""
    try:
        from apps.api.database import SyncSessionLocal
        from apps.api.models import LLMUsageLog
        cost = _estimate_cost(model, input_tokens, output_tokens)
        with SyncSessionLocal() as db:
            db.add(LLMUsageLog(
                id=uuid.uuid4(),
                timestamp=datetime.now(timezone.utc),
                model=model,
                feature=feature or _get_ctx("feature", "unknown"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                ticker=ticker or _get_ctx("ticker"),
                period_label=period or _get_ctx("period"),
                duration_ms=duration_ms,
            ))
            db.commit()
    except Exception as e:
        logger.debug("Failed to log LLM usage: %s", str(e)[:100])

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None
_executor = ThreadPoolExecutor(max_workers=int(settings.llm_max_tokens / 500) or 6)


# ── Token usage tracking ─────────────────────────────────────────

@dataclass
class _UsageTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def record_failure(self):
        self.failed_requests += 1

    @property
    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


usage_tracker = _UsageTracker()


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=120.0,  # 120 second timeout for large synthesis/extraction calls
        )
    return _client


def call_llm(prompt: str, *, max_tokens: int | None = None, temperature: float | None = None,
             model: str | None = None, feature: str | None = None,
             ticker: str | None = None, period: str | None = None) -> str:
    client = get_client()
    model = model or settings.llm_model

    # Set thread-local context so nested/parallel calls inherit it
    if feature:
        _call_context.feature = feature
    if ticker:
        _call_context.ticker = ticker
    if period:
        _call_context.period = period

    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.BadRequestError as e:
        logger.error("Anthropic 400 error with model %s: %s — prompt length: %d chars", model, e.message, len(prompt))
        # Try fallback model if primary fails
        if "model" in str(e.message).lower():
            logger.info("Trying fallback model claude-3-5-sonnet-20241022")
            resp = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e.message)
        raise

    duration_ms = int((time.time() - t0) * 1000)
    text = resp.content[0].text.strip()
    usage_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens)
    _log_usage(model, resp.usage.input_tokens, resp.usage.output_tokens, duration_ms,
               feature=feature, ticker=ticker, period=period)
    # Budget guard — track spend and raise BudgetExceeded if over cap
    if _active_budget_guard is not None:
        _active_budget_guard.track(resp.usage.input_tokens, resp.usage.output_tokens, model)
    logger.debug(
        "LLM response: %d chars, stop=%s, tokens_in=%d, tokens_out=%d, feature=%s, ticker=%s",
        len(text), resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens,
        feature or _get_ctx("feature", "?"), ticker or _get_ctx("ticker", "?"),
    )
    return text


def _repair_truncated_json(raw: str) -> Any:
    last_brace = raw.rfind("}")
    if last_brace == -1:
        raise json.JSONDecodeError("No complete JSON object found", raw, 0)
    candidate = raw[:last_brace + 1]
    if not candidate.rstrip().endswith("]"):
        candidate = candidate.rstrip().rstrip(",") + "\n]"
    return json.loads(candidate)


def _clean_json_string(raw: str) -> str:
    """Clean up common LLM JSON formatting issues."""
    import re
    cleaned = raw

    # Remove markdown code fences
    if cleaned.startswith("```"):
        # Handle ```json or just ```
        first_line_end = cleaned.find("\n")
        if first_line_end > 0:
            cleaned = cleaned[first_line_end + 1:]
        else:
            cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # Remove trailing commas before ] or } (common LLM mistake)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    # Fix missing quotes around field names (rare but happens)
    # e.g., {field: "value"} -> {"field": "value"}
    cleaned = re.sub(r'{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'{"\1":', cleaned)
    cleaned = re.sub(r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r',"\1":', cleaned)

    return cleaned


def _parse_json(raw: str) -> Any:

    # First attempt: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Second attempt: clean common issues
    cleaned = _clean_json_string(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Third attempt: repair truncated JSON
    try:
        result = _repair_truncated_json(cleaned)
        logger.warning("Repaired truncated JSON (%d chars)", len(cleaned))
        return result
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s — raw: %s", exc, raw[:1000])
        raise


def call_llm_json(prompt: str, **kwargs) -> Any:
    raw = call_llm(prompt, **kwargs)
    return _parse_json(raw)


def _snapshot_context() -> dict:
    """Capture current thread-local context for propagation to executor threads."""
    return {
        "feature": _get_ctx("feature"),
        "ticker": _get_ctx("ticker"),
        "period": _get_ctx("period"),
    }


def _restore_context(snapshot: dict):
    """Apply a context snapshot to the current thread-local."""
    for k, v in snapshot.items():
        if v is not None:
            setattr(_call_context, k, v)


async def call_llm_async(prompt: str, timeout_seconds: int = 90, **kwargs) -> str:
    """Run LLM call in thread pool with timeout for Railway compatibility."""
    loop = asyncio.get_event_loop()
    ctx = _snapshot_context()

    def _run():
        _restore_context(ctx)
        return call_llm(prompt, **kwargs)

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, _run),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.warning("LLM call timed out after %ds", timeout_seconds)
        raise TimeoutError(f"LLM request timed out after {timeout_seconds}s")


async def call_llm_json_async(prompt: str, **kwargs) -> Any:
    """Run LLM call in thread pool so multiple can run in parallel."""
    loop = asyncio.get_event_loop()
    ctx = _snapshot_context()

    def _run():
        _restore_context(ctx)
        return call_llm_json(prompt, **kwargs)

    return await loop.run_in_executor(_executor, _run)


async def call_llm_json_parallel(prompts: list[str], max_concurrency: int = 3, timeout_seconds: int = 120, **kwargs) -> list[Any]:
    """Run multiple LLM calls with limited concurrency and timeout per call."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_call(prompt: str, index: int) -> tuple[int, Any]:
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    call_llm_json_async(prompt, **kwargs),
                    timeout=timeout_seconds
                )
                return (index, result)
            except asyncio.TimeoutError:
                logger.warning("LLM call %d timed out after %ds", index + 1, timeout_seconds)
                return (index, TimeoutError(f"LLM call timed out after {timeout_seconds}s"))
            except Exception as e:
                return (index, e)

    tasks = [limited_call(p, i) for i, p in enumerate(prompts)]
    results_tuples = await asyncio.gather(*tasks)

    # Sort by index and extract results
    results_tuples.sort(key=lambda x: x[0])
    out = []
    failed_indices = []
    for i, r in results_tuples:
        if isinstance(r, Exception):
            logger.warning("Parallel LLM call %d/%d failed: %s", i + 1, len(prompts), r)
            usage_tracker.record_failure()
            failed_indices.append(i)
            out.append([])
        else:
            out.append(r)
    if failed_indices:
        logger.warning("LLM parallel batch: %d/%d calls failed (indices: %s)", len(failed_indices), len(prompts), failed_indices)
    return out
