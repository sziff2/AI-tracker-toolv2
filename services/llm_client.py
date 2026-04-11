"""
LLM client with both sync and async support for parallel calls.
Includes token usage tracking, retry logic, and improved error handling.

Calling conventions:
  call_llm()                 — sync, returns str
  call_llm_json()            — sync, returns parsed JSON (Any)
  call_llm_async()           — legacy: runs call_llm() in ThreadPoolExecutor, returns str
                               Use for existing non-agent callers only.
  call_llm_native_async()    — TRUE async via AsyncAnthropic + retry.
                               Returns {"text": str, "input_tokens": int, "output_tokens": int}
                               This is what BaseAgent.run() calls.
  call_llm_json_async()      — async JSON variant (uses ThreadPoolExecutor path, returns Any)
  call_llm_json_parallel()   — parallel JSON calls with single concurrency semaphore
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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from configs.settings import settings
from services.budget_guard import BudgetGuard, BudgetExceeded  # noqa: F401

logger = logging.getLogger(__name__)

# ── Model tier constants ────────────────────────────────────────
TIER_FAST = "fast"          # Haiku — tables, classification, mechanical extraction
TIER_DEFAULT = "default"    # Sonnet — section extraction, KPIs, guidance
TIER_ADVANCED = "advanced"  # Sonnet/Opus — synthesis, thesis comparison, judgement


def _model_for_tier(tier: str) -> str:
    """Resolve a tier name to a concrete model string."""
    if tier == TIER_FAST:
        return settings.agent_fast_model
    elif tier == TIER_ADVANCED:
        return getattr(settings, 'llm_model_advanced', settings.llm_model)
    else:
        return settings.llm_model

# ── Cost per 1M tokens ───────────────────────────────────────────
# Keep model name aliases in sync with base.py's _MODEL_PRICING and
# configs/settings.py agent_default_model / agent_fast_model values.
_COST_PER_1M: dict[str, dict[str, float]] = {
    # Current generation
    "claude-sonnet-4-6":              {"input": 3.0,  "output": 15.0},
    "claude-opus-4-6":                {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001":      {"input": 0.80, "output": 4.0},
    # Legacy aliases — kept for backward compat with any hardcoded strings
    "claude-sonnet-4-20250514":       {"input": 3.0,  "output": 15.0},
    "claude-3-5-sonnet-20241022":     {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":         {"input": 15.0, "output": 75.0},
}

# Thread-local context for usage attribution (safe across ThreadPoolExecutor)
_call_context = threading.local()


def set_llm_context(
    feature: str | None = None,
    ticker: str | None = None,
    period: str | None = None,
) -> None:
    """Set context for the next LLM call(s) so usage logs know what triggered them.
    Thread-safe: each ThreadPoolExecutor worker gets its own copy."""
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

# Circuit breaker — stops all LLM calls after a billing/auth error
_circuit_broken = False
_circuit_broken_reason = ""


class CircuitBrokenError(Exception):
    pass


def _trip_circuit(reason: str):
    global _circuit_broken, _circuit_broken_reason
    _circuit_broken = True
    _circuit_broken_reason = reason
    logger.error("LLM circuit breaker TRIPPED: %s — all further calls will fail fast", reason)


def _check_circuit():
    if _circuit_broken:
        raise CircuitBrokenError(f"LLM calls disabled: {_circuit_broken_reason}")


def reset_circuit():
    """Reset the circuit breaker (e.g. after topping up credits)."""
    global _circuit_broken, _circuit_broken_reason
    _circuit_broken = False
    _circuit_broken_reason = ""


def set_budget_guard(guard: BudgetGuard | None) -> None:
    """Attach (or clear) a BudgetGuard checked on every LLM call."""
    global _active_budget_guard
    _active_budget_guard = guard


def get_budget_guard() -> BudgetGuard | None:
    return _active_budget_guard


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
    return round(
        (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000, 6
    )


def _log_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int = 0,
    *,
    feature: str | None = None,
    ticker: str | None = None,
    period: str | None = None,
) -> None:
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


# ── Token usage tracking ─────────────────────────────────────────

@dataclass
class _UsageTracker:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0        # FIX: accumulate actual cost per call, not recalculated
    total_requests: int = 0
    failed_requests: int = 0

    def record(self, input_tokens: int, output_tokens: int, model: str) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += _estimate_cost(model, input_tokens, output_tokens)
        self.total_requests += 1

    def record_failure(self) -> None:
        self.failed_requests += 1

    @property
    def total(self) -> float:
        """Total estimated cost in USD across all tracked requests (all models)."""
        return round(self.total_cost_usd, 6)

    @property
    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total,
        }


usage_tracker = _UsageTracker()


# ── Clients ──────────────────────────────────────────────────────

_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None

# FIX: max_workers must be based on concurrency config, NOT token limits.
# settings.llm_max_tokens is the output token cap (~4096–8192) — using it
# as a thread count would create 8–16 workers for no reason.
_executor = ThreadPoolExecutor(
    max_workers=getattr(settings, "agent_max_parallel", 6),
    thread_name_prefix="llm_worker",
)

# Single global semaphore for ALL async LLM paths.
# Lazily created inside the event loop — do not instantiate at module level.
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily create semaphore (must be created inside a running event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(getattr(settings, "agent_max_parallel", 8))
    return _semaphore


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=120.0,
        )
    return _client


def get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=120.0,
        )
    return _async_client


# ── Retry policy ─────────────────────────────────────────────────
_retry_policy = retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


# ── JSON helpers ─────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> Any:
    last_brace = raw.rfind("}")
    if last_brace == -1:
        raise json.JSONDecodeError("No complete JSON object found", raw, 0)
    candidate = raw[: last_brace + 1]
    if not candidate.rstrip().endswith("]"):
        candidate = candidate.rstrip().rstrip(",") + "\n]"
    return json.loads(candidate)


def _clean_json_string(raw: str) -> str:
    """Clean up common LLM JSON formatting issues."""
    import re
    cleaned = raw
    if cleaned.startswith("```"):
        first_line_end = cleaned.find("\n")
        cleaned = cleaned[first_line_end + 1:] if first_line_end > 0 else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r"{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'{"\1":', cleaned)
    cleaned = re.sub(r",\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r',"\1":', cleaned)
    return cleaned


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    cleaned = _clean_json_string(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        result = _repair_truncated_json(cleaned)
        logger.warning("Repaired truncated JSON (%d chars)", len(cleaned))
        return result
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s — raw: %s", exc, raw[:1000])
        raise


# ── Sync calls ───────────────────────────────────────────────────

def call_llm(
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    model: str | None = None,
    tier: str | None = None,
    feature: str | None = None,
    ticker: str | None = None,
    period: str | None = None,
) -> str:
    client = get_client()
    model = model or (tier and _model_for_tier(tier)) or settings.llm_model
    if feature:
        _call_context.feature = feature
    if ticker:
        _call_context.ticker = ticker
    if period:
        _call_context.period = period

    _check_circuit()
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens or settings.llm_max_tokens,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.BadRequestError as e:
        logger.error(
            "Anthropic 400 error with model %s: %s — prompt length: %d chars",
            model, e.message, len(prompt),
        )
        if "credit balance" in str(e.message).lower():
            _trip_circuit("Anthropic credit balance too low")
            raise
        elif "model" in str(e.message).lower():
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
    usage_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens, model)
    _log_usage(
        model, resp.usage.input_tokens, resp.usage.output_tokens, duration_ms,
        feature=feature, ticker=ticker, period=period,
    )
    if _active_budget_guard is not None:
        _active_budget_guard.track(resp.usage.input_tokens, resp.usage.output_tokens, model)
    logger.debug(
        "LLM response: %d chars, stop=%s, tokens_in=%d, tokens_out=%d, feature=%s, ticker=%s",
        len(text), resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens,
        feature or _get_ctx("feature", "?"), ticker or _get_ctx("ticker", "?"),
    )
    return text


def call_llm_json(prompt: str, **kwargs) -> Any:
    raw = call_llm(prompt, **kwargs)
    return _parse_json(raw)


# ── Context propagation helpers ───────────────────────────────────

def _snapshot_context() -> dict:
    """Capture current thread-local context for propagation to executor threads."""
    return {
        "feature": _get_ctx("feature"),
        "ticker": _get_ctx("ticker"),
        "period": _get_ctx("period"),
    }


def _restore_context(snapshot: dict) -> None:
    for k, v in snapshot.items():
        if v is not None:
            setattr(_call_context, k, v)


# ── Async calls ───────────────────────────────────────────────────

async def call_llm_async(prompt: str, timeout_seconds: int = 90, **kwargs) -> str:
    """
    LEGACY async path: runs call_llm() in ThreadPoolExecutor.
    Returns plain str. Use for existing non-agent callers only.

    New code (e.g. BaseAgent.run) should use call_llm_native_async() instead,
    which returns {"text": str, "input_tokens": int, "output_tokens": int}.
    """
    loop = asyncio.get_event_loop()
    ctx = _snapshot_context()

    def _run():
        _restore_context(ctx)
        return call_llm(prompt, **kwargs)

    sem = _get_semaphore()
    async with sem:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_executor, _run),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out after %ds", timeout_seconds)
            raise TimeoutError(f"LLM request timed out after {timeout_seconds}s")


@_retry_policy
async def call_llm_native_async(
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    model: str | None = None,
    feature: str | None = None,
    ticker: str | None = None,
    period: str | None = None,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """
    TRUE async LLM call using AsyncAnthropic — no ThreadPoolExecutor.

    Returns a dict so callers get token counts for cost tracking:
        {"text": str, "input_tokens": int, "output_tokens": int}

    Includes:
      - Retry on RateLimitError / APIConnectionError (3 attempts, exp backoff)
      - Global concurrency semaphore (settings.agent_max_parallel)
      - Budget guard check
      - Usage logging

    This is the function BaseAgent.run() should call.
    """
    client = get_async_client()
    model = model or settings.llm_model

    _check_circuit()
    sem = _get_semaphore()
    async with sem:
        t0 = time.time()
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=(
                        temperature if temperature is not None else settings.llm_temperature
                    ),
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=timeout_seconds,
            )
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e.message).lower():
                _trip_circuit("Anthropic credit balance too low")
            raise
        except asyncio.TimeoutError:
            logger.warning("Native async LLM call timed out after %ds", timeout_seconds)
            raise TimeoutError(f"LLM request timed out after {timeout_seconds}s")

        duration_ms = int((time.time() - t0) * 1000)
        text = resp.content[0].text.strip()
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens

        usage_tracker.record(input_tokens, output_tokens, model)
        _log_usage(
            model, input_tokens, output_tokens, duration_ms,
            feature=feature, ticker=ticker, period=period,
        )
        if _active_budget_guard is not None:
            _active_budget_guard.track(input_tokens, output_tokens, model)

        logger.debug(
            "Native async LLM: %d chars | tokens_in=%d out=%d | cost=$%.4f | feature=%s",
            len(text), input_tokens, output_tokens,
            _estimate_cost(model, input_tokens, output_tokens),
            feature or _get_ctx("feature", "?"),
        )

        return {"text": text, "input_tokens": input_tokens, "output_tokens": output_tokens}


async def call_llm_json_async(prompt: str, **kwargs) -> Any:
    """Async JSON call using ThreadPoolExecutor path (legacy). Returns parsed JSON."""
    loop = asyncio.get_event_loop()
    ctx = _snapshot_context()

    def _run():
        _restore_context(ctx)
        return call_llm_json(prompt, **kwargs)

    sem = _get_semaphore()
    async with sem:
        return await loop.run_in_executor(_executor, _run)


async def call_llm_json_parallel(
    prompts: list[str],
    max_concurrency: int = 3,
    timeout_seconds: int = 120,
    **kwargs,
) -> list[Any]:
    """
    Run multiple LLM calls in parallel with limited concurrency.

    FIX: Uses a single local semaphore, NOT the global one, to avoid
    double-semaphore deadlock. The local semaphore is scoped to this
    batch call only and respects max_concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one_call(prompt: str, index: int) -> tuple[int, Any]:
        async with semaphore:
            loop = asyncio.get_event_loop()
            ctx = _snapshot_context()

            def _run():
                _restore_context(ctx)
                return call_llm_json(prompt, **kwargs)

            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(_executor, _run),
                    timeout=timeout_seconds,
                )
                return (index, result)
            except asyncio.TimeoutError:
                logger.warning("Parallel LLM call %d timed out after %ds", index + 1, timeout_seconds)
                return (index, TimeoutError(f"LLM call timed out after {timeout_seconds}s"))
            except Exception as e:
                return (index, e)

    results_tuples = await asyncio.gather(*[_one_call(p, i) for i, p in enumerate(prompts)])
    results_tuples = sorted(results_tuples, key=lambda x: x[0])

    out = []
    failed = []
    for i, r in results_tuples:
        if isinstance(r, Exception):
            logger.warning("Parallel LLM call %d/%d failed: %s", i + 1, len(prompts), r)
            usage_tracker.record_failure()
            failed.append(i)
            out.append([])
        else:
            out.append(r)
    if failed:
        logger.warning(
            "LLM parallel batch: %d/%d calls failed (indices: %s)",
            len(failed), len(prompts), failed,
        )
    return out
