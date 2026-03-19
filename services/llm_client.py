"""
LLM client with both sync and async support for parallel calls.
Includes token usage tracking and improved error handling.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import anthropic

from configs.settings import settings

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
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def call_llm(prompt: str, *, max_tokens: int | None = None, temperature: float | None = None) -> str:
    client = get_client()
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    usage_tracker.record(resp.usage.input_tokens, resp.usage.output_tokens)
    logger.debug(
        "LLM response: %d chars, stop=%s, tokens_in=%d, tokens_out=%d",
        len(text), resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens,
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


def _parse_json(raw: str) -> Any:
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
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


def call_llm_json(prompt: str, **kwargs) -> Any:
    raw = call_llm(prompt, **kwargs)
    return _parse_json(raw)


async def call_llm_json_async(prompt: str, **kwargs) -> Any:
    """Run LLM call in thread pool so multiple can run in parallel."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: call_llm_json(prompt, **kwargs))


async def call_llm_json_parallel(prompts: list[str], **kwargs) -> list[Any]:
    """Run multiple LLM calls in parallel and return results in order."""
    tasks = [call_llm_json_async(p, **kwargs) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    failed_indices = []
    for i, r in enumerate(results):
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
