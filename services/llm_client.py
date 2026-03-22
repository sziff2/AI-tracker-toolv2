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


def _clean_json_string(raw: str) -> str:
    """Clean up common LLM JSON formatting issues."""
    import re
    cleaned = raw

    # Remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # Fix common issues: control characters inside strings that break JSON
    # Replace literal newlines/tabs inside JSON string values with escaped versions
    # This regex finds strings and normalizes control chars within them
    def escape_control_chars_in_string(match):
        s = match.group(0)
        # Replace actual control chars with escaped versions
        s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return s

    # Process strings (anything between quotes, handling escaped quotes)
    # This is a simplified approach - fix obvious issues
    cleaned = re.sub(r'(?<!\\)"[^"]*(?<!\\)"', escape_control_chars_in_string, cleaned)

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
