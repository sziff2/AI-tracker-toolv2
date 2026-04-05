"""
HTTP fetch with exponential backoff and jitter.

Handles 429 (rate limit), 502/503/504 (server errors), and timeouts.
"""

import asyncio
import logging
import random

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 502, 503, 504}
_MAX_BACKOFF = 30.0


async def fetch_with_retry(
    url: str,
    client: httpx.AsyncClient,
    *,
    retries: int = 3,
    timeout: float = 15.0,
    **kwargs,
) -> httpx.Response:
    """Fetch a URL with retries, exponential backoff, and jitter.

    Args:
        url: The URL to fetch.
        client: An httpx.AsyncClient instance.
        retries: Maximum number of retry attempts (default 3).
        timeout: Per-request timeout in seconds (default 15).
        **kwargs: Extra keyword arguments forwarded to client.get().

    Returns:
        httpx.Response on success.

    Raises:
        httpx.HTTPStatusError: If a non-retryable HTTP error occurs.
        httpx.TimeoutException: If all retries are exhausted due to timeouts.
    """
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = await asyncio.wait_for(
                client.get(url, **kwargs),
                timeout=timeout,
            )

            if resp.status_code not in _RETRYABLE_STATUS:
                return resp

            # Retryable status code
            if attempt >= retries:
                resp.raise_for_status()
                return resp  # unreachable but satisfies type checker

            # Honour Retry-After header on 429
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = min(float(retry_after), _MAX_BACKOFF)
                    except ValueError:
                        wait = _backoff(attempt)
                else:
                    wait = _backoff(attempt)
            else:
                wait = _backoff(attempt)

            logger.info(
                "[HTTP-RETRY] %s returned %s, retrying in %.1fs (attempt %d/%d)",
                url, resp.status_code, wait, attempt + 1, retries,
            )
            await asyncio.sleep(wait)

        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            last_exc = e
            if attempt >= retries:
                raise
            wait = _backoff(attempt)
            logger.info(
                "[HTTP-RETRY] %s timed out, retrying in %.1fs (attempt %d/%d)",
                url, wait, attempt + 1, retries,
            )
            await asyncio.sleep(wait)

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise httpx.TimeoutException(f"All {retries} retries exhausted for {url}")


def _backoff(attempt: int) -> float:
    """Jittered exponential backoff capped at _MAX_BACKOFF."""
    base = min(2 ** attempt, _MAX_BACKOFF)
    return base * (0.5 + random.random() * 0.5)
