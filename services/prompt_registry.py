"""
Prompt Registry — single source of truth for live prompt lookup.

All services use get_active_prompt() to retrieve the current best
prompt for a given type. AutoResearch promotes variants in the DB;
this function ensures those promotions flow through to real analyses.

Usage:
    from services.prompt_registry import get_active_prompt

    template = await get_active_prompt(db, "extraction_earnings", fallback=EARNINGS_RELEASE_EXTRACTOR)
    prompt = template.format(company=..., ...)
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Module-level cache: {prompt_type: prompt_text}
# Refreshed per-request — cheap DB read, avoids stale in-memory state
_cache: dict[str, str] = {}


async def get_active_prompt(
    db: AsyncSession,
    prompt_type: str,
    fallback: str,
) -> str:
    """
    Return the active prompt text for a given type from the DB.
    Falls back to the hardcoded prompt if:
      - no DB variant exists for this type
      - the DB query fails for any reason

    Args:
        db:          async SQLAlchemy session
        prompt_type: e.g. "extraction_earnings", "extraction_transcript"
        fallback:    the hardcoded prompt from prompts/__init__.py

    Returns:
        prompt text string ready to be .format()-ted
    """
    try:
        from apps.api.models import PromptVariant
        result = await db.execute(
            select(PromptVariant.prompt_text).where(
                PromptVariant.prompt_type == prompt_type,
                PromptVariant.is_active == True,
            ).limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            logger.debug("prompt_registry: using DB variant for '%s'", prompt_type)
            return row
    except Exception as e:
        logger.warning("prompt_registry: DB lookup failed for '%s', using fallback: %s",
                       prompt_type, str(e)[:80])
    return fallback


async def get_active_prompts_batch(
    db: AsyncSession,
    prompt_types: list[str],
    fallbacks: dict[str, str],
) -> dict[str, str]:
    """
    Fetch multiple active prompts in one DB query.
    More efficient when a service needs several prompts at startup.

    Args:
        db:           async SQLAlchemy session
        prompt_types: list of prompt type strings
        fallbacks:    dict of {prompt_type: hardcoded_fallback}

    Returns:
        dict of {prompt_type: prompt_text}
    """
    result = {pt: fallbacks.get(pt, "") for pt in prompt_types}
    try:
        from apps.api.models import PromptVariant
        rows = await db.execute(
            select(PromptVariant.prompt_type, PromptVariant.prompt_text).where(
                PromptVariant.prompt_type.in_(prompt_types),
                PromptVariant.is_active == True,
            )
        )
        for prompt_type, prompt_text in rows.all():
            if prompt_text:
                result[prompt_type] = prompt_text
                logger.debug("prompt_registry: using DB variant for '%s'", prompt_type)
    except Exception as e:
        logger.warning("prompt_registry: batch DB lookup failed, using fallbacks: %s", str(e)[:80])
    return result
