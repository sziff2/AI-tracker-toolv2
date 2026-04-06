"""
Prompt loader — resolves prompts from DB variants or template files.

Lookup chain:
  1. Check prompt_registry DB for active variant (A/B testing override)
  2. If no DB variant, load from file prompts/agents/{agent_id}.txt
     or prompts/extraction/{name}.txt
  3. Inject system blocks (context_contract, output_constraints) automatically

Usage:
    from prompts.loader import load_prompt
    prompt = load_prompt("financial_analyst", inputs={"ticker": "ALLY US", ...})
"""

import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=50)
def _load_file(path: Path) -> str | None:
    """Load a prompt template from disk. Cached."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _load_system_block(name: str) -> str:
    """Load a system block (context_contract_block, output_constraints)."""
    path = _PROMPTS_DIR / "system" / f"{name}.txt"
    return _load_file(path) or ""


def _load_from_db(agent_id: str) -> str | None:
    """Check prompt_registry DB for an active variant. Returns None if not found."""
    try:
        from apps.api.database import SyncSessionLocal
        from apps.api.models import PromptVariant
        from sqlalchemy import select

        with SyncSessionLocal() as db:
            result = db.execute(
                select(PromptVariant)
                .where(PromptVariant.prompt_type == agent_id)
                .where(PromptVariant.is_active == True)
                .limit(1)
            )
            variant = result.scalar_one_or_none()
            if variant:
                logger.debug("Loaded prompt for %s from DB (variant: %s)", agent_id, variant.variant_name)
                return variant.prompt_text
    except Exception as exc:
        logger.debug("DB prompt lookup failed for %s: %s", agent_id, exc)
    return None


def _load_from_file(agent_id: str) -> str | None:
    """Load prompt from file — checks agents/ then extraction/ directories."""
    # Try agents/ first
    path = _PROMPTS_DIR / "agents" / f"{agent_id}.txt"
    text = _load_file(path)
    if text:
        return text

    # Try extraction/
    path = _PROMPTS_DIR / "extraction" / f"{agent_id}.txt"
    text = _load_file(path)
    if text:
        return text

    return None


def load_prompt(
    agent_id: str,
    inputs: dict | None = None,
    include_context_contract: bool = True,
    include_output_constraints: bool = True,
) -> str:
    """
    Load and assemble a prompt for an agent.

    Lookup chain:
      1. DB prompt_registry (active variant for this agent_id)
      2. File: prompts/agents/{agent_id}.txt or prompts/extraction/{agent_id}.txt
      3. Raises FileNotFoundError if neither found

    System blocks are prepended automatically unless disabled.
    Template variables in {inputs} are substituted via str.format_map().
    """
    # 1. Try DB
    template = _load_from_db(agent_id)
    source = "db"

    # 2. Try file
    if template is None:
        template = _load_from_file(agent_id)
        source = "file"

    if template is None:
        raise FileNotFoundError(f"No prompt found for agent '{agent_id}' (checked DB + files)")

    # Assemble system blocks
    parts = []
    if include_context_contract and inputs and "context_contract" in inputs:
        block = _load_system_block("context_contract_block")
        if block:
            try:
                parts.append(block.format_map(inputs["context_contract"]))
            except (KeyError, IndexError):
                parts.append(block)  # partial substitution is fine

    if include_output_constraints:
        block = _load_system_block("output_constraints")
        if block:
            parts.append(block)

    parts.append(template)
    prompt = "\n\n".join(parts)

    # Substitute template variables from inputs
    if inputs:
        try:
            prompt = prompt.format_map(inputs)
        except (KeyError, IndexError):
            pass  # partial substitution — unresolved {vars} stay as-is

    logger.debug("Loaded prompt for %s from %s (%d chars)", agent_id, source, len(prompt))
    return prompt
