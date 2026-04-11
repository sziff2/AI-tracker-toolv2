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

Cache:
    File prompts are cached in memory after first load. Call clear_prompt_cache()
    if you update a .txt file at runtime (e.g. during development or AutoResearch).
    DB prompts are always fetched fresh (no cache) to pick up Prompt Lab promotions.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────
# File loading — cached
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=50)
def _load_file(path: Path) -> str | None:
    """Load a prompt template from disk. Cached per path for process lifetime.
    Call clear_prompt_cache() to invalidate after editing a file at runtime."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def clear_prompt_cache() -> None:
    """
    Invalidate the in-memory prompt file cache.

    Call this when:
      - A new prompt variant is promoted in Prompt Lab
      - A .txt prompt file is edited at runtime (development / AutoResearch)
      - Tests want a clean slate
    """
    _load_file.cache_clear()
    logger.debug("Prompt file cache cleared")


# ─────────────────────────────────────────────────────────────────
# System blocks
# ─────────────────────────────────────────────────────────────────

def _load_system_block(name: str) -> str:
    """Load a system block from prompts/system/{name}.txt. Returns '' if missing."""
    path = _PROMPTS_DIR / "system" / f"{name}.txt"
    return _load_file(path) or ""


# ─────────────────────────────────────────────────────────────────
# DB lookup (always fresh — no cache — picks up Prompt Lab promotions)
# ─────────────────────────────────────────────────────────────────

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
                .where(PromptVariant.is_active == True)  # noqa: E712
                .limit(1)
            )
            variant = result.scalar_one_or_none()
            if variant:
                logger.debug(
                    "Loaded prompt for '%s' from DB (variant: %s)", agent_id, variant.variant_name
                )
                return variant.prompt_text
    except Exception as exc:
        logger.debug("DB prompt lookup failed for '%s': %s", agent_id, exc)
    return None


# ─────────────────────────────────────────────────────────────────
# File lookup
# ─────────────────────────────────────────────────────────────────

def _load_from_file(agent_id: str) -> str | None:
    """Load prompt from file — checks agents/ then extraction/ directories."""
    path = _PROMPTS_DIR / "agents" / f"{agent_id}.txt"
    text = _load_file(path)
    if text:
        return text

    path = _PROMPTS_DIR / "extraction" / f"{agent_id}.txt"
    return _load_file(path)


# ─────────────────────────────────────────────────────────────────
# Template substitution helpers
# ─────────────────────────────────────────────────────────────────

def _flatten_for_format(d: dict) -> dict[str, str]:
    """
    Flatten a potentially nested dict into string values for use with format_map.

    Nested dicts and lists are JSON-serialised to a readable string so they
    render correctly inside a prompt. Without this, format_map would write
    Python repr (e.g. "{'key': 'val'}") instead of valid JSON.

    Example:
        {"macro_regime": {"label": "risk_off", "rates": "higher_for_longer"}}
        → {"macro_regime": '{"label": "risk_off", "rates": "higher_for_longer"}'}
    """
    flat: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, indent=2)
        elif v is None:
            flat[k] = ""
        else:
            flat[k] = str(v)
    return flat


def _safe_format(template: str, substitutions: dict, context: str = "") -> str:
    """
    Apply {key} substitutions, warning on any missing keys.

    Unlike bare str.format_map, this logs a warning when a placeholder is
    unresolved so the developer knows the inputs dict is incomplete. Unresolved
    placeholders are left as-is (not raised) so partial substitution is allowed
    for system blocks, but the warning makes debugging much faster.

    Args:
        template:      The prompt string with {key} placeholders.
        substitutions: Dict of values to substitute.
        context:       Label for log messages (e.g. "financial_analyst/main").
    """
    flat = _flatten_for_format(substitutions)

    class _WarnOnMissing(dict):
        """Custom dict that logs a warning instead of raising KeyError."""
        def __missing__(self, key: str) -> str:
            logger.warning(
                "Prompt template variable '{%s}' not found in inputs [%s]",
                key, context or "unknown",
            )
            return "{" + key + "}"  # leave placeholder in-place

    try:
        return template.format_map(_WarnOnMissing(flat))
    except (ValueError, IndexError) as exc:
        # Malformed placeholder — log and return template unchanged
        logger.warning("format_map failed for %s: %s", context, exc)
        return template


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_prompt(
    agent_id: str,
    inputs: dict | None = None,
    include_context_contract: bool = True,
    include_output_constraints: bool = True,
) -> str:
    """
    Load and assemble a prompt for an agent.

    Lookup chain:
      1. DB prompt_registry (active variant for this agent_id) — always fresh
      2. File: prompts/agents/{agent_id}.txt or prompts/extraction/{agent_id}.txt — cached
      3. Raises FileNotFoundError if neither found

    System blocks (context_contract_block, output_constraints) are prepended
    automatically unless disabled. Template variables in {inputs} are
    substituted; missing variables log a WARNING and are left as-is.

    Args:
        agent_id:                  Agent identifier (e.g. "financial_analyst").
        inputs:                    Dict of values to substitute into {placeholders}.
        include_context_contract:  Prepend context_contract_block.txt if
                                   inputs["context_contract"] is present.
        include_output_constraints: Prepend output_constraints.txt.

    Returns:
        Assembled prompt string, ready to pass to the LLM.

    Raises:
        FileNotFoundError: If no prompt is found in DB or on disk.
    """
    inputs = inputs or {}

    # 1. Try DB (fresh every call — picks up Prompt Lab promotions immediately)
    template = _load_from_db(agent_id)
    source = "db"

    # 2. Try file (cached)
    if template is None:
        template = _load_from_file(agent_id)
        source = "file"

    if template is None:
        raise FileNotFoundError(
            f"No prompt found for agent '{agent_id}'. "
            f"Checked DB and: {_PROMPTS_DIR / 'agents' / agent_id}.txt, "
            f"{_PROMPTS_DIR / 'extraction' / agent_id}.txt"
        )

    # Assemble: system blocks first, then agent template
    parts: list[str] = []

    # Context contract block — prepended if the contract was passed in inputs
    if include_context_contract and "context_contract" in inputs:
        block = _load_system_block("context_contract_block")
        if block:
            contract = inputs["context_contract"]
            # context_contract may be a dict (from DB JSONB) — flatten to strings
            # Merge macro_assumptions into top level so {rates}, {usd} etc. resolve
            if isinstance(contract, dict):
                merged = dict(contract)
                if isinstance(merged.get("macro_assumptions"), dict):
                    merged.update(merged.pop("macro_assumptions"))
                if isinstance(merged.get("analyst_overrides"), dict):
                    merged.update(merged.pop("analyst_overrides"))
                contract_subs = _flatten_for_format(merged)
            else:
                contract_subs = {"contract": str(contract)}
            parts.append(_safe_format(block, contract_subs, context=f"{agent_id}/context_contract"))

    # Output constraints block — appended to every agent prompt for consistency
    if include_output_constraints:
        block = _load_system_block("output_constraints")
        if block:
            parts.append(block)

    parts.append(template)
    assembled = "\n\n".join(parts)

    # Substitute all {key} placeholders from inputs
    final = _safe_format(assembled, inputs, context=f"{agent_id}/main")

    logger.debug("Loaded prompt for '%s' from %s (%d chars)", agent_id, source, len(final))
    return final
