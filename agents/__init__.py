"""
Agent package for the investment research platform.

Public API:
    BaseAgent     — abstract base class for all agents
    AgentResult   — standardised result dataclass
    AgentTier     — tier enum (TASK, DOCUMENT, SPECIALIST, ...)
    AgentRegistry — registers and discovers agents

Auto-discovery:
    AgentRegistry.autodiscover() is NOT called here to avoid triggering a
    full package scan on every import (which would break tests and slow
    cold starts before the DB is ready).

    It must be called once at application startup, after the database
    connection pool is initialised. Add this to the FastAPI lifespan in
    apps/api/main.py:

        from agents.registry import AgentRegistry

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # ... DB init ...
            AgentRegistry.autodiscover()
            warnings = AgentRegistry.validate_dependencies()
            for w in warnings:
                logger.warning("Agent wiring: %s", w)
            yield
            # ... cleanup ...
"""

from agents.base import AgentResult, AgentTier, BaseAgent
from agents.registry import AgentRegistry

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AgentTier",
    "AgentRegistry",
]
