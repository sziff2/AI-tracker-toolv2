"""
Agent registry — discovers and manages available agents.
"""
import logging
from typing import Type

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry of all available agents."""

    _agents: dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """Decorator to register an agent class."""
        agent_id = agent_class.agent_id
        if agent_id in cls._agents:
            logger.warning("Agent %s already registered, overwriting", agent_id)
        cls._agents[agent_id] = agent_class
        logger.debug("Registered agent: %s (tier=%s)", agent_id, agent_class.tier)
        return agent_class

    @classmethod
    def get(cls, agent_id: str) -> Type[BaseAgent] | None:
        """Get an agent class by ID."""
        return cls._agents.get(agent_id)

    @classmethod
    def get_all(cls) -> dict[str, Type[BaseAgent]]:
        """Get all registered agents."""
        return dict(cls._agents)

    @classmethod
    def get_by_tier(cls, tier) -> list[Type[BaseAgent]]:
        """Get all agents for a given tier."""
        return [a for a in cls._agents.values() if a.tier == tier]

    @classmethod
    def get_execution_order(cls, agent_ids: list[str] | None = None) -> list[Type[BaseAgent]]:
        """Return agents in dependency-respecting execution order."""
        agents = [cls._agents[aid] for aid in (agent_ids or cls._agents.keys()) if aid in cls._agents]
        # Topological sort by depends_on
        ordered = []
        seen = set()

        def visit(agent_cls):
            if agent_cls.agent_id in seen:
                return
            seen.add(agent_cls.agent_id)
            for dep_id in agent_cls.depends_on:
                dep = cls._agents.get(dep_id)
                if dep:
                    visit(dep)
            ordered.append(agent_cls)

        for a in agents:
            visit(a)
        return ordered

    @classmethod
    def clear(cls):
        """Clear all registered agents (for testing)."""
        cls._agents.clear()
