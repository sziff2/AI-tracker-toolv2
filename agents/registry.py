"""
Agent registry — discovers and manages available agents.

Auto-discovery scans the agents/ package on first use and imports all
modules that contain BaseAgent subclasses decorated with @AgentRegistry.register.

Usage:
    # Register an agent
    @AgentRegistry.register
    class FinancialAnalystAgent(BaseAgent):
        agent_id = "financial_analyst"
        ...

    # Get execution order for a pipeline
    AgentRegistry.autodiscover()
    agents = AgentRegistry.get_execution_order(["financial_analyst", "bear_case"])
"""
import importlib
import logging
import pkgutil
from typing import Type

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry of all available agents."""

    _agents: dict[str, Type[BaseAgent]] = {}
    _discovered: bool = False  # Guard so autodiscover() only scans once per process

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #

    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """Decorator to register an agent class."""
        agent_id = agent_class.agent_id
        if not agent_id or agent_id == "base":
            raise ValueError(
                f"Cannot register agent with agent_id='{agent_id}'. "
                "Set a unique agent_id on the class."
            )
        if agent_id in cls._agents:
            logger.warning("Agent '%s' already registered — overwriting", agent_id)
        cls._agents[agent_id] = agent_class
        logger.debug("Registered agent: %s (tier=%s, layer=%d)", agent_id, agent_class.tier, agent_class.get_layer(agent_class))  # type: ignore[arg-type]
        return agent_class

    # ------------------------------------------------------------------ #
    #  Auto-discovery
    # ------------------------------------------------------------------ #

    @classmethod
    def autodiscover(cls, package_name: str = "agents") -> None:
        """
        Scan the agents/ package and import all submodules.

        Any module that contains a class decorated with @AgentRegistry.register
        will be registered automatically on import. Call this once at startup
        (or before the first pipeline run) — subsequent calls are no-ops.
        """
        if cls._discovered:
            return
        cls._discovered = True

        try:
            import agents as agents_pkg  # noqa: PLC0415
            package_path = agents_pkg.__path__  # type: ignore[attr-defined]
        except ImportError:
            logger.warning("Could not import agents package for autodiscovery")
            return

        for finder, module_name, is_pkg in pkgutil.walk_packages(
            path=package_path,
            prefix=f"{package_name}.",
            onerror=lambda name: logger.warning("Error walking package: %s", name),
        ):
            if module_name in ("agents.base", "agents.registry"):
                continue  # Skip infrastructure modules
            try:
                importlib.import_module(module_name)
                logger.debug("Autodiscovered module: %s", module_name)
            except Exception as exc:
                logger.warning("Failed to import agent module '%s': %s", module_name, exc)

        logger.info("Autodiscovery complete — %d agent(s) registered", len(cls._agents))

    # ------------------------------------------------------------------ #
    #  Lookups
    # ------------------------------------------------------------------ #

    @classmethod
    def get(cls, agent_id: str) -> Type[BaseAgent] | None:
        """Get an agent class by ID."""
        return cls._agents.get(agent_id)

    @classmethod
    def get_all(cls) -> dict[str, Type[BaseAgent]]:
        """Get all registered agents (copy — safe to iterate)."""
        return dict(cls._agents)

    @classmethod
    def get_by_tier(cls, tier) -> list[Type[BaseAgent]]:
        """Get all agents for a given tier, sorted by layer."""
        return sorted(
            [a for a in cls._agents.values() if a.tier == tier],
            key=lambda a: a.get_layer(a),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ #
    #  Execution ordering
    # ------------------------------------------------------------------ #

    @classmethod
    def get_execution_order(
        cls,
        agent_ids: list[str] | None = None,
    ) -> list[Type[BaseAgent]]:
        """
        Return agents in dependency-respecting execution order (topological sort).

        Within the same dependency level, agents are ordered by layer (tier rank)
        so DOCUMENT agents always run before SPECIALIST agents, etc.

        Raises ValueError on circular dependencies.

        Args:
            agent_ids: If provided, only include these agents (plus their
                       transitive dependencies). If None, orders all registered agents.
        """
        target_ids = list(agent_ids or cls._agents.keys())
        candidates = {aid: cls._agents[aid] for aid in target_ids if aid in cls._agents}

        # Warn about any requested agents that aren't registered
        missing = [aid for aid in target_ids if aid not in cls._agents]
        if missing:
            logger.warning("Requested agents not registered: %s", missing)

        ordered: list[Type[BaseAgent]] = []
        visited: set[str] = set()
        visiting: set[str] = set()  # Cycle detection — agents currently in the call stack

        def visit(agent_cls: Type[BaseAgent]) -> None:
            aid = agent_cls.agent_id

            if aid in visited:
                return
            if aid in visiting:
                # We've hit a cycle — build the cycle path for the error message
                cycle = " → ".join(list(visiting) + [aid])
                raise ValueError(f"Circular dependency detected: {cycle}")

            visiting.add(aid)

            # Resolve dependencies — include transitive deps even if not in candidates
            for dep_id in (agent_cls.depends_on or []):
                dep_cls = cls._agents.get(dep_id)
                if dep_cls:
                    visit(dep_cls)
                else:
                    logger.warning(
                        "Agent '%s' depends on '%s' which is not registered",
                        aid, dep_id,
                    )

            visiting.discard(aid)
            visited.add(aid)
            ordered.append(agent_cls)

        # Sort candidates by layer first so stable sort within a dependency level
        for agent_cls in sorted(candidates.values(), key=lambda a: a.get_layer(a)):  # type: ignore[arg-type]
            visit(agent_cls)

        return ordered

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @classmethod
    def validate_dependencies(cls) -> list[str]:
        """
        Check that all declared dependencies are registered.
        Returns a list of warning strings (empty = all good).
        Call this at startup after autodiscover() to catch wiring errors early.
        """
        warnings: list[str] = []
        for agent_id, agent_cls in cls._agents.items():
            for dep_id in (agent_cls.depends_on or []):
                if dep_id not in cls._agents:
                    warnings.append(
                        f"Agent '{agent_id}' depends on '{dep_id}' which is not registered"
                    )
        return warnings

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents and reset discovery state (for testing)."""
        cls._agents.clear()
        cls._discovered = False
