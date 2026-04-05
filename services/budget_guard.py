"""
Budget guard for LLM pipelines.
Tracks cumulative token spend and enforces a hard cap.
"""
import asyncio
import logging
from configs.settings import settings

logger = logging.getLogger(__name__)

# Approximate pricing per 1K tokens (update as Anthropic changes rates)
_RATES = {
    "claude-haiku-4-5-20251001":   {"input": 0.001,  "output": 0.005},
    "claude-sonnet-4-20250514":    {"input": 0.003,  "output": 0.015},
    "claude-opus-4-20250514":      {"input": 0.015,  "output": 0.075},
}


class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = budget_usd or settings.autorun_budget_usd
        self.total_spend = 0.0
        self.call_count = 0
        self._warned = False

    def track(self, input_tokens: int, output_tokens: int, model: str):
        rate = _RATES.get(model, _RATES["claude-sonnet-4-20250514"])
        cost = (input_tokens / 1000) * rate["input"] + (output_tokens / 1000) * rate["output"]
        self.total_spend += cost
        self.call_count += 1

        if not self._warned and self.total_spend >= self.budget_usd * 0.8:
            logger.warning("[BUDGET] At %.0f%% of $%.2f budget ($%.2f spent)",
                          self.total_spend / self.budget_usd * 100, self.budget_usd, self.total_spend)
            self._warned = True
            try:
                from services.slack_alerts import send_slack
                asyncio.create_task(send_slack(
                    f"Budget warning: ${self.total_spend:.2f} of ${self.budget_usd:.2f} "
                    f"({self.total_spend / self.budget_usd * 100:.0f}%) after {self.call_count} calls",
                    "warn",
                ))
            except RuntimeError:
                pass  # no running event loop (e.g. sync context)

        if self.total_spend >= self.budget_usd:
            try:
                from services.slack_alerts import send_slack
                asyncio.create_task(send_slack(
                    f"Budget EXCEEDED: ${self.total_spend:.2f} >= ${self.budget_usd:.2f} "
                    f"after {self.call_count} calls",
                    "error",
                ))
            except RuntimeError:
                pass
            raise BudgetExceeded(
                f"Budget exceeded: ${self.total_spend:.2f} >= ${self.budget_usd:.2f} "
                f"after {self.call_count} calls"
            )

        return cost
