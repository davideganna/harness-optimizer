"""Budget limits and stopping rules, enforced by code rather than by asking an agent."""

from __future__ import annotations

import time

from pydantic import BaseModel

from harness_optimizer.llm.gateway import CostLedger


class Budget(BaseModel):
    max_rounds: int = 5
    max_llm_calls: int | None = None  # uncached calls across all roles
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    patience: int = 2  # stop after this many consecutive rounds without an accepted candidate


class BudgetTracker:
    def __init__(self, budget: Budget, ledger: CostLedger) -> None:
        self.budget = budget
        self.ledger = ledger
        self.started = time.monotonic()
        self.rounds = 0
        self.rounds_without_improvement = 0

    @property
    def uncached_calls(self) -> int:
        return sum(s.calls - s.cache_hits for s in self.ledger.by_role.values())

    def end_round(self, improved: bool) -> None:
        self.rounds += 1
        self.rounds_without_improvement = 0 if improved else self.rounds_without_improvement + 1

    def stop_reason(self) -> str | None:
        """Why the run should stop now, or None to continue."""
        b = self.budget
        if self.rounds >= b.max_rounds:
            return f"max_rounds ({b.max_rounds}) reached"
        if self.rounds_without_improvement >= b.patience:
            return f"no accepted candidate for {b.patience} rounds"
        if b.max_llm_calls is not None and self.uncached_calls >= b.max_llm_calls:
            return f"LLM call budget ({b.max_llm_calls}) exhausted"
        if b.max_cost_usd is not None and self.ledger.total_cost_usd >= b.max_cost_usd:
            return f"cost budget (${b.max_cost_usd}) exhausted"
        if b.max_seconds is not None and time.monotonic() - self.started >= b.max_seconds:
            return f"time budget ({b.max_seconds}s) exhausted"
        return None
