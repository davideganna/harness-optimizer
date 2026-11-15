"""Block G - budget and policy (budget + stopping implemented; overfitting guards next).

Hard limits (USD, calls, wall-clock, candidates), stopping rules (no significant val gain
for N rounds), and overfitting guards (train/val gap tracking; test evaluated exactly once).
Limits are enforced inside agent tools, never by asking the agent to respect them.
"""

from harness_optimizer.policy.budget import Budget, BudgetTracker

__all__ = ["Budget", "BudgetTracker"]
