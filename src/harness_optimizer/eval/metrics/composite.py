"""Combine metrics: weighted average plus hard gates."""

from __future__ import annotations

import asyncio

from harness_optimizer.core.dataset import Example
from harness_optimizer.core.results import Score
from harness_optimizer.eval.metrics.base import Metric


class Composite(Metric):
    """score = 0 if any gate fails, else the weighted mean of the weighted metrics.

    Example: ``Composite({ExactMatch(): 1.0}, gates=[JsonValid()])``.
    """

    name = "composite"

    def __init__(
        self,
        weighted: dict[Metric, float],
        gates: list[Metric] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not weighted:
            raise ValueError("need at least one weighted metric")
        self.weighted = weighted
        self.gates = gates or []

    async def ascore(self, example: Example, output: str) -> Score:
        gate_scores = await asyncio.gather(*(g.ascore(example, output) for g in self.gates))
        for gate, s in zip(self.gates, gate_scores, strict=True):
            if not s.passed:
                return Score(value=0.0, rationale=f"gate {gate.name} failed: {s.rationale}")

        metrics = list(self.weighted)
        scores = await asyncio.gather(*(m.ascore(example, output) for m in metrics))
        total_w = sum(self.weighted.values())
        value = sum(self.weighted[m] * s.value for m, s in zip(metrics, scores, strict=True))
        value /= total_w
        rationale = "; ".join(f"{m.name}: {s.rationale}" for m, s in zip(metrics, scores) if s.rationale)
        return Score(
            value=value,
            rationale=rationale or None,
            details={m.name: s.value for m, s in zip(metrics, scores, strict=True)},
        )
