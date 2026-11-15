"""Acceptance rule: when does a challenger replace the incumbent?

A challenger is accepted only if a paired bootstrap on val shows it beats the incumbent by
more than `min_delta`, with one-sided p <= alpha. That is what keeps a greedy search from
climbing on noise.

Known limitation (research backlog): comparing several challengers against the same val set
across many rounds is a multiple-comparisons problem, so the final test evaluation is the
only unbiased estimate.
"""

from __future__ import annotations

from pydantic import BaseModel

from harness_optimizer.core.results import EvalReport
from harness_optimizer.eval.stats import PairedComparison, paired_bootstrap


class PairedImprovement(BaseModel):
    alpha: float = 0.1
    min_delta: float = 0.0
    seed: int = 0

    def compare(self, incumbent: EvalReport, challenger: EvalReport) -> PairedComparison:
        return paired_bootstrap(incumbent.per_example(), challenger.per_example(), seed=self.seed)

    def accepts(self, cmp: PairedComparison) -> bool:
        return cmp.mean_diff > self.min_delta and cmp.p_value <= self.alpha
