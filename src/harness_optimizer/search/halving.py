"""Successive halving: spend eval budget on candidates in proportion to how promising they are.

Every candidate is scored on a small slice of train; the better half (1/eta) moves on to a
larger slice, and so on. Slices are nested prefixes of one fixed shuffle, so each rung reuses
the examples of the previous one (and, through the gateway cache, their answers). The
incumbent is scored on every rung too, so candidates are always compared on the same examples.

Only candidates that beat the incumbent on the largest rung are returned as promising.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.dataset import Dataset
from harness_optimizer.core.results import EvalReport

Evaluate = Callable[[Candidate, Dataset], Awaitable[EvalReport]]


class Rung(BaseModel):
    size: int
    incumbent_score: float
    scores: dict[str, float]  # candidate id -> mean score on this rung
    promoted: list[str]


class HalvingResult(BaseModel):
    rungs: list[Rung] = Field(default_factory=list)
    promising: list[str] = Field(default_factory=list)  # beat the incumbent on the last rung, best first
    final_scores: dict[str, float] = Field(default_factory=dict)  # score on the last rung reached
    reached: dict[str, int] = Field(default_factory=dict)  # size of the last rung reached
    incumbent_score: float | None = None  # on the largest rung run

    def score(self, candidate_id: str) -> float | None:
        return self.final_scores.get(candidate_id)


class SuccessiveHalving(BaseModel):
    rungs: tuple[int, ...] = (4, 8, 16)
    eta: int = 2
    seed: int = 0

    def slices(self, dataset: Dataset) -> list[Dataset]:
        order = list(dataset.examples)
        random.Random(self.seed).shuffle(order)
        sizes = sorted({min(s, len(order)) for s in self.rungs if s > 0})
        return [Dataset(name=f"{dataset.name}[sh{n}@{self.seed}]", examples=order[:n]) for n in sizes]

    async def run(
        self,
        incumbent: Candidate,
        candidates: list[Candidate],
        dataset: Dataset,
        evaluate: Evaluate,
    ) -> HalvingResult:
        result = HalvingResult()
        alive = list(candidates)
        slices = self.slices(dataset)
        for i, data in enumerate(slices):
            if not alive:
                break
            reports = await asyncio.gather(evaluate(incumbent, data), *(evaluate(c, data) for c in alive))
            inc_score = reports[0].mean()
            scores = {c.id: rep.mean() for c, rep in zip(alive, reports[1:], strict=True)}
            for cid, s in scores.items():
                result.final_scores[cid] = s
                result.reached[cid] = len(data)
            result.incumbent_score = inc_score
            ranked = sorted(alive, key=lambda c: scores[c.id], reverse=True)
            last = i == len(slices) - 1
            if last:
                promoted = [c for c in ranked if scores[c.id] > inc_score]
            else:
                promoted = ranked[: max(1, math.ceil(len(ranked) / self.eta))]
            result.rungs.append(
                Rung(size=len(data), incumbent_score=inc_score, scores=scores, promoted=[c.id for c in promoted])
            )
            alive = promoted
        if slices and result.rungs and result.rungs[-1].size == len(slices[-1]):
            result.promising = list(result.rungs[-1].promoted)
        return result
