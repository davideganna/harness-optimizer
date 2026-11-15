"""Metric interface.

A metric maps (example, model output) -> Score in [0, 1]. Metrics may be sync (override
`score`) or async (override `ascore`, e.g. LLM judges). An optional `extract` callable
pulls the answer out of a longer output before scoring (e.g. the last line, a JSON field).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

from harness_optimizer.core.dataset import Example
from harness_optimizer.core.results import Score

Extractor = Callable[[str], str]


class Metric(ABC):
    name: str = "metric"

    def __init__(self, name: str | None = None, extract: Extractor | None = None) -> None:
        if name:
            self.name = name
        self.extract = extract

    def prepare(self, output: str) -> str:
        return self.extract(output) if self.extract else output

    def score(self, example: Example, output: str) -> Score:
        raise NotImplementedError(f"{type(self).__name__} must implement score() or ascore()")

    async def ascore(self, example: Example, output: str) -> Score:
        return self.score(example, output)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
