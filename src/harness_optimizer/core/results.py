"""Evaluation results: per-example records and the aggregate report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness_optimizer.llm.base import Usage


class Score(BaseModel):
    """A metric's verdict on one output. `value` is always normalized to [0, 1]."""

    value: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.value >= 1.0


class ExampleResult(BaseModel):
    example_id: str
    sample_index: int = 0
    inputs: dict[str, Any]
    expected: Any = None
    output: str | None
    error: str | None = None
    stop_reason: str | None = None
    scores: dict[str, Score]
    latency_s: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    cached: bool = False


class Failure(BaseModel):
    """What optimizer agents will read: one failed example, fully contextualized."""

    example_id: str
    inputs: dict[str, Any]
    expected: Any
    output: str | None
    error: str | None
    score: float
    rationale: str | None

    def to_text(self) -> str:
        lines = [f"[example {self.example_id}] score={self.score:.2f}"]
        lines.append(f"inputs: {self.inputs}")
        if self.expected is not None:
            lines.append(f"expected: {self.expected!r}")
        lines.append(f"got: {self.error and f'ERROR {self.error}' or repr(self.output)}")
        if self.rationale:
            lines.append(f"why: {self.rationale}")
        return "\n".join(lines)


class EvalReport(BaseModel):
    candidate_id: str
    prompt_fingerprint: str
    dataset_name: str
    split: str
    target_model: str
    metric_names: list[str]
    primary_metric: str
    results: list[ExampleResult]

    # ---- aggregates -----------------------------------------------------
    def per_example(self, metric: str | None = None) -> dict[str, float]:
        """Mean score per example (averaging over samples). The unit for statistics."""
        metric = metric or self.primary_metric
        acc: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            acc[r.example_id].append(r.scores[metric].value)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def mean(self, metric: str | None = None) -> float:
        vals = list(self.per_example(metric).values())
        return sum(vals) / len(vals) if vals else 0.0

    def ci(self, metric: str | None = None, confidence: float = 0.95, seed: int = 0):
        from harness_optimizer.eval.stats import bootstrap_ci

        return bootstrap_ci(list(self.per_example(metric).values()), confidence, seed=seed)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for r in self.results:
            total = total + r.usage
        return total

    @property
    def error_rate(self) -> float:
        return sum(r.error is not None for r in self.results) / max(len(self.results), 1)

    def failures(
        self, metric: str | None = None, threshold: float = 1.0, k: int | None = None
    ) -> list[Failure]:
        """Results scoring below `threshold`, worst first."""
        metric = metric or self.primary_metric
        bad = [r for r in self.results if r.scores[metric].value < threshold]
        bad.sort(key=lambda r: r.scores[metric].value)
        out = [
            Failure(
                example_id=r.example_id,
                inputs=r.inputs,
                expected=r.expected,
                output=r.output,
                error=r.error,
                score=r.scores[metric].value,
                rationale=r.scores[metric].rationale,
            )
            for r in bad
        ]
        return out[:k] if k else out

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "split": self.split,
            "dataset": self.dataset_name,
            "target_model": self.target_model,
            "n_examples": len(self.per_example()),
            "n_results": len(self.results),
            "error_rate": round(self.error_rate, 4),
            "cost_usd": round(self.total_cost_usd, 6),
            "metrics": {},
        }
        for m in self.metric_names:
            ci = self.ci(m)
            out["metrics"][m] = {
                "mean": round(ci.mean, 4),
                "ci95": [round(ci.low, 4), round(ci.high, 4)],
            }
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> EvalReport:
        return cls.model_validate_json((Path(directory) / "report.json").read_text("utf-8"))
