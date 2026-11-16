"""Ablations: does each optional component of the optimizer actually help?

A variant is a set of PipelineConfig overrides (for example "critic on"). Every variant runs
once per seed; the seed changes the train/val/test split and the minibatches, so the spread
across seeds shows how much of a difference is real. All variants of one seed share the same
split, so they are compared on identical data.

Cost is reported as *total* LLM calls (cache hits included), split into target calls (eval
cost) and agent calls. Cache hits are not subtracted on purpose: variants share one response
cache, so whichever variant runs first would otherwise look more expensive.

Results are appended to `rows.jsonl` as each run finishes; re-running with the same output
directory skips runs that are already there, so a long ablation can be interrupted and resumed.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from harness_optimizer.agents.pipeline import OptimizationPipeline, PipelineConfig
from harness_optimizer.core.dataset import Dataset
from harness_optimizer.core.prompt import PromptTemplate
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.eval.metrics.base import Metric
from harness_optimizer.llm.gateway import LLMGateway
from harness_optimizer.policy.budget import Budget
from harness_optimizer.search.acceptance import PairedImprovement

log = logging.getLogger(__name__)

TARGET_ROLE = "target"


class Variant(BaseModel):
    name: str
    config: dict = Field(default_factory=dict)  # PipelineConfig overrides


# The plain pipeline, each component added on its own, and everything together.
PRESETS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant(name="base", config={"use_reflector": False}),
        Variant(name="reflector", config={"use_reflector": True}),
        Variant(name="critic", config={"use_reflector": False, "use_critic": True}),
        Variant(name="halving", config={"use_reflector": False, "screen": "halving"}),
        Variant(name="full", config={"use_reflector": True, "use_critic": True, "screen": "halving"}),
    ]
}


class RunRow(BaseModel):
    variant: str
    seed: int
    val_baseline: float | None = None
    val_best: float | None = None
    test_baseline: float | None = None
    test_best: float | None = None
    rounds: int = 0
    accepted: int = 0
    stop_reason: str | None = None
    target_calls: int = 0
    agent_calls: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def test_gain(self) -> float | None:
        if self.test_best is None or self.test_baseline is None:
            return None
        return self.test_best - self.test_baseline


class VariantSummary(BaseModel):
    variant: str
    runs: int
    failed_runs: int
    test_best_mean: float | None
    test_best_std: float | None
    test_gain_mean: float | None
    accepted_mean: float | None
    target_calls_mean: float | None
    agent_calls_mean: float | None
    seconds_mean: float | None


class AblationResult(BaseModel):
    rows: list[RunRow]
    summaries: list[VariantSummary]

    def table(self) -> str:
        """Markdown table, one line per variant."""

        def f(x: float | None, digits: int = 3) -> str:
            return "-" if x is None else f"{x:.{digits}f}"

        lines = [
            "| variant | runs | test best (mean ± sd) | test gain | accepted/run | target calls | agent calls | seconds |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in self.summaries:
            runs = f"{s.runs}" + (f" ({s.failed_runs} failed)" if s.failed_runs else "")
            lines.append(
                f"| {s.variant} | {runs} | {f(s.test_best_mean)} ± {f(s.test_best_std)} | "
                f"{f(s.test_gain_mean)} | {f(s.accepted_mean, 1)} | {f(s.target_calls_mean, 0)} | "
                f"{f(s.agent_calls_mean, 0)} | {f(s.seconds_mean, 0)} |"
            )
        return "\n".join(lines)


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def summarize(rows: list[RunRow], order: list[str] | None = None) -> list[VariantSummary]:
    names = order or list(dict.fromkeys(r.variant for r in rows))
    out = []
    for name in names:
        rs = [r for r in rows if r.variant == name]
        ok = [r for r in rs if r.error is None]

        def mean(xs: list[float]) -> float | None:
            return _mean_std(xs)[0]

        best_m, best_sd = _mean_std([r.test_best for r in ok if r.test_best is not None])
        out.append(VariantSummary(
            variant=name,
            runs=len(rs),
            failed_runs=len(rs) - len(ok),
            test_best_mean=best_m,
            test_best_std=best_sd,
            test_gain_mean=mean([r.test_gain for r in ok if r.test_gain is not None]),
            accepted_mean=mean([float(r.accepted) for r in ok]),
            target_calls_mean=mean([float(r.target_calls) for r in ok]),
            agent_calls_mean=mean([float(r.agent_calls) for r in ok]),
            seconds_mean=mean([r.seconds for r in ok]),
        ))
    return out


def _load_rows(path: Path) -> list[RunRow]:
    if not path.exists():
        return []
    return [RunRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_ablation(
    prompt: PromptTemplate,
    dataset: Dataset,
    variants: list[Variant],
    seeds: list[int],
    make_gateway: Callable[[str, int], LLMGateway],
    target: ModelConfig,
    metrics: list[Metric],
    agent_model: ModelConfig | None = None,
    base_config: PipelineConfig | None = None,
    budget: Budget | None = None,
    acceptance: PairedImprovement | None = None,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    stratify_by: str | None = None,
    out_dir: str | Path | None = None,
    on_row: Callable[[RunRow], None] | None = None,
) -> AblationResult:
    """`make_gateway(variant, seed)` must return a fresh gateway (fresh cost ledger) per run;
    sharing one response cache between them is fine and saves a lot of time."""
    base_config = base_config or PipelineConfig()
    budget = budget or Budget()
    rows_path = Path(out_dir) / "rows.jsonl" if out_dir else None
    if rows_path:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(rows_path) if rows_path else []
    done = {(r.variant, r.seed) for r in rows if r.error is None}
    rows = [r for r in rows if r.error is None]  # failed runs are retried

    for seed in seeds:
        splits = dataset.split(*fractions, seed=seed, stratify_by=stratify_by)
        for variant in variants:
            if (variant.name, seed) in done:
                continue
            gateway = make_gateway(variant.name, seed)
            config = base_config.model_copy(update={**variant.config, "seed": seed})
            pipeline = OptimizationPipeline(
                gateway, target=target, metrics=metrics, splits=splits, agent_model=agent_model,
                config=config, budget=budget, acceptance=acceptance,
            )
            row = RunRow(variant=variant.name, seed=seed)
            t0 = time.monotonic()
            try:
                result = await pipeline.arun(prompt)
                row.val_baseline, row.val_best = result.val["baseline"], result.val["best"]
                row.test_baseline, row.test_best = result.test["baseline"], result.test["best"]
                row.rounds = len(result.rounds)
                row.accepted = sum(r.accepted_id is not None for r in result.rounds)
                row.stop_reason = result.stop_reason
                if out_dir:
                    run_dir = Path(out_dir) / f"{variant.name}-seed{seed}"
                    pipeline.store.save(run_dir)
                    (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            except Exception as e:  # one broken run must not lose the others
                log.exception("run %s seed %d failed", variant.name, seed)
                row.error = f"{type(e).__name__}: {e}"
            row.seconds = time.monotonic() - t0
            for role, stats in gateway.ledger.by_role.items():
                if role == TARGET_ROLE:
                    row.target_calls += stats.calls
                else:
                    row.agent_calls += stats.calls
            rows.append(row)
            if rows_path:
                with rows_path.open("a", encoding="utf-8") as f:
                    f.write(row.model_dump_json() + "\n")
            if on_row:
                on_row(row)

    order = [v.name for v in variants]
    selected = [r for r in rows if r.variant in order and r.seed in seeds]
    result = AblationResult(rows=selected, summaries=summarize(selected, order))
    if out_dir:
        (Path(out_dir) / "summary.md").write_text(result.table() + "\n", encoding="utf-8")
        (Path(out_dir) / "summary.json").write_text(
            json.dumps([s.model_dump() for s in result.summaries], indent=2), encoding="utf-8"
        )
    return result
