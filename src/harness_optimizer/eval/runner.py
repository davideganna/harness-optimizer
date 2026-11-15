"""Evaluation runner: (candidate, dataset) -> EvalReport.

Rules baked in here because they matter for research validity:
  * Failed calls are kept and scored 0 - silently dropping them would bias the mean upward
    for prompts that trigger errors/refusals.
  * Template rendering errors raise immediately: they are configuration bugs, not results.
  * `samples > 1` re-runs each example with distinct cache keys, to measure output variance.
"""

from __future__ import annotations

import asyncio

from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.dataset import Dataset, Example
from harness_optimizer.core.results import EvalReport, ExampleResult, Score
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.eval.metrics.base import Metric
from harness_optimizer.llm.base import LLMError, LLMRequest, Message
from harness_optimizer.llm.gateway import LLMGateway


class EvalRunner:
    def __init__(
        self,
        gateway: LLMGateway,
        target: ModelConfig,
        metrics: list[Metric],
        primary_metric: str | None = None,
    ) -> None:
        if not metrics:
            raise ValueError("at least one metric is required")
        names = [m.name for m in metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"metric names must be unique: {names}")
        self.gateway = gateway
        self.target = target
        self.metrics = metrics
        self.primary_metric = primary_metric or names[0]
        if self.primary_metric not in names:
            raise ValueError(f"primary metric {self.primary_metric!r} not in {names}")

    def build_request(self, candidate: Candidate, example: Example, sample_index: int = 0) -> LLMRequest:
        rendered = candidate.prompt.render(example.inputs)
        return LLMRequest(
            model=self.target.model,
            system=rendered.system,
            messages=[Message(role="user", content=rendered.user)],
            max_tokens=self.target.max_tokens,
            params=self.target.params,
            sample_index=sample_index,
        )

    async def _run_one(self, candidate: Candidate, example: Example, sample_index: int) -> ExampleResult:
        request = self.build_request(candidate, example, sample_index)
        try:
            resp = await self.gateway.complete(request, provider=self.target.provider, role="target")
        except LLMError as e:
            return ExampleResult(
                example_id=example.id,
                sample_index=sample_index,
                inputs=example.inputs,
                expected=example.expected,
                output=None,
                error=f"{type(e).__name__}: {e}",
                scores={m.name: Score(value=0.0, rationale="generation failed") for m in self.metrics},
            )

        scores = await asyncio.gather(*(m.ascore(example, resp.text) for m in self.metrics))
        return ExampleResult(
            example_id=example.id,
            sample_index=sample_index,
            inputs=example.inputs,
            expected=example.expected,
            output=resp.text,
            stop_reason=resp.stop_reason,
            scores={m.name: s for m, s in zip(self.metrics, scores, strict=True)},
            latency_s=resp.latency_s,
            usage=resp.usage,
            cost_usd=resp.cost_usd,
            cached=resp.cached,
        )

    async def aevaluate(
        self,
        candidate: Candidate,
        dataset: Dataset,
        split: str = "unspecified",
        samples: int = 1,
    ) -> EvalReport:
        # Fail fast on template/variable mismatch before spending anything.
        for example in dataset:
            candidate.prompt.render(example.inputs)

        tasks = [
            self._run_one(candidate, example, s) for example in dataset for s in range(samples)
        ]
        results = await asyncio.gather(*tasks)
        return EvalReport(
            candidate_id=candidate.id,
            prompt_fingerprint=candidate.fingerprint,
            dataset_name=dataset.name,
            split=split,
            target_model=self.target.model,
            metric_names=[m.name for m in self.metrics],
            primary_metric=self.primary_metric,
            results=list(results),
        )

    def evaluate(self, *args, **kwargs) -> EvalReport:
        """Sync convenience wrapper around `aevaluate` (not for use inside an event loop)."""
        return asyncio.run(self.aevaluate(*args, **kwargs))
