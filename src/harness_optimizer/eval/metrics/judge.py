"""LLM-as-judge metric (rubric-based, pointwise).

Known failure modes this module should eventually guard against (research backlog):
  * verbosity / position / self-preference bias
  * the optimizer learning to please the judge instead of solving the task
  * judge drift across model versions
Mitigations available now: multiple samples (`n_samples`) averaged, an explicit rubric,
and recording the judge's rationale so a human can audit it. Next steps: pairwise judging
with position swap, and agreement checks against a small human-labelled set.
"""

from __future__ import annotations

import asyncio
import json
import re

from harness_optimizer.core.dataset import Example
from harness_optimizer.core.results import Score
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.eval.metrics.base import Extractor, Metric
from harness_optimizer.llm.base import LLMError, LLMRequest, Message
from harness_optimizer.llm.gateway import LLMGateway

JUDGE_SYSTEM = """You are a strict, impartial evaluator. You grade a model's output against a \
rubric. Judge only what the rubric asks for; do not reward length or style unless the rubric \
does. Respond with a single JSON object and nothing else."""

JUDGE_TEMPLATE = """## Rubric
{rubric}

## Task input
{inputs}

## Reference answer
{expected}

## Output to grade
{output}

Grade the output on an integer scale from {lo} to {hi}.
Respond with JSON: {{"reasoning": "<one or two sentences>", "score": <integer {lo}-{hi}>}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudge(Metric):
    name = "judge"

    def __init__(
        self,
        gateway: LLMGateway,
        model: ModelConfig,
        rubric: str,
        scale: tuple[int, int] = (1, 5),
        n_samples: int = 1,
        name: str | None = None,
        extract: Extractor | None = None,
    ) -> None:
        super().__init__(name=name, extract=extract)
        self.gateway = gateway
        self.model = model
        self.rubric = rubric
        self.lo, self.hi = scale
        self.n_samples = n_samples

    def _request(self, example: Example, output: str, sample_index: int) -> LLMRequest:
        prompt = JUDGE_TEMPLATE.format(
            rubric=self.rubric,
            inputs=json.dumps(example.inputs, ensure_ascii=False, indent=2),
            expected="(none)" if example.expected is None else example.expected,
            output=output,
            lo=self.lo,
            hi=self.hi,
        )
        return LLMRequest(
            model=self.model.model,
            system=JUDGE_SYSTEM,
            messages=[Message(role="user", content=prompt)],
            max_tokens=self.model.max_tokens,
            params=self.model.params,
            sample_index=sample_index,
        )

    def _parse(self, text: str) -> tuple[float, str]:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError("no JSON object in judge response")
        data = json.loads(m.group(0))
        raw = int(data["score"])
        if not self.lo <= raw <= self.hi:
            raise ValueError(f"score {raw} outside {self.lo}-{self.hi}")
        return (raw - self.lo) / (self.hi - self.lo), str(data.get("reasoning", ""))

    async def _one(self, example: Example, output: str, i: int) -> tuple[float, str] | str:
        try:
            resp = await self.gateway.complete(
                self._request(example, output, i), provider=self.model.provider, role="judge"
            )
            return self._parse(resp.text)
        except (LLMError, ValueError, KeyError, json.JSONDecodeError) as e:
            return f"{type(e).__name__}: {e}"

    async def ascore(self, example: Example, output: str) -> Score:
        output = self.prepare(output)
        results = await asyncio.gather(*(self._one(example, output, i) for i in range(self.n_samples)))
        ok = [r for r in results if isinstance(r, tuple)]
        errors = [r for r in results if isinstance(r, str)]
        if not ok:
            return Score(value=0.0, rationale="judge failed", details={"errors": errors})
        value = sum(v for v, _ in ok) / len(ok)
        return Score(
            value=value,
            rationale=ok[0][1],
            details={"samples": [v for v, _ in ok], "errors": errors},
        )
