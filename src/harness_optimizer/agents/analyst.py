"""Failure Analyst: turns failed train examples into ranked, evidence-backed hypotheses.

It only ever sees *train* failures. Val and test examples never reach an agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from harness_optimizer.agents.base import load_prompt, render_sections, structured_call
from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.results import EvalReport, Failure
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.llm.gateway import LLMGateway


class Hypothesis(BaseModel):
    description: str = Field(description="root cause of the failures, in one or two sentences")
    example_ids: list[str] = Field(description="ids of the failed examples this explains")
    suggested_fix: str = Field(description="what to change in the prompt")


class Analysis(BaseModel):
    hypotheses: list[Hypothesis]


class FailureAnalyst:
    role = "analyst"

    def __init__(self, gateway: LLMGateway, model: ModelConfig, max_hypotheses: int = 2) -> None:
        self.gateway = gateway
        self.model = model
        self.max_hypotheses = max_hypotheses
        self.system = load_prompt("analyst")

    def build_user_message(
        self, candidate: Candidate, report: EvalReport, failures: list[Failure], lessons: str = ""
    ) -> str:
        lesson_block = f"## Lessons from earlier rounds\n{lessons}\n\n" if lessons else ""
        return (
            f"## Current prompt\n{render_sections(candidate.prompt)}\n\n"
            f"## Score\n{report.primary_metric} = {report.mean():.2f} "
            f"on {len(report.per_example())} training examples\n\n"
            f"## Failed examples\n" + "\n\n".join(f.to_text() for f in failures) + "\n\n"
            + lesson_block
            + f"Return at most {self.max_hypotheses} hypotheses."
        )

    async def analyze(
        self,
        candidate: Candidate,
        report: EvalReport,
        failures: list[Failure],
        sample_index: int = 0,
        lessons: str = "",
    ) -> Analysis:
        analysis = await structured_call(
            self.gateway,
            self.model,
            self.system,
            self.build_user_message(candidate, report, failures, lessons),
            Analysis,
            role=self.role,
            sample_index=sample_index,
        )
        known = {f.example_id for f in failures}
        hyps = []
        for h in analysis.hypotheses[: self.max_hypotheses]:
            # Keep only real example ids. Small models sometimes invent them.
            hyps.append(h.model_copy(update={"example_ids": [i for i in h.example_ids if i in known]}))
        return Analysis(hypotheses=hyps)
