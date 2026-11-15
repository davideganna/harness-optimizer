"""Critic: an LLM review of a candidate edit before it is measured.

The deterministic guards catch structural problems for free; the Critic catches semantic ones
(edits that hard-code example answers, contradict another section, change the task, or are too
vague to matter). It costs one agent call per candidate, so it is optional: whether it saves
more eval budget than it costs is exactly the kind of question an ablation should answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness_optimizer.agents.base import load_prompt, render_sections, structured_call
from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.llm.gateway import LLMGateway


class Critique(BaseModel):
    verdict: Literal["approve", "reject"]
    issues: list[str] = Field(default_factory=list, description="one sentence per problem found")


def critique_problems(c: Critique) -> list[str]:
    if c.verdict == "reject" and not [i for i in c.issues if i.strip()]:
        return ["a rejection must list at least one issue"]
    return []


def describe_edit(parent: Candidate, child: Candidate) -> str:
    """The changed section before and after, as the Critic sees it."""
    edit = child.edit
    name = edit.target_section if edit else None
    before = {s.name: s.content for s in parent.prompt.sections}
    after = {s.name: s.content for s in child.prompt.sections}
    op = edit.operator if edit else "unknown"
    lines = [f"operator: {op}", f"section: {name}"]
    if name in before:
        lines.append(f"before:\n{before[name]}")
    lines.append(f"after:\n{after[name]}" if name in after else "after: (section removed)")
    return "\n".join(lines)


class Critic:
    role = "critic"

    def __init__(self, gateway: LLMGateway, model: ModelConfig) -> None:
        self.gateway = gateway
        self.model = model
        self.system = load_prompt("critic")

    def build_user_message(self, parent: Candidate, child: Candidate) -> str:
        hypothesis = child.edit.hypothesis if child.edit else ""
        rationale = child.edit.rationale if child.edit else ""
        return (
            f"## Prompt before the edit\n{render_sections(parent.prompt)}\n\n"
            f"## Hypothesis being fixed\n{hypothesis}\n\n"
            f"## The edit\n{describe_edit(parent, child)}\n\n"
            f"## Editor's rationale\n{rationale}"
        )

    async def review(self, parent: Candidate, child: Candidate, sample_index: int = 0) -> Critique:
        return await structured_call(
            self.gateway,
            self.model,
            self.system,
            self.build_user_message(parent, child),
            Critique,
            role=self.role,
            sample_index=sample_index,
            validate=critique_problems,
        )
