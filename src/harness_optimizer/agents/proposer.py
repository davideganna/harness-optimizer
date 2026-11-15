"""Proposer: turns one hypothesis into a concrete one-section prompt edit.

Edits are restricted to a small set of operators so that every candidate in the lineage
tree has an attributable, analyzable change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness_optimizer.agents.analyst import Hypothesis
from harness_optimizer.agents.base import load_prompt, render_sections, structured_call
from harness_optimizer.core.candidate import Candidate, Edit
from harness_optimizer.core.prompt import PromptTemplate
from harness_optimizer.core.results import Failure
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.llm.gateway import LLMGateway

Operator = Literal["rewrite_section", "add_section", "remove_section"]


class Proposal(BaseModel):
    rationale: str = Field(description="why this edit fixes the hypothesis")
    operator: Operator
    section: str = Field(description="name of the section to rewrite/remove, or the new section name")
    content: str = Field(default="", description="new section content (empty for remove_section)")
    role: Literal["system", "user"] = "user"


class InvalidProposal(ValueError):
    pass


def apply_proposal(prompt: PromptTemplate, p: Proposal) -> PromptTemplate:
    names = {s.name for s in prompt.sections}
    name = p.section.strip()
    if p.operator == "rewrite_section":
        if name not in names:
            raise InvalidProposal(f"cannot rewrite unknown section {name!r}; sections: {sorted(names)}")
        if not p.content.strip():
            raise InvalidProposal("rewrite_section needs non-empty content")
        return prompt.with_section(name, p.content.strip())
    if p.operator == "add_section":
        if name in names:
            raise InvalidProposal(f"section {name!r} already exists")
        if not p.content.strip():
            raise InvalidProposal("add_section needs non-empty content")
        return prompt.with_section(name, p.content.strip(), role=p.role)
    if p.operator == "remove_section":
        if name not in names:
            raise InvalidProposal(f"cannot remove unknown section {name!r}")
        if len(names) == 1:
            raise InvalidProposal("cannot remove the only section")
        return prompt.without_section(name)
    raise InvalidProposal(f"unknown operator {p.operator!r}")  # pragma: no cover


def proposal_problems(prompt: PromptTemplate, p: Proposal) -> list[str]:
    """Problems that make a proposal unusable, phrased so the model can fix them."""
    try:
        new_prompt = apply_proposal(prompt, p)
    except InvalidProposal as e:
        return [str(e)]
    problems = []
    missing = prompt.variables - new_prompt.variables
    extra = new_prompt.variables - prompt.variables
    if missing:
        names = ", ".join(f"{{{{{v}}}}}" for v in sorted(missing))
        problems.append(
            f"your edit removes the placeholder(s) {names}; the prompt must still contain them "
            "exactly once. Keep them in the rewritten content, or use add_section instead"
        )
    if extra:
        names = ", ".join(f"{{{{{v}}}}}" for v in sorted(extra))
        problems.append(f"your edit adds unknown placeholder(s) {names}; only use existing ones")
    return problems


class Proposer:
    role = "proposer"

    def __init__(self, gateway: LLMGateway, model: ModelConfig) -> None:
        self.gateway = gateway
        self.model = model
        self.system = load_prompt("proposer")

    def build_user_message(
        self, candidate: Candidate, hypothesis: Hypothesis, failures: list[Failure], lessons: str = ""
    ) -> str:
        relevant = [f for f in failures if f.example_id in set(hypothesis.example_ids)] or failures
        lesson_block = f"## Lessons from earlier rounds\n{lessons}\n\n" if lessons else ""
        return (
            f"## Current prompt\n{render_sections(candidate.prompt)}\n\n"
            f"## Hypothesis\n{hypothesis.description}\n\n"
            f"## Suggested fix\n{hypothesis.suggested_fix}\n\n"
            + lesson_block
            +             "## Failed examples\n" + "\n\n".join(f.to_text() for f in relevant[:4])
        )

    async def propose(
        self,
        candidate: Candidate,
        hypothesis: Hypothesis,
        failures: list[Failure],
        sample_index: int = 0,
        lessons: str = "",
    ) -> tuple[Proposal, Candidate]:
        """Returns the raw proposal and the child candidate. Raises InvalidProposal or AgentError."""
        proposal = await structured_call(
            self.gateway,
            self.model,
            self.system,
            self.build_user_message(candidate, hypothesis, failures, lessons),
            Proposal,
            role=self.role,
            sample_index=sample_index,
            max_attempts=3,
            validate=lambda p: proposal_problems(candidate.prompt, p),
        )
        new_prompt = apply_proposal(candidate.prompt, proposal)
        child = candidate.child(
            new_prompt,
            Edit(
                operator=proposal.operator,
                target_section=proposal.section,
                rationale=proposal.rationale,
                hypothesis=hypothesis.description,
            ),
            created_by=self.role,
        )
        return proposal, child
