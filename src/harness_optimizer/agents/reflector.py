"""Reflector: turns one round's outcomes into short, reusable lessons.

Without memory, every round starts from scratch and the Proposer can retry an edit that
already failed. The Reflector writes a few general lessons after each round; the pipeline
stores them and shows the most recent ones to the Analyst and the Proposer.

It sees train scores and val *aggregates* (means, differences, p-values), never val examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from harness_optimizer.agents.base import load_prompt, structured_call
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.llm.gateway import LLMGateway
from harness_optimizer.memory.lessons import Lesson, render_lessons

if TYPE_CHECKING:
    from harness_optimizer.agents.pipeline import RoundRecord


class Reflection(BaseModel):
    lessons: list[str] = Field(description="new general lessons, at most a few; empty if none")


def render_round(record: RoundRecord, edits: dict[str, str]) -> str:
    """Readable account of a round. `edits` maps candidate id -> the edited section's new text."""
    lines = [
        f"Current prompt: train score {record.incumbent_train:.2f}, val score {record.incumbent_val:.2f}",
        "",
        "Hypotheses tested:",
    ]
    lines += [f"- {h.description}" for h in record.hypotheses] or ["- (none)"]
    lines += ["", "Edits:"]
    for o in record.outcomes:
        head = f"- {o.operator or '?'} [{o.section or '?'}] for hypothesis: {o.hypothesis}"
        lines.append(head)
        if o.candidate_id and o.candidate_id in edits:
            lines.append(f"  new text: {edits[o.candidate_id][:300]!r}")
        if o.rejected:
            lines.append(f"  rejected before measurement: {'; '.join(o.rejected)[:300]}")
            continue
        if o.train_score is not None:
            lines.append(f"  train score {o.train_score:.2f} (current prompt {record.incumbent_train:.2f})")
        if o.val_score is not None:
            lines.append(f"  val score {o.val_score:.2f}, diff {o.val_diff:+.2f}, p={o.p_value:.2f}")
        lines.append("  ACCEPTED" if o.accepted else "  not accepted")
    if not record.outcomes:
        lines.append("- (none)")
    if record.note:
        lines += ["", f"Note: {record.note}"]
    return "\n".join(lines)


class Reflector:
    role = "reflector"

    def __init__(self, gateway: LLMGateway, model: ModelConfig, max_lessons: int = 2) -> None:
        self.gateway = gateway
        self.model = model
        self.max_lessons = max_lessons
        self.system = load_prompt("reflector")

    def build_user_message(self, record: RoundRecord, edits: dict[str, str], earlier: list[Lesson]) -> str:
        return (
            f"## This round\n{render_round(record, edits)}\n\n"
            f"## Earlier lessons\n{render_lessons(earlier) or '(none)'}\n\n"
            f"Return at most {self.max_lessons} new lessons."
        )

    async def reflect(
        self, record: RoundRecord, edits: dict[str, str], earlier: list[Lesson], sample_index: int = 0
    ) -> list[Lesson]:
        reflection = await structured_call(
            self.gateway,
            self.model,
            self.system,
            self.build_user_message(record, edits, earlier),
            Reflection,
            role=self.role,
            sample_index=sample_index,
        )
        known = {lesson.text.strip().lower() for lesson in earlier}
        out = []
        for text in reflection.lessons[: self.max_lessons]:
            text = text.strip()
            if text and text.lower() not in known:
                known.add(text.lower())
                out.append(Lesson(round=record.round, text=text))
        return out
