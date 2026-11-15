"""Lesson memory: short, general notes the Reflector writes after each round.

Lessons are fed back to the Analyst and the Proposer, so later rounds can build on what worked
and avoid repeating edits that already failed. They are derived from train failures and val
*aggregates* only, so they cannot leak val or test examples.
"""

from __future__ import annotations

from pydantic import BaseModel


class Lesson(BaseModel):
    round: int
    text: str


def render_lessons(lessons: list[Lesson]) -> str:
    """Bullet list for an agent prompt ('' when there are none)."""
    return "\n".join(f"- (round {lesson.round}) {lesson.text}" for lesson in lessons)
