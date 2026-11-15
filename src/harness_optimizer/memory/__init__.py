"""Block F - experiment memory.

  * store.py   - every Candidate and EvalReport, the lineage tree, and the lessons
  * lessons.py - short Reflector notes retrieved into the Analyst's and Proposer's context

Planned: a state summarizer (bounded-size experiment summary for agent mode's orchestrator).
"""

from harness_optimizer.memory.lessons import Lesson, render_lessons
from harness_optimizer.memory.store import ExperimentStore

__all__ = ["ExperimentStore", "Lesson", "render_lessons"]
