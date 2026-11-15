"""Experiment store: every candidate, every evaluation report, the lineage tree and the lessons."""

from __future__ import annotations

import json
from pathlib import Path

from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.results import EvalReport
from harness_optimizer.memory.lessons import Lesson


class ExperimentStore:
    def __init__(self) -> None:
        self.candidates: dict[str, Candidate] = {}
        self.reports: list[EvalReport] = []
        self.lessons: list[Lesson] = []

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def add_report(self, report: EvalReport) -> None:
        self.reports.append(report)

    def add_lesson(self, lesson: Lesson) -> None:
        self.lessons.append(lesson)

    def recent_lessons(self, k: int) -> list[Lesson]:
        return self.lessons[-k:] if k > 0 else []

    def reports_for(self, candidate_id: str, split: str | None = None) -> list[EvalReport]:
        return [
            r for r in self.reports
            if r.candidate_id == candidate_id and (split is None or r.split == split)
        ]

    def lineage(self, candidate_id: str) -> list[Candidate]:
        """Root-first chain of ancestors ending at `candidate_id`."""
        chain = []
        cur: str | None = candidate_id
        while cur is not None:
            c = self.candidates[cur]
            chain.append(c)
            cur = c.parent_id
        return chain[::-1]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        (directory / "reports").mkdir(parents=True, exist_ok=True)
        with (directory / "candidates.jsonl").open("w", encoding="utf-8") as f:
            for c in self.candidates.values():
                f.write(c.model_dump_json() + "\n")
        with (directory / "lessons.jsonl").open("w", encoding="utf-8") as f:
            for lesson in self.lessons:
                f.write(lesson.model_dump_json() + "\n")
        for i, r in enumerate(self.reports):
            name = f"{i:04d}_{r.candidate_id}_{r.split}.json"
            (directory / "reports" / name).write_text(r.model_dump_json(), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ExperimentStore:
        directory = Path(directory)
        store = cls()
        for line in (directory / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                store.add_candidate(Candidate.model_validate(json.loads(line)))
        for path in sorted((directory / "reports").glob("*.json")):
            store.add_report(EvalReport.model_validate_json(path.read_text(encoding="utf-8")))
        lessons = directory / "lessons.jsonl"
        if lessons.exists():
            for line in lessons.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    store.add_lesson(Lesson.model_validate_json(line))
        return store
