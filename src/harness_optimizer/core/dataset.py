"""Examples, datasets and train/val/test splits.

Split discipline is a first-class concern:
  * train - the optimizer may read individual failures here
  * val   - the optimizer only ever sees aggregate scores (used for selection)
  * test  - never reachable from agent tools; evaluated once for the final report
The agent tool layer (not yet built) is responsible for enforcing those visibility rules;
this module just makes the splits explicit and reproducible.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


class Example(BaseModel):
    id: str
    inputs: dict[str, Any]
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str = "dataset"
    examples: list[Example]

    @model_validator(mode="after")
    def _unique_ids(self) -> Dataset:
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("example ids must be unique")
        return self

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):  # type: ignore[override]
        return iter(self.examples)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> Dataset:
        """Each line: {"id": ..., "inputs": {...}, "expected": ..., "metadata": {...}}.

        `id` is optional (defaults to the line number).
        """
        path = Path(path)
        examples = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("id", str(i))
                row["id"] = str(row["id"])
                examples.append(Example.model_validate(row))
        return cls(name=name or path.stem, examples=examples)

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(e.model_dump_json() + "\n" for e in self.examples)

    def subset(self, ids: list[str], name: str | None = None) -> Dataset:
        by_id = {e.id: e for e in self.examples}
        return Dataset(name=name or self.name, examples=[by_id[i] for i in ids])

    def sample(self, n: int, seed: int = 0) -> Dataset:
        """Random minibatch (used for cheap evaluations before full ones)."""
        if n >= len(self):
            return self
        rng = random.Random(seed)
        return Dataset(name=f"{self.name}[{n}@{seed}]", examples=rng.sample(self.examples, n))

    def split(
        self,
        train: float = 0.5,
        val: float = 0.25,
        test: float = 0.25,
        seed: int = 0,
        stratify_by: str | None = None,
    ) -> DatasetSplits:
        """Deterministic split. `stratify_by` names a metadata key (or 'expected')."""
        if abs(train + val + test - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        rng = random.Random(seed)

        groups: dict[Any, list[Example]] = defaultdict(list)
        for e in self.examples:
            if stratify_by is None:
                key = None
            elif stratify_by == "expected":
                key = json.dumps(e.expected, sort_keys=True, default=str)
            else:
                key = e.metadata.get(stratify_by)
            groups[key].append(e)

        parts: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
        for key in sorted(groups, key=str):
            items = groups[key][:]
            rng.shuffle(items)
            n_train = round(len(items) * train)
            n_val = round(len(items) * val)
            parts["train"] += items[:n_train]
            parts["val"] += items[n_train : n_train + n_val]
            parts["test"] += items[n_train + n_val :]

        return DatasetSplits(
            train=Dataset(name=f"{self.name}/train", examples=parts["train"]),
            val=Dataset(name=f"{self.name}/val", examples=parts["val"]),
            test=Dataset(name=f"{self.name}/test", examples=parts["test"]),
        )


class DatasetSplits(BaseModel):
    train: Dataset
    val: Dataset
    test: Dataset

    def get(self, split: SplitName | str) -> Dataset:
        split = SplitName(split)
        if split is SplitName.ALL:
            return Dataset(
                name="all", examples=self.train.examples + self.val.examples + self.test.examples
            )
        return getattr(self, split.value)
