"""Deterministic metrics: cheap, reproducible, and hard for an optimizer to game."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from harness_optimizer.core.dataset import Example
from harness_optimizer.core.results import Score
from harness_optimizer.eval.metrics.base import Extractor, Metric


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .!?\"'`")


class ExactMatch(Metric):
    name = "exact_match"

    def __init__(self, normalize: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.normalize = normalize

    def score(self, example: Example, output: str) -> Score:
        got, want = self.prepare(output), str(example.expected)
        if self.normalize:
            got, want = normalize_text(got), normalize_text(want)
        ok = got == want
        return Score(value=float(ok), rationale=None if ok else f"expected {want!r}, got {got!r}")


class Contains(Metric):
    """Passes if the expected string (or any of a list of strings) appears in the output."""

    name = "contains"

    def score(self, example: Example, output: str) -> Score:
        got = normalize_text(self.prepare(output))
        wants = example.expected if isinstance(example.expected, list) else [example.expected]
        ok = any(normalize_text(str(w)) in got for w in wants)
        return Score(value=float(ok), rationale=None if ok else f"none of {wants!r} in output")


class RegexMatch(Metric):
    name = "regex"

    def __init__(self, pattern: str, flags: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.pattern = re.compile(pattern, flags)

    def score(self, example: Example, output: str) -> Score:
        ok = self.pattern.search(self.prepare(output)) is not None
        return Score(value=float(ok), rationale=None if ok else f"no match for /{self.pattern.pattern}/")


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


class JsonValid(Metric):
    """Output parses as JSON and, if a Pydantic `schema` is given, validates against it."""

    name = "json_valid"

    def __init__(self, schema: type[BaseModel] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.schema = schema

    def score(self, example: Example, output: str) -> Score:
        text = _strip_code_fence(self.prepare(output)).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return Score(value=0.0, rationale=f"invalid JSON: {e}")
        if self.schema is not None:
            try:
                self.schema.model_validate(data)
            except ValidationError as e:
                return Score(value=0.0, rationale=f"schema violation: {e.errors()[:3]}")
        return Score(value=1.0)


class NumericMatch(Metric):
    """Extracts the last number in the output and compares it with tolerance."""

    name = "numeric"
    _NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

    def __init__(self, rel_tol: float = 1e-6, abs_tol: float = 1e-9, **kw: Any) -> None:
        super().__init__(**kw)
        self.rel_tol, self.abs_tol = rel_tol, abs_tol

    def score(self, example: Example, output: str) -> Score:
        nums = self._NUM.findall(self.prepare(output))
        if not nums:
            return Score(value=0.0, rationale="no number found in output")
        got = float(nums[-1].replace(",", ""))
        want = float(example.expected)
        ok = abs(got - want) <= max(self.abs_tol, self.rel_tol * abs(want))
        return Score(value=float(ok), rationale=None if ok else f"expected {want}, got {got}")


class FunctionMetric(Metric):
    """Wrap any `fn(example, output) -> float | bool | Score`."""

    def __init__(
        self,
        fn: Callable[[Example, str], float | bool | Score],
        name: str,
        extract: Extractor | None = None,
    ) -> None:
        super().__init__(name=name, extract=extract)
        self.fn = fn

    def score(self, example: Example, output: str) -> Score:
        result = self.fn(example, self.prepare(output))
        if isinstance(result, Score):
            return result
        return Score(value=float(result))
