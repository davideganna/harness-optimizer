"""Prompt templates stored as named sections.

Prompts are kept as an ordered list of sections (role, task, constraints, output format,
examples, ...) instead of one opaque string. That lets optimizer agents make targeted,
attributable edits ("rewrite the `constraints` section") and lets analysis code measure
which sections change across a lineage.

Variables use ``{{name}}`` placeholders so literal JSON braces in prompts need no escaping.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class MissingVariableError(KeyError):
    """Raised when rendering a template without all of its variables."""


class PromptSection(BaseModel):
    name: str
    content: str
    role: Literal["system", "user"] = "user"


class RenderedPrompt(BaseModel):
    system: str | None
    user: str


class PromptTemplate(BaseModel):
    sections: list[PromptSection] = Field(min_length=1)
    separator: str = "\n\n"

    @model_validator(mode="after")
    def _unique_names(self) -> PromptTemplate:
        names = [s.name for s in self.sections]
        if len(names) != len(set(names)):
            raise ValueError(f"section names must be unique, got {names}")
        return self

    # ---- construction -------------------------------------------------
    @classmethod
    def from_string(cls, user: str, system: str | None = None) -> PromptTemplate:
        sections = []
        if system:
            sections.append(PromptSection(name="system", content=system, role="system"))
        sections.append(PromptSection(name="task", content=user, role="user"))
        return cls(sections=sections)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromptTemplate:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # ---- inspection ---------------------------------------------------
    @property
    def variables(self) -> set[str]:
        return {m for s in self.sections for m in _VAR_RE.findall(s.content)}

    def section(self, name: str) -> PromptSection:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(name)

    def fingerprint(self) -> str:
        """Content hash: identical prompts share a fingerprint regardless of lineage."""
        blob = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # ---- immutable edits (what optimizer agents will call) ------------
    def with_section(
        self, name: str, content: str, role: Literal["system", "user"] | None = None
    ) -> PromptTemplate:
        """Return a copy with section `name` replaced (or appended if missing)."""
        new = [s.model_copy() for s in self.sections]
        for i, s in enumerate(new):
            if s.name == name:
                new[i] = PromptSection(name=name, content=content, role=role or s.role)
                return self.model_copy(update={"sections": new})
        new.append(PromptSection(name=name, content=content, role=role or "user"))
        return self.model_copy(update={"sections": new})

    def without_section(self, name: str) -> PromptTemplate:
        self.section(name)  # raise if missing
        return self.model_copy(
            update={"sections": [s.model_copy() for s in self.sections if s.name != name]}
        )

    # ---- rendering ----------------------------------------------------
    def render(self, variables: dict[str, Any]) -> RenderedPrompt:
        missing = self.variables - variables.keys()
        if missing:
            raise MissingVariableError(f"missing template variables: {sorted(missing)}")

        def sub(text: str) -> str:
            return _VAR_RE.sub(lambda m: str(variables[m.group(1)]), text)

        system = [sub(s.content) for s in self.sections if s.role == "system"]
        user = [sub(s.content) for s in self.sections if s.role == "user"]
        return RenderedPrompt(
            system=self.separator.join(system) if system else None,
            user=self.separator.join(user),
        )
