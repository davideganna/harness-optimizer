"""Candidates: versioned prompts that form a lineage tree.

Every prompt the optimizer ever considers becomes a Candidate with a pointer to its parent
and a record of the edit that produced it. The tree is the main research artifact: it
answers "which kinds of edits actually helped?".
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from harness_optimizer.core.prompt import PromptTemplate


class Edit(BaseModel):
    operator: str = Field(description="e.g. 'rewrite_section', 'add_fewshot', 'add_constraint'")
    rationale: str = ""
    target_section: str | None = None
    hypothesis: str | None = Field(default=None, description="failure hypothesis it addresses")


class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: PromptTemplate
    parent_id: str | None = None
    edit: Edit | None = None
    created_by: str = "user"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def fingerprint(self) -> str:
        return self.prompt.fingerprint()

    @classmethod
    def root(cls, prompt: PromptTemplate) -> Candidate:
        return cls(prompt=prompt, created_by="user")

    def child(self, prompt: PromptTemplate, edit: Edit, created_by: str) -> Candidate:
        return Candidate(prompt=prompt, parent_id=self.id, edit=edit, created_by=created_by)
