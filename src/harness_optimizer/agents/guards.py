"""Deterministic candidate guards: a free, instant pre-screen before spending eval budget.

These catch structurally broken candidates, plus the most blatant form of overfitting: an edit
that pastes a failed example's input verbatim (usually to hard-code its answer). The optional
LLM Critic (critic.py) judges subtler semantic problems.

Leakage note: the Proposer only ever sees train examples, so it cannot leak val/test
answers into a prompt. Deliberate few-shot demonstrations are a separate (planned) operator;
the copy check below targets free-form edits.
"""

from __future__ import annotations

from harness_optimizer.core.candidate import Candidate


def check_candidate(
    parent: Candidate,
    child: Candidate,
    seen_fingerprints: set[str],
    max_chars: int = 8000,
    example_snippets: list[str] | None = None,
    min_snippet_chars: int = 25,
) -> list[str]:
    """Return the reasons to reject `child` (an empty list means it passes)."""
    problems = []
    if child.prompt.variables != parent.prompt.variables:
        missing = parent.prompt.variables - child.prompt.variables
        extra = child.prompt.variables - parent.prompt.variables
        if missing:
            problems.append(f"drops template variables {sorted(missing)}")
        if extra:
            problems.append(f"introduces unknown variables {sorted(extra)}")
    if child.fingerprint in seen_fingerprints:
        problems.append("duplicate of an already-evaluated prompt")
    size = sum(len(s.content) for s in child.prompt.sections)
    if size > max_chars:
        problems.append(f"prompt too long ({size} > {max_chars} chars)")
    if example_snippets:
        old = "\n".join(s.content for s in parent.prompt.sections).lower()
        new = "\n".join(s.content for s in child.prompt.sections).lower()
        for snippet in example_snippets:
            snip = " ".join(snippet.split()).lower()
            if len(snip) >= min_snippet_chars and snip in " ".join(new.split()) and snip not in " ".join(old.split()):
                problems.append(f"copies a training example's input verbatim ({snippet[:40]!r}...)")
                break
    return problems
