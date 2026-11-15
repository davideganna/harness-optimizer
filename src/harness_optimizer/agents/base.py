"""Shared machinery for optimizer agents.

The key primitive is `structured_call`: ask a model for JSON matching a Pydantic schema,
validate it, and on failure feed the validation error back to the model for one repair
attempt. This "generate, validate, repair" loop is the basic building block of reliable
agentic pipelines, and it matters most with small local models.

For Ollama models the schema is also passed as `format`, which turns on constrained
decoding: the model can only emit tokens that keep the output schema-valid.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from importlib import resources
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from harness_optimizer.core.prompt import PromptTemplate
from harness_optimizer.core.task import ModelConfig
from harness_optimizer.llm.base import LLMRequest, Message
from harness_optimizer.llm.gateway import LLMGateway

T = TypeVar("T", bound=BaseModel)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class AgentError(RuntimeError):
    """An agent could not produce a valid result after its repair attempts."""


def load_prompt(name: str) -> str:
    """Load an agent system prompt from `agents/prompts/<name>.md` (versioned with the code)."""
    return resources.files("harness_optimizer.agents.prompts").joinpath(f"{name}.md").read_text(
        encoding="utf-8"
    )


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` so constrained decoders that lack $ref support can use it."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                return resolve(copy.deepcopy(target))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in `text` (tolerates code fences and surrounding prose)."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    m = _JSON_OBJ.search(candidate)
    if not m:
        raise ValueError("no JSON object found in the reply")
    return json.loads(m.group(0))


def render_sections(prompt: PromptTemplate) -> str:
    """Human/LLM-readable view of a prompt's sections, listing each section's placeholders."""
    parts = []
    for s in prompt.sections:
        variables = PromptTemplate(sections=[s]).variables
        var_note = " | placeholders: " + ", ".join(f"{{{{{v}}}}}" for v in sorted(variables)) if variables else ""
        parts.append(f"[section: {s.name} | role: {s.role}{var_note}]\n{s.content}")
    return "\n\n".join(parts)


async def structured_call(
    gateway: LLMGateway,
    model: ModelConfig,
    system: str,
    user: str,
    schema: type[T],
    role: str,
    sample_index: int = 0,
    max_attempts: int = 2,
    validate: Callable[[T], list[str]] | None = None,
) -> T:
    """`validate` adds semantic checks on top of the schema: it returns a list of problems,
    which are fed back to the model exactly like schema errors."""
    json_schema = inline_refs(schema.model_json_schema())
    system_full = (
        f"{system}\n\nRespond with a single JSON object matching this JSON schema, and nothing "
        f"else:\n{json.dumps(json_schema)}"
    )
    params = dict(model.params)
    if model.provider == "ollama":
        params["format"] = json_schema

    messages = [Message(role="user", content=user)]
    last_error = ""
    for attempt in range(max_attempts):
        request = LLMRequest(
            model=model.model,
            system=system_full,
            messages=messages,
            max_tokens=model.max_tokens,
            params=params,
            sample_index=sample_index,
        )
        resp = await gateway.complete(request, provider=model.provider, role=role)
        try:
            result = schema.model_validate(extract_json(resp.text))
            problems = validate(result) if validate else []
            if not problems:
                return result
            last_error = "; ".join(problems)
        except (ValueError, ValidationError) as e:  # json.JSONDecodeError is a ValueError
            last_error = str(e)[:500]
        if attempt < max_attempts - 1:
            gateway.tracer.emit({"type": "agent_repair", "role": role, "error": last_error})
            messages = messages + [
                Message(role="assistant", content=resp.text),
                Message(
                    role="user",
                    content=f"That reply was invalid: {last_error}\n"
                    "Fix the problem and reply again with only a JSON object that matches the schema.",
                ),
            ]
    raise AgentError(f"{role}: no valid {schema.__name__} after {max_attempts} attempts: {last_error}")
