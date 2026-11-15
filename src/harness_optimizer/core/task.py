"""Task-level configuration: which model is being prompted and with what settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_MODELS = {"ollama": DEFAULT_MODEL, "anthropic": "claude-opus-5-5", "mock": "mock"}
# Optimizer agents need stronger reasoning than the target but make few calls per round.
# On a 4 GB GPU qwen3:4b proposed working edits where llama3.2:3b did not.
DEFAULT_AGENT_MODELS = {"ollama": "qwen3:4b", "anthropic": "claude-opus-5-5", "mock": "mock"}


class ModelConfig(BaseModel):
    """The model a prompt is evaluated against (the *target*), or that an agent/judge uses.

    Defaults to a free local Ollama model; use ``provider="anthropic"`` for Claude.

    `params` is passed through to the provider adapter untouched, e.g.
    ``{"output_config": {"effort": "low"}}`` for Anthropic models.
    """

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    params: dict[str, Any] = Field(default_factory=dict)
