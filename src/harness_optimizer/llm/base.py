"""Provider-agnostic request/response types and the client protocol.

This module must not import from `harness_optimizer.core` (core depends on it).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 1024
    params: dict[str, Any] = Field(default_factory=dict)
    sample_index: int = Field(
        default=0,
        description="Distinguishes repeated samples of the same request so each gets its own "
        "cache entry. Not sent to the provider.",
    )

    def cache_key(self, provider: str) -> str:
        blob = json.dumps(
            {"provider": provider, **self.model_dump()}, sort_keys=True, default=str
        ).encode()
        return hashlib.sha256(blob).hexdigest()


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None
    latency_s: float = 0.0
    cached: bool = False
    cost_usd: float = 0.0  # set by the gateway (0 for cache hits and free providers)
    response_id: str | None = None


class LLMError(Exception):
    """Non-retryable provider error (bad request, auth, ...)."""


class RetryableLLMError(LLMError):
    """Transient error (rate limit, overload, network). The gateway retries these."""


@runtime_checkable
class LLMClient(Protocol):
    provider: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
