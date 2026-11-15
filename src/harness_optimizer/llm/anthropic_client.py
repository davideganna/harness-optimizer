"""Anthropic Messages API adapter.

Design notes:
  * The SDK's own retries are disabled (`max_retries=0`) so the gateway is the single,
    observable retry layer.
  * Sampling params such as `temperature` are intentionally not exposed: current Claude
    models reject them. Control behaviour via `params`, e.g.
    ``{"output_config": {"effort": "low"}}``.
  * Server-side refusal fallbacks are deliberately NOT enabled. In an evaluation harness a
    silent switch to another model would contaminate the measurement; a refusal is
    recorded as `stop_reason="refusal"` and scored like any other output.
"""

from __future__ import annotations

import time
from typing import Any

from harness_optimizer.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    RetryableLLMError,
    Usage,
)


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, client: Any | None = None, timeout_s: float = 300.0) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The Anthropic adapter needs the optional dependency: "
                "pip install 'harness-optimizer[anthropic]'"
            ) from e
        self._sdk = anthropic
        self._client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=timeout_s)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [m.model_dump() for m in request.messages],
            **request.params,
        }
        if request.system:
            kwargs["system"] = request.system

        sdk = self._sdk
        t0 = time.perf_counter()
        try:
            resp = await self._client.messages.create(**kwargs)
        except (sdk.RateLimitError, sdk.InternalServerError, sdk.APIConnectionError) as e:
            raise RetryableLLMError(f"{type(e).__name__}: {e}") from e
        except sdk.APIStatusError as e:
            raise LLMError(f"{type(e).__name__} ({e.status_code}): {e}") from e
        latency = time.perf_counter() - t0

        text = "".join(block.text for block in resp.content if block.type == "text")
        u = resp.usage
        return LLMResponse(
            text=text,
            model=resp.model,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
            ),
            stop_reason=resp.stop_reason,
            latency_s=latency,
            response_id=resp.id,
        )
