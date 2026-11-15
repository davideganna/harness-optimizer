"""Deterministic offline client for tests, demos and harness development."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from harness_optimizer.llm.base import LLMRequest, LLMResponse, RetryableLLMError, Usage

Responder = Callable[[LLMRequest], str]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLM:
    """Returns `responder(request)`; optionally fails the first `fail_first` calls.

    `fail_first` lets tests exercise the gateway's retry logic.
    """

    provider = "mock"

    def __init__(
        self,
        responder: Responder | None = None,
        latency_s: float = 0.0,
        fail_first: int = 0,
    ) -> None:
        self.responder = responder or (lambda req: "mock response")
        self.latency_s = latency_s
        self._fail_remaining = fail_first
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RetryableLLMError("simulated transient failure")
        text = self.responder(request)
        prompt_text = (request.system or "") + "".join(m.content for m in request.messages)
        return LLMResponse(
            text=text,
            model=request.model,
            usage=Usage(input_tokens=_approx_tokens(prompt_text), output_tokens=_approx_tokens(text)),
            stop_reason="end_turn",
            latency_s=self.latency_s,
        )
