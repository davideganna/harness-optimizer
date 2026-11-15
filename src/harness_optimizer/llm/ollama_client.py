"""Ollama adapter (native API via the official `ollama` Python library).

Runs open models on the local Ollama server (default http://localhost:11434, override with
the OLLAMA_HOST env var). Local inference is free, so the gateway reports cost 0.

Design notes:
  * `think=False` by default: for thinking-capable models (qwen3, ...) this skips the hidden
    reasoning phase, which is the biggest speed lever on a small GPU. Turn it on per model via
    ``ModelConfig(params={"think": True})``.
  * `num_ctx` defaults to 4096 so small models stay fully on a 4 GB GPU; larger contexts
    spill to system RAM and get much slower.
  * Accepted `params`: think, format, keep_alive, options (merged into Ollama `options`).
    Unknown keys raise, so Anthropic-only params (e.g. output_config) are not silently ignored.
"""

from __future__ import annotations

import os
import time
from typing import Any

from harness_optimizer.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    RetryableLLMError,
    Usage,
)

_ALLOWED_PARAMS = {"think", "format", "keep_alive", "options"}
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        host: str | None = None,
        think: bool | str = False,
        num_ctx: int = 4096,
        keep_alive: str = "10m",
        timeout_s: float = 300.0,
        client: Any | None = None,
    ) -> None:
        try:
            import ollama
        except ImportError as e:  # pragma: no cover
            raise ImportError("The Ollama adapter needs: pip install ollama") from e
        self._sdk = ollama
        self.host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.think = think
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        self._injected = client
        self._loop: Any = None
        self._client: Any = None

    def _get_client(self) -> Any:
        """One AsyncClient per event loop: an httpx connection pool is bound to the loop that
        created it, and the sync `evaluate()` wrapper starts a new loop on every call."""
        if self._injected is not None:
            return self._injected
        import asyncio

        loop = asyncio.get_running_loop()
        # Compare loop objects, not id(): a closed loop's id can be reused by a new loop.
        if self._loop is not loop:
            self._loop = loop
            self._client = self._sdk.AsyncClient(host=self.host, timeout=self.timeout_s)
        return self._client

    def _build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        unknown = set(request.params) - _ALLOWED_PARAMS
        if unknown:
            raise LLMError(
                f"unsupported params for Ollama: {sorted(unknown)} (allowed: {sorted(_ALLOWED_PARAMS)})"
            )
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [m.model_dump() for m in request.messages]

        options = {"num_predict": request.max_tokens, "num_ctx": self.num_ctx}
        options.update(request.params.get("options", {}))
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "options": options,
            "think": request.params.get("think", self.think),
            "keep_alive": request.params.get("keep_alive", self.keep_alive),
            "stream": False,
        }
        if "format" in request.params:
            kwargs["format"] = request.params["format"]
        return kwargs

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_kwargs(request)
        sdk = self._sdk
        t0 = time.perf_counter()
        try:
            resp = await self._get_client().chat(**kwargs)
        except sdk.ResponseError as e:
            if e.status_code == 404:
                raise LLMError(
                    f"model {request.model!r} not found on the Ollama server; "
                    f"run: ollama pull {request.model}"
                ) from e
            if "terminated" in str(e.error) or "unable to load" in str(e.error):
                # The model runner crashed while loading (e.g. unsupported GPU driver).
                # Retrying just repeats a multi-minute failure, so fail fast.
                raise LLMError(
                    f"Ollama could not load {request.model!r}: {e.error}. "
                    "Check the Ollama server log; on older NVIDIA drivers set OLLAMA_LLM_LIBRARY=vulkan."
                ) from e
            if e.status_code == 429 or e.status_code >= 500:
                raise RetryableLLMError(f"Ollama {e.status_code}: {e.error}") from e
            raise LLMError(f"Ollama {e.status_code}: {e.error}") from e
        except ConnectionError as e:
            raise LLMError(
                f"cannot reach Ollama at {self.host}; is the Ollama app running? ({e})"
            ) from e
        except TimeoutError as e:
            raise RetryableLLMError(f"Ollama timeout: {e}") from e
        except Exception as e:
            # httpx transport errors (timeouts, resets) surface here.
            if type(e).__module__.startswith("httpx"):
                raise RetryableLLMError(f"{type(e).__name__}: {e}") from e
            raise
        latency = time.perf_counter() - t0

        return LLMResponse(
            text=resp.message.content or "",
            model=resp.model or request.model,
            usage=Usage(
                input_tokens=resp.prompt_eval_count or 0,
                output_tokens=resp.eval_count or 0,
            ),
            stop_reason=_STOP_REASONS.get(resp.done_reason or "", resp.done_reason),
            latency_s=latency,
        )
