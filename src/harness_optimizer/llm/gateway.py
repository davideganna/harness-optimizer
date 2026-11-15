"""The single choke point for every LLM call in the system.

Responsibilities: provider routing, response caching, bounded concurrency, retries with
exponential backoff, cost/token accounting per *role* (target, judge, proposer, ...), and
emitting a trace event per call.
"""

from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel, Field

from harness_optimizer.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    RetryableLLMError,
    Usage,
)
from harness_optimizer.llm.cache import NullCache, ResponseCache
from harness_optimizer.llm.pricing import cost_usd
from harness_optimizer.obs.tracing import NullTracer, Tracer


class RoleStats(BaseModel):
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    errors: int = 0
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0  # actual spend (cache hits cost nothing)


class CostLedger(BaseModel):
    by_role: dict[str, RoleStats] = Field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.by_role.values())

    def stats(self, role: str) -> RoleStats:
        return self.by_role.setdefault(role, RoleStats())


class LLMGateway:
    def __init__(
        self,
        clients: LLMClient | list[LLMClient] | dict[str, LLMClient],
        cache: ResponseCache | None = None,
        tracer: Tracer | None = None,
        max_concurrency: int = 8,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        if isinstance(clients, dict):
            self.clients = dict(clients)
        elif isinstance(clients, list):
            self.clients = {c.provider: c for c in clients}
        else:
            self.clients = {clients.provider: clients}
        self.cache = cache or NullCache()
        self.tracer = tracer or NullTracer()
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self.ledger = CostLedger()

    def _sem(self) -> asyncio.Semaphore:
        # A semaphore is bound to the event loop it is used on; recreate per loop so the
        # gateway survives multiple `asyncio.run` calls (e.g. sync wrappers, tests).
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    def _client(self, provider: str | None) -> LLMClient:
        if provider is None:
            if len(self.clients) != 1:
                raise ValueError(f"provider required; configured: {sorted(self.clients)}")
            return next(iter(self.clients.values()))
        try:
            return self.clients[provider]
        except KeyError:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(self.clients)}")

    async def complete(
        self, request: LLMRequest, *, provider: str | None = None, role: str = "target"
    ) -> LLMResponse:
        client = self._client(provider)
        key = request.cache_key(client.provider)
        stats = self.ledger.stats(role)
        stats.calls += 1

        cached = self.cache.get(key)
        if cached is not None:
            stats.cache_hits += 1
            resp = cached.model_copy(update={"cached": True, "cost_usd": 0.0})
            self._trace(role, client.provider, key, request, resp, cost=0.0)
            return resp

        attempt = 0
        async with self._sem():
            while True:
                try:
                    resp = await client.complete(request)
                    break
                except RetryableLLMError as e:
                    if attempt >= self.max_retries:
                        stats.errors += 1
                        self._trace_error(role, client.provider, key, request, e)
                        raise
                    attempt += 1
                    stats.retries += 1
                    delay = min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1))
                    await asyncio.sleep(delay * (0.5 + random.random() / 2))  # jitter
                except LLMError as e:
                    stats.errors += 1
                    self._trace_error(role, client.provider, key, request, e)
                    raise

        cost = cost_usd(request.model, resp.usage, provider=client.provider)
        resp = resp.model_copy(update={"cost_usd": cost})
        stats.usage = stats.usage + resp.usage
        stats.cost_usd += cost
        self.cache.set(key, resp)
        self._trace(role, client.provider, key, request, resp, cost=cost)
        return resp

    # ---- tracing --------------------------------------------------------
    def _trace(self, role, provider, key, request, resp: LLMResponse, cost: float) -> None:
        self.tracer.emit(
            {
                "type": "llm_call",
                "role": role,
                "provider": provider,
                "model": request.model,
                "key": key,
                "cached": resp.cached,
                "latency_s": round(resp.latency_s, 4),
                "usage": resp.usage.model_dump(),
                "cost_usd": cost,
                "stop_reason": resp.stop_reason,
                "request": request.model_dump(),
                "response_text": resp.text,
            }
        )

    def _trace_error(self, role, provider, key, request, err: Exception) -> None:
        self.tracer.emit(
            {
                "type": "llm_error",
                "role": role,
                "provider": provider,
                "model": request.model,
                "key": key,
                "error": f"{type(err).__name__}: {err}",
            }
        )
