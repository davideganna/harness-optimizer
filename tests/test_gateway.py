import asyncio

import pytest

from harness_optimizer.llm import (
    LLMError,
    LLMGateway,
    LLMRequest,
    MemoryCache,
    Message,
    MockLLM,
    RetryableLLMError,
    SQLiteCache,
)
from harness_optimizer.obs import MemoryTracer


def req(text="hi", sample_index=0):
    return LLMRequest(model="mock", messages=[Message(role="user", content=text)], sample_index=sample_index)


def test_cache_hit_skips_client_and_is_free():
    client = MockLLM()
    gw = LLMGateway(client, cache=MemoryCache())
    asyncio.run(gw.complete(req()))
    second = asyncio.run(gw.complete(req()))
    assert second.cached and len(client.calls) == 1
    stats = gw.ledger.stats("target")
    assert stats.calls == 2 and stats.cache_hits == 1


def test_sample_index_busts_cache():
    client = MockLLM()
    gw = LLMGateway(client, cache=MemoryCache())
    asyncio.run(gw.complete(req(sample_index=0)))
    asyncio.run(gw.complete(req(sample_index=1)))
    assert len(client.calls) == 2


def test_retries_transient_errors():
    client = MockLLM(fail_first=2)
    gw = LLMGateway(client, backoff_base_s=0.001)
    resp = asyncio.run(gw.complete(req()))
    assert resp.text == "mock response"
    assert gw.ledger.stats("target").retries == 2


def test_gives_up_after_max_retries():
    gw = LLMGateway(MockLLM(fail_first=10), max_retries=2, backoff_base_s=0.001)
    with pytest.raises(RetryableLLMError):
        asyncio.run(gw.complete(req()))
    assert gw.ledger.stats("target").errors == 1


def test_non_retryable_error_is_not_retried():
    class Bad(MockLLM):
        async def complete(self, request):
            self.calls.append(request)
            raise LLMError("400 bad request")

    client = Bad()
    gw = LLMGateway(client)
    with pytest.raises(LLMError):
        asyncio.run(gw.complete(req()))
    assert len(client.calls) == 1


def test_tracer_records_calls():
    tracer = MemoryTracer()
    gw = LLMGateway(MockLLM(), tracer=tracer)
    asyncio.run(gw.complete(req(), role="judge"))
    assert tracer.events[0]["type"] == "llm_call" and tracer.events[0]["role"] == "judge"


def test_sqlite_cache_persists(tmp_path):
    path = tmp_path / "c.sqlite"
    gw = LLMGateway(MockLLM(), cache=SQLiteCache(path))
    asyncio.run(gw.complete(req()))
    client2 = MockLLM()
    gw2 = LLMGateway(client2, cache=SQLiteCache(path))
    assert asyncio.run(gw2.complete(req())).cached
    assert client2.calls == []


def test_multiple_providers_require_explicit_choice():
    class Other(MockLLM):
        provider = "other"

    gw = LLMGateway([MockLLM(), Other()])
    with pytest.raises(ValueError):
        asyncio.run(gw.complete(req()))
    assert asyncio.run(gw.complete(req(), provider="other")).text == "mock response"
