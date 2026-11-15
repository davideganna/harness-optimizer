"""Offline tests of the Ollama adapter using a fake client (no server needed)."""

import asyncio
from types import SimpleNamespace

import ollama
import pytest

from harness_optimizer.core import ModelConfig
from harness_optimizer.llm import (
    LLMError,
    LLMGateway,
    LLMRequest,
    Message,
    OllamaClient,
    RetryableLLMError,
)


class FakeOllama:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.kwargs = None

    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return self.behaviour()


def ok():
    return SimpleNamespace(
        model="qwen3:4b", done_reason="stop", prompt_eval_count=12, eval_count=3,
        message=SimpleNamespace(content="positive", thinking=None),
    )


REQ = LLMRequest(model="qwen3:4b", system="sys", messages=[Message(role="user", content="hi")],
                 max_tokens=32, sample_index=2)


def test_maps_request_and_response():
    fake = FakeOllama(ok)
    resp = asyncio.run(OllamaClient(client=fake).complete(REQ))
    assert resp.text == "positive" and resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 12 and resp.usage.output_tokens == 3
    kw = fake.kwargs
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["options"] == {"num_predict": 32, "num_ctx": 4096}
    assert kw["think"] is False and kw["stream"] is False


def test_params_passthrough_and_validation():
    fake = FakeOllama(ok)
    req = REQ.model_copy(update={"params": {"think": True, "options": {"seed": 1}, "format": "json"}})
    asyncio.run(OllamaClient(client=fake).complete(req))
    assert fake.kwargs["think"] is True and fake.kwargs["format"] == "json"
    assert fake.kwargs["options"]["seed"] == 1
    bad = REQ.model_copy(update={"params": {"output_config": {"effort": "low"}}})
    with pytest.raises(LLMError):
        asyncio.run(OllamaClient(client=fake).complete(bad))


@pytest.mark.parametrize(
    "exc, expected, text",
    [
        (lambda: ollama.ResponseError("model not found", 404), LLMError, "ollama pull"),
        (lambda: ollama.ResponseError("overloaded", 503), RetryableLLMError, "503"),
        (lambda: ollama.ResponseError("llama runner process has terminated: exit status 0xc0000005", 500),
         LLMError, "could not load"),
        (lambda: ConnectionError("refused"), LLMError, "Ollama app running"),
    ],
)
def test_error_mapping(exc, expected, text):
    def raise_():
        raise exc()

    with pytest.raises(expected, match=text) as info:
        asyncio.run(OllamaClient(client=FakeOllama(raise_)).complete(REQ))
    if expected is LLMError:
        assert not isinstance(info.value, RetryableLLMError)


def test_gateway_reports_zero_cost_for_ollama():
    gw = LLMGateway(OllamaClient(client=FakeOllama(ok)))
    resp = asyncio.run(gw.complete(REQ))
    assert resp.cost_usd == 0.0 and gw.ledger.total_cost_usd == 0.0


def test_default_model_config_is_ollama():
    cfg = ModelConfig()
    assert cfg.provider == "ollama" and cfg.model == "llama3.2:3b"


def test_new_client_per_event_loop():
    # Regression: the sync evaluate() wrapper runs each call in a fresh event loop.
    c = OllamaClient()

    async def grab():
        return c._get_client()

    first, second = asyncio.run(grab()), asyncio.run(grab())
    assert first is not second
