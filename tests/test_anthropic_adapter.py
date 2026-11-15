"""Offline test of the Anthropic adapter using a fake SDK client (no network)."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from harness_optimizer.llm import LLMError, LLMRequest, Message, RetryableLLMError
from harness_optimizer.llm.anthropic_client import AnthropicClient


class FakeMessages:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.behaviour()


def fake_client(behaviour):
    return SimpleNamespace(messages=FakeMessages(behaviour))


def ok_response():
    return SimpleNamespace(
        id="msg_1",
        model="claude-opus-5-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="positive")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2, cache_read_input_tokens=None,
                              cache_creation_input_tokens=0),
    )


REQ = LLMRequest(model="claude-opus-5-5", system="sys", messages=[Message(role="user", content="hi")],
                 max_tokens=64, params={"output_config": {"effort": "low"}}, sample_index=3)


def test_maps_request_and_response():
    fc = fake_client(ok_response)
    resp = asyncio.run(AnthropicClient(client=fc).complete(REQ))
    assert resp.text == "positive" and resp.usage.input_tokens == 10 and resp.usage.cache_read_tokens == 0
    kw = fc.messages.kwargs
    assert kw["system"] == "sys" and kw["output_config"] == {"effort": "low"}
    assert "sample_index" not in kw and "temperature" not in kw


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(code, request=req), body=None)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (lambda: _status_error(anthropic.RateLimitError, 429), RetryableLLMError),
        (lambda: _status_error(anthropic.InternalServerError, 500), RetryableLLMError),
        (lambda: _status_error(anthropic.BadRequestError, 400), LLMError),
    ],
)
def test_error_mapping(exc, expected):
    def raise_():
        raise exc()

    with pytest.raises(expected) as info:
        asyncio.run(AnthropicClient(client=fake_client(raise_)).complete(REQ))
    if expected is LLMError:
        assert not isinstance(info.value, RetryableLLMError)
