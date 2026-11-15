import asyncio

from pydantic import BaseModel

from harness_optimizer.core import Example, ModelConfig
from harness_optimizer.eval.metrics import (
    Composite,
    Contains,
    ExactMatch,
    FunctionMetric,
    JsonValid,
    LLMJudge,
    NumericMatch,
    RegexMatch,
)
from harness_optimizer.llm import LLMGateway, MockLLM

ex = Example(id="1", inputs={}, expected="Positive")


def test_exact_match_normalizes():
    assert ExactMatch().score(ex, "  positive. ").value == 1.0
    assert ExactMatch(normalize=False).score(ex, "positive").value == 0.0
    s = ExactMatch().score(ex, "negative")
    assert s.value == 0.0 and "expected" in s.rationale


def test_extract_hook():
    m = ExactMatch(extract=lambda o: o.splitlines()[-1])
    assert m.score(ex, "thinking...\npositive").value == 1.0


def test_contains_and_regex():
    assert Contains().score(ex, "It is positive overall").value == 1.0
    assert RegexMatch(r"^\d{3}$").score(ex, "123").value == 1.0


def test_json_valid_with_schema():
    class Out(BaseModel):
        label: str

    m = JsonValid(schema=Out)
    assert m.score(ex, '```json\n{"label": "x"}\n```').value == 1.0
    assert m.score(ex, '{"nope": 1}').value == 0.0
    assert m.score(ex, "not json").value == 0.0


def test_numeric_match():
    e = Example(id="n", inputs={}, expected=1234.5)
    assert NumericMatch().score(e, "so the answer is 1,234.5").value == 1.0
    assert NumericMatch().score(e, "no idea").value == 0.0


def test_composite_gate_and_weights():
    half = FunctionMetric(lambda e, o: 0.5, name="half")
    m = Composite({ExactMatch(): 1.0, half: 1.0}, gates=[Contains()])
    s = asyncio.run(m.ascore(ex, "positive"))
    assert s.value == 0.75
    gated = asyncio.run(Composite({half: 1.0}, gates=[ExactMatch()]).ascore(ex, "negative"))
    assert gated.value == 0.0 and "gate" in gated.rationale


def test_llm_judge_parses_and_averages():
    replies = iter(['{"reasoning": "ok", "score": 5}', 'Sure! {"reasoning": "meh", "score": 3}'])
    gw = LLMGateway(MockLLM(lambda req: next(replies)))
    judge = LLMJudge(gw, ModelConfig(provider="mock", model="mock"), rubric="Is it right?", n_samples=2)
    s = asyncio.run(judge.ascore(ex, "positive"))
    assert s.value == (1.0 + 0.5) / 2
    assert gw.ledger.stats("judge").calls == 2


def test_llm_judge_handles_garbage():
    gw = LLMGateway(MockLLM(lambda req: "I refuse to output JSON"))
    judge = LLMJudge(gw, ModelConfig(provider="mock", model="mock"), rubric="r")
    s = asyncio.run(judge.ascore(ex, "x"))
    assert s.value == 0.0 and s.details["errors"]
