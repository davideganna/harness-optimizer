import pytest

from harness_optimizer.core import Candidate, MissingVariableError, ModelConfig, PromptTemplate
from harness_optimizer.eval import EvalRunner
from harness_optimizer.eval.metrics import ExactMatch
from harness_optimizer.llm import LLMError, LLMGateway, MemoryCache, MockLLM

TARGET = ModelConfig(provider="mock", model="mock")


def doubler(request):
    x = int(request.messages[0].content.split()[-1])
    return str(x * 2) if x % 4 else "wrong"  # fails on multiples of 4


def make_runner(client=None):
    gw = LLMGateway(client or MockLLM(doubler), cache=MemoryCache())
    return EvalRunner(gw, TARGET, [ExactMatch()]), gw


def test_evaluate_scores_and_failures(tiny_dataset):
    runner, _ = make_runner()
    cand = Candidate.root(PromptTemplate.from_string("Double {{x}}"))
    report = runner.evaluate(cand, tiny_dataset, split="all")
    assert report.mean() == pytest.approx(15 / 20)
    fails = report.failures()
    assert sorted(f.example_id for f in fails) == ["0", "12", "16", "4", "8"]
    assert "wrong" in fails[0].to_text()
    ci = report.ci()
    assert ci.low <= report.mean() <= ci.high


def test_samples_aggregate_per_example(tiny_dataset):
    runner, gw = make_runner()
    cand = Candidate.root(PromptTemplate.from_string("Double {{x}}"))
    report = runner.evaluate(cand, tiny_dataset, samples=3)
    assert len(report.results) == 60
    assert len(report.per_example()) == 20
    assert gw.ledger.stats("target").cache_hits == 0


def test_errors_are_kept_and_scored_zero(tiny_dataset):
    class Flaky(MockLLM):
        async def complete(self, request):
            if request.messages[0].content.endswith(" 3"):
                raise LLMError("400 bad")
            return await super().complete(request)

    runner, _ = make_runner(Flaky(doubler))
    report = runner.evaluate(Candidate.root(PromptTemplate.from_string("Double {{x}}")), tiny_dataset)
    assert len(report.results) == 20
    assert report.error_rate == pytest.approx(1 / 20)
    assert report.mean() == pytest.approx(14 / 20)


def test_template_mismatch_fails_before_any_call(tiny_dataset):
    client = MockLLM(doubler)
    runner, _ = make_runner(client)
    with pytest.raises(MissingVariableError):
        runner.evaluate(Candidate.root(PromptTemplate.from_string("{{y}}")), tiny_dataset)
    assert client.calls == []


def test_report_save_load(tmp_path, tiny_dataset):
    runner, _ = make_runner()
    report = runner.evaluate(Candidate.root(PromptTemplate.from_string("Double {{x}}")), tiny_dataset)
    report.save(tmp_path)
    from harness_optimizer.core import EvalReport

    assert EvalReport.load(tmp_path).mean() == report.mean()
