import asyncio

from harness_optimizer.core import (
    Candidate,
    Dataset,
    Edit,
    EvalReport,
    Example,
    ExampleResult,
    PromptTemplate,
    Score,
)
from harness_optimizer.search import SuccessiveHalving

DATA = Dataset(name="d", examples=[Example(id=f"e{i}", inputs={"x": str(i)}, expected="y") for i in range(20)])


def make_eval(quality: dict[str, float], calls: list):
    """Candidate with quality q gets example i right iff (i * 37 % 100) / 100 < q."""

    async def evaluate(c: Candidate, d: Dataset) -> EvalReport:
        calls.append((c.id, len(d)))
        q = quality[c.prompt.sections[0].content]
        results = [
            ExampleResult(example_id=e.id, inputs=e.inputs, output="",
                          scores={"m": Score(value=1.0 if (int(e.id[1:]) * 37 % 100) / 100 < q else 0.0)})
            for e in d.examples
        ]
        return EvalReport(candidate_id=c.id, prompt_fingerprint=c.fingerprint, dataset_name=d.name,
                          split="train", target_model="mock", metric_names=["m"],
                          primary_metric="m", results=results)

    return evaluate


def cand(text: str, parent: Candidate | None = None) -> Candidate:
    p = PromptTemplate.from_string(text)
    return Candidate.root(p) if parent is None else parent.child(p, Edit(operator="rewrite_section", rationale=text), created_by="test")


def test_slices_are_nested_prefixes_and_capped():
    sh = SuccessiveHalving(rungs=(4, 8, 50), seed=3)
    slices = sh.slices(DATA)
    assert [len(s) for s in slices] == [4, 8, 20]
    assert [e.id for e in slices[1].examples[:4]] == [e.id for e in slices[0].examples]


def test_halving_promotes_the_best_and_spends_less():
    inc = cand("inc")
    cands = [cand(t, inc) for t in ["a", "b", "c", "d"]]
    quality = {"inc": 0.3, "a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2}
    calls: list = []
    res = asyncio.run(SuccessiveHalving(rungs=(4, 8, 20), eta=2).run(inc, cands, DATA, make_eval(quality, calls)))

    assert res.promising and res.promising[0] == cands[1].id
    assert res.reached[cands[1].id] == 20
    assert min(res.reached.values()) < 20  # weak candidates were dropped early
    # 4 + 2 + 1 candidate evals plus the incumbent on each rung
    assert len([c for c in calls if c[0] != inc.id]) == 7
    assert [r.size for r in res.rungs] == [4, 8, 20]


def test_halving_returns_nothing_when_no_one_beats_the_incumbent():
    inc = cand("inc")
    cands = [cand("a", inc), cand("b", inc)]
    res = asyncio.run(SuccessiveHalving(rungs=(4, 20)).run(inc, cands, DATA, make_eval(
        {"inc": 0.9, "a": 0.2, "b": 0.5}, [])))
    assert res.promising == []
    assert res.incumbent_score is not None
