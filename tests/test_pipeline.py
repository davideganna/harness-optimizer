"""End-to-end pipeline test with a scripted mock LLM playing target, analyst and proposer."""

import json
import re

from harness_optimizer.agents import OptimizationPipeline, PipelineConfig
from harness_optimizer.core import Dataset, Example, ModelConfig, PromptTemplate
from harness_optimizer.eval.metrics import ExactMatch
from harness_optimizer.llm import LLMGateway, MemoryCache, MockLLM
from harness_optimizer.obs import MemoryTracer
from harness_optimizer.policy import Budget

MOCK = ModelConfig(provider="mock", model="mock")
SEED_PROMPT = PromptTemplate.from_string("Is this review positive or negative?\nReview: {{text}}")


def dataset() -> Dataset:
    examples = []
    for i in range(24):
        good = i % 2 == 0
        examples.append(Example(id=f"e{i:02d}", inputs={"text": f"this is {'good' if good else 'bad'} #{i}"},
                                expected="positive" if good else "negative"))
    return Dataset(name="toy", examples=examples)


def make_responder(proposal_content: str, critic_rejects: str | None = None):
    def respond(req) -> str:
        system = req.system or ""
        if "careful reviewer of LLM prompt edits" in system:
            if critic_rejects and critic_rejects in req.messages[0].content.split("## The edit")[-1]:
                return json.dumps({"verdict": "reject", "issues": ["too vague to change behaviour"]})
            return json.dumps({"verdict": "approve", "issues": []})
        if "write short lessons" in system:
            n = req.messages[0].content.count("(round ")  # earlier lessons shown -> distinct lesson
            return json.dumps({"lessons": [f"Lesson number {n}."]})
        if "diagnose why an LLM prompt" in system:
            ids = re.findall(r"\[example (\w+)\]", req.messages[0].content)
            return json.dumps({"hypotheses": [{
                "description": "The model answers with a sentence instead of the bare label.",
                "example_ids": ids, "suggested_fix": "Ask for the label only."}]})
        if "editing exactly one section" in system:
            return json.dumps({"rationale": "pin the output format", "operator": "add_section",
                               "section": "output_format", "content": proposal_content})
        # target model
        prompt = req.messages[-1].content
        label = "positive" if "good" in prompt.split("Review:")[-1] else "negative"
        return label if "label only" in prompt.lower() else f"The answer is {label}."
    return respond


def build(responder, budget=None, **config):
    tracer = MemoryTracer()
    client = MockLLM(responder)
    gw = LLMGateway(client, cache=MemoryCache(), tracer=tracer)
    pipe = OptimizationPipeline(
        gw, target=MOCK, metrics=[ExactMatch()],
        splits=dataset().split(0.5, 0.25, 0.25, seed=0, stratify_by="expected"),
        agent_model=MOCK, config=PipelineConfig(minibatch_size=8, **config),
        budget=budget or Budget(max_rounds=3),
    )
    pipe.client = client
    return pipe, tracer


def test_pipeline_finds_and_accepts_the_fix():
    pipe, tracer = build(make_responder("Answer with the label only, in lowercase."))
    result = pipe.run(SEED_PROMPT)

    assert result.val["baseline"] == 0.0 and result.val["best"] == 1.0
    assert result.test["baseline"] == 0.0 and result.test["best"] == 1.0
    assert result.test_comparison is not None and result.test_comparison.mean_diff == 1.0
    assert result.stop_reason == "incumbent is already perfect on val"
    assert "output_format" in {s.name for s in result.best_prompt.sections}

    first = result.rounds[0]
    assert first.accepted_id == result.best_id
    # two proposals per hypothesis; the identical second one is caught by the guards
    assert sum(o.accepted for o in first.outcomes) == 1
    assert any("duplicate" in r for o in first.outcomes for r in o.rejected)

    lineage = pipe.store.lineage(result.best_id)
    assert lineage[0].id == result.baseline_id and lineage[-1].edit.operator == "add_section"
    assert any(e["type"] == "round" for e in tracer.events)
    # agents and target are accounted separately
    assert {"target", "analyst", "proposer"} <= set(result.llm_ledger)


def test_pipeline_stops_on_patience_when_nothing_helps():
    pipe, _ = build(make_responder("Be helpful."), budget=Budget(max_rounds=5, patience=2))
    result = pipe.run(SEED_PROMPT)
    assert result.best_id == result.baseline_id
    assert result.test_comparison is None
    assert "no accepted candidate for 2 rounds" in result.stop_reason
    assert all(r.accepted_id is None for r in result.rounds)


def test_test_split_is_evaluated_only_once_per_candidate():
    pipe, _ = build(make_responder("Answer with the label only, in lowercase."))
    result = pipe.run(SEED_PROMPT)
    test_reports = [r for r in pipe.store.reports if r.split == "test"]
    assert sorted(r.candidate_id for r in test_reports) == sorted({result.baseline_id, result.best_id})


def test_lessons_are_stored_and_shown_to_later_rounds():
    pipe, _ = build(make_responder("Be helpful."), budget=Budget(max_rounds=2, patience=5))
    result = pipe.run(SEED_PROMPT)
    assert all(r.lessons for r in result.rounds)
    assert len(pipe.store.lessons) == 2
    proposer_prompts = [c.messages[0].content for c in pipe.client.calls
                        if "editing exactly one section" in (c.system or "")]
    assert "Lessons from earlier rounds" not in proposer_prompts[0]
    assert "Lessons from earlier rounds" in proposer_prompts[-1]


def test_no_reflector_means_no_lessons():
    pipe, _ = build(make_responder("Be helpful."), budget=Budget(max_rounds=2, patience=5), use_reflector=False)
    result = pipe.run(SEED_PROMPT)
    assert pipe.store.lessons == [] and "reflector" not in result.llm_ledger


def test_critic_rejects_before_any_eval_is_spent():
    pipe, _ = build(make_responder("Be helpful.", critic_rejects="Be helpful."),
                    budget=Budget(max_rounds=1), use_critic=True)
    result = pipe.run(SEED_PROMPT)
    outcomes = result.rounds[0].outcomes
    assert any(r.startswith("critic:") for o in outcomes for r in o.rejected)
    assert all(o.train_score is None for o in outcomes)
    assert "critic" in result.llm_ledger


def test_halving_screen_finds_the_fix():
    pipe, _ = build(make_responder("Answer with the label only, in lowercase."),
                    screen="halving", halving_rungs=(4, 12))
    result = pipe.run(SEED_PROMPT)
    assert result.val["best"] == 1.0 and result.test["best"] == 1.0
    accepted = next(o for o in result.rounds[0].outcomes if o.accepted)
    assert accepted.train_n == 12 and accepted.train_score == 1.0


# Module-level responder for the CLI test (`--mock-responder test_pipeline:cli_responder`).
cli_responder = make_responder("Answer with the label only, in lowercase.")
