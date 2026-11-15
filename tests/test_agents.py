import asyncio
import json

import pytest

from harness_optimizer.agents import (
    AgentError,
    Analysis,
    FailureAnalyst,
    InvalidProposal,
    Proposal,
    apply_proposal,
    check_candidate,
    structured_call,
)
from harness_optimizer.agents.base import extract_json, inline_refs, load_prompt
from harness_optimizer.core import (
    Candidate,
    EvalReport,
    ExampleResult,
    Failure,
    ModelConfig,
    PromptTemplate,
    Score,
)
from harness_optimizer.llm import LLMGateway, MockLLM

MODEL = ModelConfig(provider="mock", model="mock")
PROMPT = PromptTemplate.from_string("Classify: {{text}}", system="You classify.")


def test_prompts_are_packaged():
    assert "diagnose" in load_prompt("analyst")
    assert "exactly one section" in load_prompt("proposer")


def test_inline_refs_removes_defs():
    schema = inline_refs(Analysis.model_json_schema())
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["hypotheses"]["items"]["properties"]["description"]["type"] == "string"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blah {"a": {"b": 2}} blah') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_call_repairs_invalid_reply():
    replies = iter(["not json at all", '{"hypotheses": []}'])
    client = MockLLM(lambda req: next(replies))
    gw = LLMGateway(client)
    out = asyncio.run(structured_call(gw, MODEL, "sys", "user", Analysis, role="analyst"))
    assert out.hypotheses == []
    second = client.calls[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert "invalid" in second.messages[-1].content


def test_structured_call_gives_up():
    gw = LLMGateway(MockLLM(lambda req: "nope"))
    with pytest.raises(AgentError):
        asyncio.run(structured_call(gw, MODEL, "s", "u", Analysis, role="analyst", max_attempts=2))


def test_structured_call_passes_schema_as_format_for_ollama():
    class FakeOllama(MockLLM):
        provider = "ollama"

    client = FakeOllama(lambda req: '{"hypotheses": []}')
    asyncio.run(structured_call(LLMGateway(client), ModelConfig(provider="ollama", model="m"),
                                "s", "u", Analysis, role="analyst"))
    assert client.calls[0].params["format"]["type"] == "object"


def test_apply_proposal_operators():
    p = apply_proposal(PROMPT, Proposal(rationale="r", operator="add_section", section="format", content="Label only."))
    assert [s.name for s in p.sections] == ["system", "task", "format"]
    p2 = apply_proposal(p, Proposal(rationale="r", operator="rewrite_section", section="task", content="Label: {{text}}"))
    assert p2.section("task").content == "Label: {{text}}"
    p3 = apply_proposal(p2, Proposal(rationale="r", operator="remove_section", section="format"))
    assert "format" not in {s.name for s in p3.sections}
    with pytest.raises(InvalidProposal):
        apply_proposal(PROMPT, Proposal(rationale="r", operator="rewrite_section", section="nope", content="x"))
    with pytest.raises(InvalidProposal):
        apply_proposal(PROMPT, Proposal(rationale="r", operator="add_section", section="task", content="x"))


def test_guards():
    parent = Candidate.root(PROMPT)
    lost_var = Candidate.root(PROMPT.with_section("task", "Classify the text."))
    assert any("drops template variables" in p for p in check_candidate(parent, lost_var, set()))
    ok = Candidate.root(PROMPT.with_section("format", "Label only."))
    assert check_candidate(parent, ok, set()) == []
    assert any("duplicate" in p for p in check_candidate(parent, ok, {ok.fingerprint}))


def test_analyst_drops_invented_example_ids():
    reply = {"hypotheses": [{"description": "d", "example_ids": ["e1", "made-up"], "suggested_fix": "f"}]}
    gw = LLMGateway(MockLLM(lambda req: json.dumps(reply)))
    analyst = FailureAnalyst(gw, MODEL)
    failure = Failure(example_id="e1", inputs={"text": "x"}, expected="a", output="b", error=None, score=0, rationale=None)
    report = EvalReport(
        candidate_id="c", prompt_fingerprint="f", dataset_name="d", split="train", target_model="m",
        metric_names=["m"], primary_metric="m",
        results=[ExampleResult(example_id="e1", inputs={}, output="b", scores={"m": Score(value=0)})],
    )
    analysis = asyncio.run(analyst.analyze(Candidate.root(PROMPT), report, [failure]))
    assert analysis.hypotheses[0].example_ids == ["e1"]


def test_render_sections_lists_placeholders():
    from harness_optimizer.agents.base import render_sections

    text = render_sections(PROMPT)
    assert "[section: task | role: user | placeholders: {{text}}]" in text
    assert "[section: system | role: system]" in text


def test_proposer_repairs_dropped_placeholder():
    from harness_optimizer.agents import Hypothesis, Proposer

    replies = iter([
        json.dumps({"rationale": "r", "operator": "rewrite_section", "section": "task",
                    "content": "Classify the review."}),
        json.dumps({"rationale": "r", "operator": "rewrite_section", "section": "task",
                    "content": "Classify the review: {{text}}"}),
    ])
    client = MockLLM(lambda req: next(replies))
    proposer = Proposer(LLMGateway(client), MODEL)
    hyp = Hypothesis(description="d", example_ids=[], suggested_fix="f")
    _, child = asyncio.run(proposer.propose(Candidate.root(PROMPT), hyp, []))
    assert child.prompt.section("task").content == "Classify the review: {{text}}"
    assert "removes the placeholder(s) {{text}}" in client.calls[1].messages[-1].content


# ---- milestone 4: critic, reflector, lessons, copy guard ------------------------------------

def _child(parent: Candidate, content: str, section: str = "output_format") -> Candidate:
    from harness_optimizer.core import Edit

    return parent.child(
        parent.prompt.with_section(section, content),
        Edit(operator="add_section", target_section=section, rationale="r", hypothesis="h"),
        created_by="proposer",
    )


def test_new_prompts_are_packaged():
    assert "reviewer" in load_prompt("critic")
    assert "lessons" in load_prompt("reflector")


def test_critic_rejection_needs_issues_and_is_repaired():
    from harness_optimizer.agents import Critic

    replies = iter(['{"verdict": "reject", "issues": []}',
                    '{"verdict": "reject", "issues": ["hard-codes an answer"]}'])
    client = MockLLM(lambda req: next(replies))
    parent = Candidate.root(PROMPT)
    child = _child(parent, "Answer only with the label.")
    out = asyncio.run(Critic(LLMGateway(client), MODEL).review(parent, child))
    assert out.verdict == "reject" and out.issues == ["hard-codes an answer"]
    assert "at least one issue" in client.calls[1].messages[-1].content
    # the critic sees the edited section before/after and the hypothesis
    assert "after:\nAnswer only with the label." in client.calls[0].messages[0].content


def test_reflector_drops_duplicate_lessons():
    from harness_optimizer.agents import Reflector
    from harness_optimizer.agents.pipeline import CandidateOutcome, RoundRecord
    from harness_optimizer.memory import Lesson

    reply = json.dumps({"lessons": ["Old lesson.", "New lesson.", "new lesson.", "  "]})
    client = MockLLM(lambda req: reply)
    record = RoundRecord(round=1, incumbent_id="x", incumbent_train=0.5, incumbent_val=0.5,
                         outcomes=[CandidateOutcome(hypothesis="h", operator="add_section", section="s",
                                                    train_score=0.4, rejected=[])])
    lessons = asyncio.run(Reflector(LLMGateway(client), MODEL, max_lessons=4).reflect(
        record, {}, [Lesson(round=0, text="old lesson.")]))
    assert [lesson.text for lesson in lessons] == ["New lesson."]
    assert lessons[0].round == 1
    assert "(round 0) old lesson." in client.calls[0].messages[0].content


def test_guard_blocks_verbatim_example_copies():
    parent = Candidate.root(PROMPT)
    review = "The battery died after two days and support never answered"
    child = _child(parent, f"If the review says '{review}', answer negative.")
    assert any("verbatim" in p for p in check_candidate(parent, child, set(), example_snippets=[review]))
    short = _child(parent, "Answer negative for 'bad'.")
    assert check_candidate(parent, short, set(), example_snippets=["bad"]) == []


def test_store_saves_and_loads_lessons(tmp_path):
    from harness_optimizer.memory import ExperimentStore, Lesson

    store = ExperimentStore()
    store.add_candidate(Candidate.root(PROMPT))
    store.add_lesson(Lesson(round=0, text="a"))
    store.add_lesson(Lesson(round=1, text="b"))
    assert [x.text for x in store.recent_lessons(1)] == ["b"]
    loaded = ExperimentStore.load(store.save(tmp_path))
    assert [x.text for x in loaded.lessons] == ["a", "b"]
