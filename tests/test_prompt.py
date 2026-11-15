import pytest

from harness_optimizer.core import Candidate, Edit, MissingVariableError, PromptTemplate


def test_render_system_and_user_sections():
    p = PromptTemplate.from_string("Double {{ x }}. JSON: {\"a\": 1}", system="Be terse.")
    r = p.render({"x": 3})
    assert r.system == "Be terse."
    assert r.user == 'Double 3. JSON: {"a": 1}'
    assert p.variables == {"x"}


def test_missing_variable_raises():
    with pytest.raises(MissingVariableError):
        PromptTemplate.from_string("{{a}} {{b}}").render({"a": 1})


def test_with_section_is_immutable_and_changes_fingerprint():
    p = PromptTemplate.from_string("hi")
    q = p.with_section("format", "Answer in one word.")
    assert [s.name for s in p.sections] == ["task"]
    assert [s.name for s in q.sections] == ["task", "format"]
    assert p.fingerprint() != q.fingerprint()
    assert q.without_section("format").fingerprint() == p.fingerprint()


def test_duplicate_section_names_rejected():
    with pytest.raises(ValueError):
        PromptTemplate.model_validate(
            {"sections": [{"name": "a", "content": "1"}, {"name": "a", "content": "2"}]}
        )


def test_yaml_roundtrip(tmp_path):
    p = PromptTemplate.from_string("Q: {{q}}", system="sys")
    p.to_yaml(tmp_path / "p.yaml")
    assert PromptTemplate.from_yaml(tmp_path / "p.yaml") == p


def test_candidate_lineage():
    root = Candidate.root(PromptTemplate.from_string("a"))
    child = root.child(PromptTemplate.from_string("b"), Edit(operator="rewrite"), created_by="proposer")
    assert child.parent_id == root.id and child.id != root.id
