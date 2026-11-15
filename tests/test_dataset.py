import json

from harness_optimizer.core import Dataset


def test_split_is_deterministic_disjoint_and_complete(tiny_dataset):
    a = tiny_dataset.split(0.5, 0.25, 0.25, seed=1)
    b = tiny_dataset.split(0.5, 0.25, 0.25, seed=1)
    ids = lambda d: [e.id for e in d]
    assert ids(a.train) == ids(b.train)
    all_ids = ids(a.train) + ids(a.val) + ids(a.test)
    assert sorted(all_ids) == sorted(ids(tiny_dataset))
    assert len(set(all_ids)) == len(all_ids)
    assert (len(a.train), len(a.val), len(a.test)) == (10, 5, 5)


def test_stratified_split_balances_groups(tiny_dataset):
    s = tiny_dataset.split(0.5, 0.25, 0.25, seed=0, stratify_by="parity")
    parities = [e.metadata["parity"] for e in s.train]
    assert parities.count(0) == parities.count(1) == 5


def test_from_jsonl_defaults_ids(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps({"inputs": {"q": i}, "expected": i}) for i in range(3)) + "\n"
    )
    d = Dataset.from_jsonl(path)
    assert [e.id for e in d] == ["0", "1", "2"]
    assert d.name == "d"


def test_sample(tiny_dataset):
    assert len(tiny_dataset.sample(5, seed=0)) == 5
    assert [e.id for e in tiny_dataset.sample(5, seed=0)] == [e.id for e in tiny_dataset.sample(5, seed=0)]
