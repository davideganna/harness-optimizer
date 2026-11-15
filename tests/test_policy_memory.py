from harness_optimizer.core import Candidate, Edit, PromptTemplate
from harness_optimizer.eval.stats import PairedComparison
from harness_optimizer.llm import CostLedger
from harness_optimizer.memory import ExperimentStore
from harness_optimizer.policy import Budget, BudgetTracker
from harness_optimizer.search import PairedImprovement


def test_budget_rounds_and_patience():
    t = BudgetTracker(Budget(max_rounds=5, patience=2), CostLedger())
    assert t.stop_reason() is None
    t.end_round(improved=False)
    t.end_round(improved=False)
    assert "no accepted candidate" in t.stop_reason()
    t2 = BudgetTracker(Budget(max_rounds=1), CostLedger())
    t2.end_round(improved=True)
    assert "max_rounds" in t2.stop_reason()


def test_budget_call_cap_counts_only_uncached():
    ledger = CostLedger()
    s = ledger.stats("target")
    s.calls, s.cache_hits = 10, 6
    t = BudgetTracker(Budget(max_llm_calls=5), ledger)
    assert t.stop_reason() is None
    s.calls = 11
    assert "call budget" in t.stop_reason()


def test_acceptance_rule():
    rule = PairedImprovement(alpha=0.1, min_delta=0.0)
    base = {"n": 10, "mean_a": 0.5, "mean_b": 0.7, "low": 0.1, "high": 0.3, "confidence": 0.95}
    assert rule.accepts(PairedComparison(mean_diff=0.2, p_value=0.01, **base))
    assert not rule.accepts(PairedComparison(mean_diff=0.2, p_value=0.3, **base))
    assert not rule.accepts(PairedComparison(mean_diff=0.0, p_value=0.0, **base))


def test_store_lineage_and_roundtrip(tmp_path):
    store = ExperimentStore()
    root = Candidate.root(PromptTemplate.from_string("a"))
    child = root.child(PromptTemplate.from_string("b"), Edit(operator="rewrite_section"), "proposer")
    grandchild = child.child(PromptTemplate.from_string("c"), Edit(operator="add_section"), "proposer")
    for c in (root, child, grandchild):
        store.add_candidate(c)
    assert [c.id for c in store.lineage(grandchild.id)] == [root.id, child.id, grandchild.id]
    store.save(tmp_path)
    loaded = ExperimentStore.load(tmp_path)
    assert [c.id for c in loaded.lineage(grandchild.id)] == [root.id, child.id, grandchild.id]
