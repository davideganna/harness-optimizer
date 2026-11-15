import pytest

from harness_optimizer.eval import bootstrap_ci, paired_bootstrap


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = [1.0] * 70 + [0.0] * 30
    a, b = bootstrap_ci(vals, seed=3), bootstrap_ci(vals, seed=3)
    assert a == b
    assert a.mean == pytest.approx(0.7)
    assert 0.6 < a.low < 0.7 < a.high < 0.8


def test_constant_values_have_zero_width():
    ci = bootstrap_ci([1.0] * 10)
    assert ci.low == ci.high == 1.0


def test_paired_detects_clear_improvement():
    a = {str(i): 0.0 for i in range(50)}
    b = {str(i): 1.0 if i < 30 else 0.0 for i in range(50)}
    res = paired_bootstrap(a, b)
    assert res.mean_diff == pytest.approx(0.6)
    assert res.significant and res.p_value < 0.01


def test_paired_no_difference_is_not_significant():
    a = {str(i): float(i % 2) for i in range(40)}
    res = paired_bootstrap(a, dict(a))
    assert res.mean_diff == 0 and not res.significant


def test_paired_requires_overlap():
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})
