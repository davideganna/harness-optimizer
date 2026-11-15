"""Statistics for deciding whether a candidate is *actually* better.

Pure Python (no numpy) to keep the core dependency-light. Bootstrap resampling is done over
examples (not over samples of the same example), since examples are the independent unit.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel


class ConfidenceInterval(BaseModel):
    mean: float
    low: float
    high: float
    confidence: float
    n: int


class PairedComparison(BaseModel):
    """Result of comparing candidate B against baseline A on the same examples."""

    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    low: float
    high: float
    p_value: float  # one-sided bootstrap estimate of P(diff <= 0)
    confidence: float

    @property
    def significant(self) -> bool:
        return self.low > 0


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    n = len(values)
    if n == 0:
        return ConfidenceInterval(mean=0.0, low=0.0, high=0.0, confidence=confidence, n=0)
    mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return ConfidenceInterval(
        mean=mean,
        low=_quantile(means, alpha),
        high=_quantile(means, 1 - alpha),
        confidence=confidence,
        n=n,
    )


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedComparison:
    """Paired comparison on the examples both dicts share (keys are example ids)."""
    ids = sorted(a.keys() & b.keys())
    if not ids:
        raise ValueError("no overlapping example ids to compare")
    diffs = [b[i] - a[i] for i in ids]
    n = len(diffs)
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return PairedComparison(
        n=n,
        mean_a=sum(a[i] for i in ids) / n,
        mean_b=sum(b[i] for i in ids) / n,
        mean_diff=sum(diffs) / n,
        low=_quantile(boot, alpha),
        high=_quantile(boot, 1 - alpha),
        p_value=sum(d <= 0 for d in boot) / n_resamples,
        confidence=confidence,
    )
