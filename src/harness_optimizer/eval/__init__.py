"""Block C - evaluation harness: runner, metrics, judges, statistics."""

from harness_optimizer.eval.runner import EvalRunner
from harness_optimizer.eval.stats import (
    ConfidenceInterval,
    PairedComparison,
    bootstrap_ci,
    paired_bootstrap,
)

__all__ = [
    "ConfidenceInterval",
    "EvalRunner",
    "PairedComparison",
    "bootstrap_ci",
    "paired_bootstrap",
]
