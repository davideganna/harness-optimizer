"""Experiments: running the optimizer many times to answer research questions.

  * ablation.py - run pipeline variants (components on/off) over several seeds and summarize
"""

from harness_optimizer.experiments.ablation import (
    PRESETS,
    AblationResult,
    RunRow,
    Variant,
    VariantSummary,
    run_ablation,
    summarize,
)

__all__ = ["PRESETS", "AblationResult", "RunRow", "Variant", "VariantSummary", "run_ablation", "summarize"]
