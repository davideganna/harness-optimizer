"""Harness Optimizer - agentic prompt optimization built on a rigorous evaluation harness."""

from harness_optimizer.core import (
    Candidate,
    Dataset,
    Edit,
    EvalReport,
    Example,
    ModelConfig,
    PromptTemplate,
    Score,
)
from harness_optimizer.eval import EvalRunner, bootstrap_ci, paired_bootstrap
from harness_optimizer.llm import LLMGateway, MockLLM, SQLiteCache

__version__ = "0.1.0.dev0"

__all__ = [
    "Candidate",
    "Dataset",
    "Edit",
    "EvalReport",
    "EvalRunner",
    "Example",
    "LLMGateway",
    "MockLLM",
    "ModelConfig",
    "PromptTemplate",
    "SQLiteCache",
    "Score",
    "bootstrap_ci",
    "paired_bootstrap",
]
