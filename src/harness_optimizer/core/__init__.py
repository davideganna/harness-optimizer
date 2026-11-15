"""Block A - core data model shared by every other block."""

from harness_optimizer.core.candidate import Candidate, Edit
from harness_optimizer.core.dataset import Dataset, DatasetSplits, Example, SplitName
from harness_optimizer.core.prompt import (
    MissingVariableError,
    PromptSection,
    PromptTemplate,
    RenderedPrompt,
)
from harness_optimizer.core.results import EvalReport, ExampleResult, Failure, Score
from harness_optimizer.core.task import ModelConfig

__all__ = [
    "Candidate",
    "Dataset",
    "DatasetSplits",
    "Edit",
    "EvalReport",
    "Example",
    "ExampleResult",
    "Failure",
    "MissingVariableError",
    "ModelConfig",
    "PromptSection",
    "PromptTemplate",
    "RenderedPrompt",
    "Score",
    "SplitName",
]
