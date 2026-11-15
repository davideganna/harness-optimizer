"""Block D - optimization agents.

Implemented (milestone 3, pipeline mode):
  * base.py      - structured JSON calls with validate-and-repair, prompt loading
  * analyst.py   - Failure Analyst: train failures -> ranked hypotheses
  * proposer.py  - Proposer: hypothesis -> one-section prompt edit
  * guards.py    - deterministic pre-screen of candidates
  * pipeline.py  - the fixed optimization loop (the baseline for agent mode)
  * prompts/     - the agents' own system prompts, versioned as files

Implemented (milestone 4):
  * critic.py    - optional LLM review of each edit before it is measured
  * reflector.py - turns each round into lessons fed to later rounds

Planned:
  * fewshot.py      - few-shot demonstration selection as an edit operator
  * orchestrator.py - agent mode, driving the loop through tool calls (milestone 5)
"""

from harness_optimizer.agents.analyst import Analysis, FailureAnalyst, Hypothesis
from harness_optimizer.agents.base import AgentError, structured_call
from harness_optimizer.agents.critic import Critic, Critique
from harness_optimizer.agents.guards import check_candidate
from harness_optimizer.agents.pipeline import (
    CandidateOutcome,
    OptimizationPipeline,
    OptimizationResult,
    PipelineConfig,
    RoundRecord,
)
from harness_optimizer.agents.proposer import InvalidProposal, Proposal, Proposer, apply_proposal
from harness_optimizer.agents.reflector import Reflection, Reflector

__all__ = [
    "AgentError",
    "Analysis",
    "CandidateOutcome",
    "Critic",
    "Critique",
    "FailureAnalyst",
    "Hypothesis",
    "InvalidProposal",
    "OptimizationPipeline",
    "OptimizationResult",
    "PipelineConfig",
    "Proposal",
    "Proposer",
    "Reflection",
    "Reflector",
    "RoundRecord",
    "apply_proposal",
    "check_candidate",
    "structured_call",
]
