"""Block E - search strategy.

Decides which candidates survive and get expanded, independently of how they are proposed.

Implemented:
  * acceptance.py - paired-bootstrap acceptance rule (challenger vs incumbent on val)
  * halving.py    - successive halving: cheap small-slice evals first, larger evals only for
                    the promising candidates

Planned: beam / evolutionary search, Pareto selection over score / prompt length / cost.
"""

from harness_optimizer.search.acceptance import PairedImprovement
from harness_optimizer.search.halving import HalvingResult, Rung, SuccessiveHalving

__all__ = ["HalvingResult", "PairedImprovement", "Rung", "SuccessiveHalving"]
