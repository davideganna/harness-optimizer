"""Pipeline mode: a fixed, deterministic optimization loop (the baseline for agent mode).

One round:
  1. evaluate the incumbent on a train minibatch and collect failures
  2. Failure Analyst  -> ranked hypotheses             (agents see train failures only)
  3. Proposer         -> one-section edits per hypothesis
  4. guards           -> drop structurally broken, duplicate or example-copying candidates (free)
     critic           -> optional LLM review that rejects clearly bad edits (1 call/candidate)
  5. screen           -> on train: either the same minibatch for everyone ("minibatch"), or
                         successive halving over growing train slices ("halving")
  6. validate         -> the top-k on val; accept the best one if it significantly beats the
                         incumbent (paired bootstrap). Agents never see val examples.
  7. reflect          -> optional Reflector turns the round into lessons for later rounds
After the loop, baseline and best are each evaluated on test exactly once.

Every optional component has a config switch, so its effect can be measured by ablation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from harness_optimizer.agents.analyst import FailureAnalyst, Hypothesis
from harness_optimizer.agents.base import AgentError
from harness_optimizer.agents.critic import Critic
from harness_optimizer.agents.guards import check_candidate
from harness_optimizer.agents.proposer import InvalidProposal, Proposer
from harness_optimizer.agents.reflector import Reflector
from harness_optimizer.core.candidate import Candidate
from harness_optimizer.core.dataset import Dataset, DatasetSplits
from harness_optimizer.core.prompt import PromptTemplate
from harness_optimizer.core.results import EvalReport, Failure
from harness_optimizer.core.task import DEFAULT_AGENT_MODELS, DEFAULT_PROVIDER, ModelConfig
from harness_optimizer.eval.metrics.base import Metric
from harness_optimizer.eval.runner import EvalRunner
from harness_optimizer.eval.stats import PairedComparison, paired_bootstrap
from harness_optimizer.llm.base import LLMError
from harness_optimizer.llm.gateway import LLMGateway
from harness_optimizer.memory.lessons import render_lessons
from harness_optimizer.memory.store import ExperimentStore
from harness_optimizer.policy.budget import Budget, BudgetTracker
from harness_optimizer.search.acceptance import PairedImprovement
from harness_optimizer.search.halving import SuccessiveHalving

log = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    minibatch_size: int = 12
    max_failures_shown: int = 6
    max_hypotheses: int = 2
    proposals_per_hypothesis: int = 2
    val_candidates: int = 2  # how many screened candidates get a (more expensive) val eval
    seed: int = 0
    screen: Literal["minibatch", "halving"] = "minibatch"
    halving_rungs: tuple[int, ...] = (4, 8, 16)
    halving_eta: int = 2
    use_critic: bool = False
    use_reflector: bool = True
    max_lessons_shown: int = 6
    block_example_copies: bool = True


class CandidateOutcome(BaseModel):
    candidate_id: str | None = None
    hypothesis: str
    operator: str | None = None
    section: str | None = None
    rationale: str | None = None
    rejected: list[str] = Field(default_factory=list)
    train_score: float | None = None
    train_n: int | None = None  # how many train examples train_score is based on
    val_score: float | None = None
    val_diff: float | None = None
    p_value: float | None = None
    accepted: bool = False


class RoundRecord(BaseModel):
    round: int
    incumbent_id: str
    incumbent_train: float
    incumbent_val: float
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    outcomes: list[CandidateOutcome] = Field(default_factory=list)
    accepted_id: str | None = None
    lessons: list[str] = Field(default_factory=list)
    note: str | None = None


class OptimizationResult(BaseModel):
    baseline_id: str
    best_id: str
    best_prompt: PromptTemplate
    rounds: list[RoundRecord]
    stop_reason: str
    val: dict[str, float]
    test: dict[str, float]
    test_comparison: PairedComparison | None
    llm_ledger: dict

    def summary_text(self) -> str:
        lines = [
            f"stop reason: {self.stop_reason}",
            f"rounds: {len(self.rounds)}, accepted: {sum(r.accepted_id is not None for r in self.rounds)}",
            f"val  baseline {self.val['baseline']:.3f} -> best {self.val['best']:.3f}",
            f"test baseline {self.test['baseline']:.3f} -> best {self.test['best']:.3f}",
        ]
        if self.test_comparison:
            c = self.test_comparison
            lines.append(
                f"test diff {c.mean_diff:+.3f} CI [{c.low:+.3f}, {c.high:+.3f}] "
                f"p={c.p_value:.3f} (n={c.n})"
            )
        return "\n".join(lines)


class OptimizationPipeline:
    def __init__(
        self,
        gateway: LLMGateway,
        target: ModelConfig,
        metrics: list[Metric],
        splits: DatasetSplits,
        agent_model: ModelConfig | None = None,
        config: PipelineConfig | None = None,
        budget: Budget | None = None,
        acceptance: PairedImprovement | None = None,
        store: ExperimentStore | None = None,
        on_round: Callable[[RoundRecord], None] | None = None,
    ) -> None:
        self.gateway = gateway
        self.splits = splits
        self.config = config or PipelineConfig()
        self.budget = budget or Budget()
        self.acceptance = acceptance or PairedImprovement()
        self.store = store or ExperimentStore()
        self.on_round = on_round
        self.runner = EvalRunner(gateway, target, metrics)
        agent_model = agent_model or ModelConfig(
            provider=DEFAULT_PROVIDER, model=DEFAULT_AGENT_MODELS[DEFAULT_PROVIDER]
        )
        self.analyst = FailureAnalyst(gateway, agent_model, self.config.max_hypotheses)
        self.proposer = Proposer(gateway, agent_model)
        self.critic = Critic(gateway, agent_model) if self.config.use_critic else None
        self.reflector = Reflector(gateway, agent_model) if self.config.use_reflector else None

    # ---- helpers ------------------------------------------------------------
    async def _eval(self, candidate: Candidate, dataset: Dataset, split: str) -> EvalReport:
        report = await self.runner.aevaluate(candidate, dataset, split=split)
        self.store.add_report(report)
        return report

    def _note(self, record: RoundRecord, text: str) -> None:
        record.note = f"{record.note}; {text}" if record.note else text

    def _lessons_text(self) -> str:
        return render_lessons(self.store.recent_lessons(self.config.max_lessons_shown))

    async def _propose(
        self,
        incumbent: Candidate,
        hypothesis: Hypothesis,
        failures: list[Failure],
        sample_index: int,
        lessons: str,
    ) -> tuple[CandidateOutcome, Candidate | None]:
        outcome = CandidateOutcome(hypothesis=hypothesis.description)
        try:
            proposal, child = await self.proposer.propose(
                incumbent, hypothesis, failures, sample_index, lessons=lessons
            )
        except (AgentError, InvalidProposal, LLMError) as e:
            outcome.rejected.append(f"proposal failed: {e}")
            return outcome, None
        outcome.candidate_id = child.id
        outcome.operator = proposal.operator
        outcome.section = proposal.section
        outcome.rationale = proposal.rationale
        return outcome, child

    async def _critique(self, parent: Candidate, child: Candidate, outcome: CandidateOutcome, i: int) -> bool:
        """True if the candidate passes the critic (or the critic is off / failed)."""
        if self.critic is None:
            return True
        try:
            verdict = await self.critic.review(parent, child, sample_index=i)
        except (AgentError, LLMError) as e:
            log.warning("critic failed, letting the candidate through: %s", e)
            return True
        if verdict.verdict == "reject":
            outcome.rejected.extend(f"critic: {issue}" for issue in verdict.issues)
            return False
        return True

    async def _screen(
        self,
        incumbent: Candidate,
        inc_train: EvalReport,
        screen_set: Dataset,
        survivors: list[tuple[CandidateOutcome, Candidate]],
        r: int,
    ) -> list[tuple[CandidateOutcome, Candidate]]:
        """Score survivors on train; return the promising ones (beat the incumbent), best first."""
        cfg = self.config
        promising = []
        if cfg.screen == "halving":
            sh = SuccessiveHalving(rungs=cfg.halving_rungs, eta=cfg.halving_eta, seed=cfg.seed + r)
            res = await sh.run(
                incumbent, [c for _, c in survivors], self.splits.train,
                lambda c, d: self._eval(c, d, "train"),
            )
            for outcome, child in survivors:
                outcome.train_score = res.score(child.id)
                outcome.train_n = res.reached.get(child.id)
            by_id = {c.id: (o, c) for o, c in survivors}
            promising = [by_id[cid] for cid in res.promising]
            return promising
        screened = await asyncio.gather(*(self._eval(c, screen_set, "train") for _, c in survivors))
        for (outcome, child), rep in zip(survivors, screened, strict=True):
            outcome.train_score = rep.mean()
            outcome.train_n = len(rep.per_example())
            if outcome.train_score > inc_train.mean():
                promising.append((outcome, child))
        promising.sort(key=lambda oc: oc[0].train_score or 0.0, reverse=True)
        return promising

    async def _reflect(self, record: RoundRecord) -> None:
        if self.reflector is None:
            return
        edits = {}
        for o in record.outcomes:
            child = self.store.candidates.get(o.candidate_id or "")
            if child is not None:
                content = {s.name: s.content for s in child.prompt.sections}
                edits[child.id] = content.get(o.section or "", "(section removed)")
        try:
            lessons = await self.reflector.reflect(
                record, edits, self.store.recent_lessons(self.config.max_lessons_shown),
                sample_index=record.round,
            )
        except (AgentError, LLMError) as e:
            self._note(record, f"reflector failed: {e}")
            return
        for lesson in lessons:
            self.store.add_lesson(lesson)
        record.lessons = [lesson.text for lesson in lessons]

    def _emit(self, record: RoundRecord) -> None:
        self.gateway.tracer.emit({"type": "round", **record.model_dump()})
        accepted = record.accepted_id or "none"
        log.info(
            "round %d: incumbent train=%.3f val=%.3f | %d hypotheses, %d candidates, accepted=%s%s",
            record.round, record.incumbent_train, record.incumbent_val, len(record.hypotheses),
            len(record.outcomes), accepted, f" ({record.note})" if record.note else "",
        )
        if self.on_round:
            self.on_round(record)

    # ---- main loop ----------------------------------------------------------
    async def arun(self, prompt: PromptTemplate) -> OptimizationResult:
        cfg = self.config
        baseline = Candidate.root(prompt)
        self.store.add_candidate(baseline)
        incumbent = baseline
        incumbent_val = await self._eval(incumbent, self.splits.val, "val")
        baseline_val = incumbent_val
        seen = {baseline.fingerprint}
        tracker = BudgetTracker(self.budget, self.gateway.ledger)
        rounds: list[RoundRecord] = []
        r = 0

        while True:
            stop = tracker.stop_reason()
            if stop:
                break
            if incumbent_val.mean() >= 1.0:
                stop = "incumbent is already perfect on val"
                break

            # 1. failures on a train minibatch (fall back to the full train split)
            screen_set = self.splits.train.sample(cfg.minibatch_size, seed=cfg.seed + r)
            inc_train = await self._eval(incumbent, screen_set, "train")
            failures = inc_train.failures(k=cfg.max_failures_shown)
            if not failures and len(screen_set) < len(self.splits.train):
                screen_set = self.splits.train
                inc_train = await self._eval(incumbent, screen_set, "train")
                failures = inc_train.failures(k=cfg.max_failures_shown)
            if not failures:
                stop = "no train failures left"
                break

            record = RoundRecord(
                round=r,
                incumbent_id=incumbent.id,
                incumbent_train=inc_train.mean(),
                incumbent_val=incumbent_val.mean(),
            )

            # 2. hypotheses
            lessons = self._lessons_text()
            try:
                analysis = await self.analyst.analyze(
                    incumbent, inc_train, failures, sample_index=r, lessons=lessons
                )
            except (AgentError, LLMError) as e:
                self._note(record, f"analyst failed: {e}")
                rounds.append(record)
                self._emit(record)
                tracker.end_round(improved=False)
                r += 1
                continue
            record.hypotheses = analysis.hypotheses
            if not analysis.hypotheses:
                record.note = "analyst returned no hypotheses"

            # 3. proposals (distinct sample_index per job -> distinct samples, distinct cache keys)
            jobs = [h for h in analysis.hypotheses for _ in range(cfg.proposals_per_hypothesis)]
            proposed = await asyncio.gather(
                *(
                    self._propose(incumbent, h, failures, sample_index=r * 1000 + i, lessons=lessons)
                    for i, h in enumerate(jobs)
                )
            )

            # 4. guards (free), then the optional critic (one agent call per candidate)
            snippets = (
                [str(v) for f in failures for v in f.inputs.values()] if cfg.block_example_copies else None
            )
            guarded: list[tuple[CandidateOutcome, Candidate]] = []
            for outcome, child in proposed:
                record.outcomes.append(outcome)
                if child is None:
                    continue
                problems = check_candidate(incumbent, child, seen, example_snippets=snippets)
                if problems:
                    outcome.rejected.extend(problems)
                    continue
                seen.add(child.fingerprint)
                self.store.add_candidate(child)
                guarded.append((outcome, child))
            verdicts = await asyncio.gather(
                *(self._critique(incumbent, c, o, r * 1000 + i) for i, (o, c) in enumerate(guarded))
            )
            survivors = [oc for oc, ok in zip(guarded, verdicts, strict=True) if ok]

            # 5. screen on train
            promising = await self._screen(incumbent, inc_train, screen_set, survivors, r)

            # 6. validate the top-k and accept the best significant improvement
            best: tuple[Candidate, EvalReport, CandidateOutcome] | None = None
            for outcome, child in promising[: cfg.val_candidates]:
                val_rep = await self._eval(child, self.splits.val, "val")
                cmp = self.acceptance.compare(incumbent_val, val_rep)
                outcome.val_score = val_rep.mean()
                outcome.val_diff = cmp.mean_diff
                outcome.p_value = cmp.p_value
                if self.acceptance.accepts(cmp) and (best is None or val_rep.mean() > best[1].mean()):
                    best = (child, val_rep, outcome)

            if best is not None:
                incumbent, incumbent_val, outcome = best
                outcome.accepted = True
                record.accepted_id = incumbent.id
            elif not record.note:
                record.note = "no candidate significantly beat the incumbent on val"

            # 7. lessons for later rounds
            await self._reflect(record)

            rounds.append(record)
            self._emit(record)
            tracker.end_round(improved=best is not None)
            r += 1

        # Final, one-time test evaluation.
        test_base = await self._eval(baseline, self.splits.test, "test")
        if incumbent is baseline:
            test_best, test_cmp = test_base, None
        else:
            test_best = await self._eval(incumbent, self.splits.test, "test")
            test_cmp = paired_bootstrap(test_base.per_example(), test_best.per_example())

        return OptimizationResult(
            baseline_id=baseline.id,
            best_id=incumbent.id,
            best_prompt=incumbent.prompt,
            rounds=rounds,
            stop_reason=stop,
            val={"baseline": baseline_val.mean(), "best": incumbent_val.mean()},
            test={"baseline": test_base.mean(), "best": test_best.mean()},
            test_comparison=test_cmp,
            llm_ledger={k: v.model_dump() for k, v in self.gateway.ledger.by_role.items()},
        )

    def run(self, prompt: PromptTemplate) -> OptimizationResult:
        return asyncio.run(self.arun(prompt))
