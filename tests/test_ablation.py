import asyncio
import json

from test_pipeline import SEED_PROMPT, cli_responder, dataset

from harness_optimizer.agents import PipelineConfig
from harness_optimizer.core import ModelConfig
from harness_optimizer.eval.metrics import ExactMatch
from harness_optimizer.experiments import PRESETS, RunRow, run_ablation, summarize
from harness_optimizer.llm import LLMGateway, MemoryCache, MockLLM
from harness_optimizer.policy import Budget

MOCK = ModelConfig(provider="mock", model="mock")


def run(responder, variants, seeds, out_dir=None, calls=None):
    cache = MemoryCache()

    def make_gateway(variant, seed):
        if calls is not None:
            calls.append((variant, seed))
        return LLMGateway(MockLLM(responder), cache=cache)

    return asyncio.run(run_ablation(
        SEED_PROMPT, dataset(), variants, seeds, make_gateway, target=MOCK, metrics=[ExactMatch()],
        agent_model=MOCK, base_config=PipelineConfig(minibatch_size=8), budget=Budget(max_rounds=2),
        fractions=(0.5, 0.25, 0.25), stratify_by="expected", out_dir=out_dir,
    ))


def test_ablation_runs_every_variant_and_seed(tmp_path):
    variants = [PRESETS["base"], PRESETS["full"]]
    result = run(cli_responder, variants, [0, 1], out_dir=tmp_path)
    assert {(r.variant, r.seed) for r in result.rows} == {("base", 0), ("full", 0), ("base", 1), ("full", 1)}
    assert all(r.error is None and r.test_best == 1.0 and r.test_baseline == 0.0 for r in result.rows)
    by_name = {s.variant: s for s in result.summaries}
    assert [s.variant for s in result.summaries] == ["base", "full"]
    assert by_name["base"].test_gain_mean == 1.0 and by_name["base"].test_best_std == 0.0
    # cache hits still count as calls, so a variant is not cheaper just because it ran second
    assert all(r.target_calls > 0 and r.agent_calls > 0 for r in result.rows)
    # the full variant also pays for the critic and the reflector
    assert by_name["full"].agent_calls_mean > by_name["base"].agent_calls_mean
    assert (tmp_path / "summary.md").read_text().startswith("| variant")
    assert (tmp_path / "base-seed0" / "result.json").exists()
    assert len(json.loads((tmp_path / "summary.json").read_text())) == 2


def test_ablation_resumes_and_retries_failures(tmp_path):
    variants = [PRESETS["base"]]
    run(cli_responder, variants, [0], out_dir=tmp_path)
    # a failed run recorded earlier is retried; a finished one is skipped
    with (tmp_path / "rows.jsonl").open("a") as f:
        f.write(RunRow(variant="base", seed=1, error="boom").model_dump_json() + "\n")
    calls = []
    result = run(cli_responder, variants, [0, 1], out_dir=tmp_path, calls=calls)
    assert calls == [("base", 1)]
    assert sorted(r.seed for r in result.rows) == [0, 1] and all(r.error is None for r in result.rows)


def test_a_broken_run_is_recorded_not_raised(monkeypatch):
    from harness_optimizer.agents.pipeline import OptimizationPipeline

    real = OptimizationPipeline.arun

    async def flaky(self, prompt):
        if self.config.use_critic:
            raise RuntimeError("exploded")
        return await real(self, prompt)

    monkeypatch.setattr(OptimizationPipeline, "arun", flaky)
    result = run(cli_responder, [PRESETS["critic"], PRESETS["base"]], [0])
    rows = {r.variant: r for r in result.rows}
    assert rows["critic"].error == "RuntimeError: exploded"
    assert rows["base"].error is None and rows["base"].test_best == 1.0
    assert {s.variant: s.failed_runs for s in result.summaries} == {"critic": 1, "base": 0}


def test_summarize_mean_and_std():
    rows = [RunRow(variant="a", seed=s, test_baseline=0.0, test_best=b, accepted=1)
            for s, b in enumerate([0.5, 1.0])]
    rows.append(RunRow(variant="a", seed=9, error="x"))
    s = summarize(rows)[0]
    assert s.runs == 3 and s.failed_runs == 1
    assert s.test_best_mean == 0.75 and abs(s.test_best_std - 0.35355) < 1e-4
