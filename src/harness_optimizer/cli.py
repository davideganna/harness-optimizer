"""Command-line interface.

    harness-opt eval    --prompt p.yaml --data d.jsonl --metric exact_match [--split val]
    harness-opt compare runs/<a> runs/<b>
    harness-opt optimize --prompt p.yaml --data d.jsonl --rounds 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from harness_optimizer.config import load_env
from harness_optimizer.core import Candidate, Dataset, EvalReport, ModelConfig, PromptTemplate
from harness_optimizer.core.task import DEFAULT_AGENT_MODELS, DEFAULT_MODELS, DEFAULT_PROVIDER
from harness_optimizer.eval import EvalRunner, paired_bootstrap
from harness_optimizer.eval.metrics import BUILTIN
from harness_optimizer.llm import LLMGateway, MockLLM, NullCache, SQLiteCache
from harness_optimizer.obs import JsonlTracer


def _import_obj(path: str):
    module, _, attr = path.partition(":")
    if not attr:
        raise SystemExit(f"expected 'module:attribute', got {path!r}")
    return getattr(importlib.import_module(module), attr)


def _build_client(args):
    if args.provider == "ollama":
        from harness_optimizer.llm.ollama_client import OllamaClient

        return OllamaClient(num_ctx=args.num_ctx)
    if args.provider == "anthropic":
        from harness_optimizer.llm.anthropic_client import AnthropicClient

        return AnthropicClient()
    if args.provider == "mock":
        return MockLLM(_import_obj(args.mock_responder) if args.mock_responder else None)
    raise SystemExit(f"unknown provider {args.provider!r}")


def cmd_eval(args) -> int:
    prompt = PromptTemplate.from_yaml(args.prompt)
    data = Dataset.from_jsonl(args.data)
    if args.split != "all":
        data = data.split(seed=args.seed).get(args.split)
    if args.n:
        data = data.sample(args.n, seed=args.seed)

    out_dir = Path(args.out) / (args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    gateway = LLMGateway(
        _build_client(args),
        cache=NullCache() if args.no_cache else SQLiteCache(args.cache),
        tracer=JsonlTracer(out_dir / "trace.jsonl"),
        max_concurrency=args.concurrency,
    )
    metrics = [BUILTIN[name]() for name in args.metric]
    runner = EvalRunner(gateway, _target_config(args), metrics)

    report = runner.evaluate(Candidate.root(prompt), data, split=args.split, samples=args.samples)
    report.save(out_dir)

    summary = report.summary()
    summary["llm_ledger"] = {r: s.model_dump() for r, s in gateway.ledger.by_role.items()}
    print(json.dumps(summary, indent=2))
    failures = report.failures(k=args.show_failures)
    if failures:
        print(f"\n--- {len(report.failures())} failures (showing {len(failures)}) ---")
        for f in failures:
            print(f.to_text(), end="\n\n")
    print(f"saved to {out_dir}")
    return 0


def _target_config(args) -> ModelConfig:
    params = {}
    if args.provider == "ollama" and args.temperature is not None:
        params = {"options": {"temperature": args.temperature}}
    return ModelConfig(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS[args.provider],
        max_tokens=args.max_tokens,
        params=params,
    )


def cmd_optimize(args) -> int:
    from harness_optimizer.agents import OptimizationPipeline, PipelineConfig, RoundRecord
    from harness_optimizer.policy import Budget
    from harness_optimizer.search import PairedImprovement

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    prompt = PromptTemplate.from_yaml(args.prompt)
    splits = Dataset.from_jsonl(args.data).split(
        args.train, args.val, round(1 - args.train - args.val, 10), seed=args.seed,
        stratify_by=args.stratify_by,
    )
    out_dir = Path(args.out) / (args.run_name or datetime.now().strftime("opt-%Y%m%d-%H%M%S"))
    gateway = LLMGateway(
        _build_client(args),
        cache=NullCache() if args.no_cache else SQLiteCache(args.cache),
        tracer=JsonlTracer(out_dir / "trace.jsonl"),
        max_concurrency=args.concurrency,
    )
    agent_model = ModelConfig(
        provider=args.provider,
        model=args.agent_model or DEFAULT_AGENT_MODELS[args.provider],
        max_tokens=args.agent_max_tokens,
    )

    def show(record: RoundRecord) -> None:
        for h in record.hypotheses:
            print(f"   hypothesis: {h.description}")
        for o in record.outcomes:
            if o.rejected:
                status = "rejected: " + "; ".join(o.rejected)[:120]
            else:
                status = f"train={o.train_score:.2f} (n={o.train_n})"
                if o.val_score is not None:
                    status += f" val={o.val_score:.2f} diff={o.val_diff:+.2f} p={o.p_value:.2f}"
                if o.accepted:
                    status += "  <- ACCEPTED"
            print(f"   - {o.operator or '?'} [{o.section or '?'}] {status}")
        for lesson in record.lessons:
            print(f"   lesson: {lesson}")

    print(f"splits: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} | "
          f"target={args.provider}/{_target_config(args).model} agents={agent_model.model}")
    pipeline = OptimizationPipeline(
        gateway,
        target=_target_config(args),
        metrics=[BUILTIN[name]() for name in args.metric],
        splits=splits,
        agent_model=agent_model,
        config=PipelineConfig(
            minibatch_size=args.minibatch,
            max_hypotheses=args.hypotheses,
            proposals_per_hypothesis=args.proposals,
            val_candidates=args.val_candidates,
            seed=args.seed,
            screen=args.screen,
            halving_rungs=tuple(args.rungs),
            use_critic=args.critic,
            use_reflector=not args.no_reflector,
        ),
        budget=Budget(max_rounds=args.rounds, patience=args.patience, max_llm_calls=args.max_calls),
        acceptance=PairedImprovement(alpha=args.alpha),
        on_round=show,
    )
    result = pipeline.run(prompt)

    pipeline.store.save(out_dir)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    result.best_prompt.to_yaml(out_dir / "best_prompt.yaml")
    print("\n" + result.summary_text())
    lineage = pipeline.store.lineage(result.best_id)
    if len(lineage) > 1:
        print("\nlineage:")
        for c in lineage[1:]:
            print(f"  {c.edit.operator} [{c.edit.target_section}]: {c.edit.rationale}")
    print(f"\nbest prompt -> {out_dir / 'best_prompt.yaml'}")
    return 0


def cmd_compare(args) -> int:
    a, b = EvalReport.load(args.baseline), EvalReport.load(args.candidate)
    metric = args.metric or a.primary_metric
    res = paired_bootstrap(a.per_example(metric), b.per_example(metric))
    print(json.dumps(res.model_dump() | {"metric": metric, "significant": res.significant}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness-opt")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--prompt", required=True, help="prompt template YAML")
        sp.add_argument("--data", required=True, help="dataset JSONL")
        sp.add_argument("--metric", nargs="+", default=["exact_match"], choices=sorted(BUILTIN))
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(DEFAULT_MODELS))
        sp.add_argument("--model", help=f"target model id (defaults: {DEFAULT_MODELS})")
        sp.add_argument("--num-ctx", type=int, default=4096, help="Ollama context window")
        sp.add_argument("--max-tokens", type=int, default=256, help="target max output tokens")
        sp.add_argument("--temperature", type=float, default=0.0,
                        help="Ollama target temperature (0 = reproducible evals)")
        sp.add_argument("--mock-responder", help="module:function used by the mock provider")
        sp.add_argument("--concurrency", type=int, default=8)
        sp.add_argument("--cache", default=".harness_cache/llm.sqlite")
        sp.add_argument("--no-cache", action="store_true")
        sp.add_argument("--out", default="runs")
        sp.add_argument("--run-name", help="run directory name (default: timestamp)")

    e = sub.add_parser("eval", help="evaluate a prompt on a dataset")
    add_common(e)
    e.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    e.add_argument("--n", type=int, help="random subsample size")
    e.add_argument("--samples", type=int, default=1, help="samples per example")
    e.add_argument("--show-failures", type=int, default=5)
    e.set_defaults(func=cmd_eval)

    o = sub.add_parser("optimize", help="optimize a prompt with the agent pipeline")
    add_common(o)
    o.add_argument("--agent-model", help=f"model for the optimizer agents (defaults: {DEFAULT_AGENT_MODELS})")
    o.add_argument("--agent-max-tokens", type=int, default=1024)
    o.add_argument("--rounds", type=int, default=3)
    o.add_argument("--patience", type=int, default=2)
    o.add_argument("--max-calls", type=int, help="cap on uncached LLM calls")
    o.add_argument("--minibatch", type=int, default=12)
    o.add_argument("--hypotheses", type=int, default=2)
    o.add_argument("--proposals", type=int, default=2, help="proposals per hypothesis")
    o.add_argument("--val-candidates", type=int, default=2)
    o.add_argument("--screen", default="minibatch", choices=["minibatch", "halving"],
                   help="train screening: one shared minibatch, or successive halving")
    o.add_argument("--rungs", type=int, nargs="+", default=[4, 8, 16],
                   help="successive-halving slice sizes (with --screen halving)")
    o.add_argument("--critic", action="store_true", help="LLM review of each edit before measuring it")
    o.add_argument("--no-reflector", action="store_true", help="disable lessons between rounds")
    o.add_argument("--alpha", type=float, default=0.1, help="acceptance significance level")
    o.add_argument("--train", type=float, default=0.5, help="train fraction")
    o.add_argument("--val", type=float, default=0.25, help="val fraction (test = rest)")
    o.add_argument("--stratify-by", help="metadata key or 'expected' to stratify splits")
    o.set_defaults(func=cmd_optimize)

    c = sub.add_parser("compare", help="paired bootstrap between two saved runs")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--metric")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
