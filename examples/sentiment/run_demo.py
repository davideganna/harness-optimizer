"""End-to-end demo of the eval harness.

    python examples/sentiment/run_demo.py                        # mock "model", offline
    python examples/sentiment/run_demo.py --live                 # local Ollama (default llama3.2:3b)
    python examples/sentiment/run_demo.py --live --model qwen3:4b
    python examples/sentiment/run_demo.py --anthropic            # Claude (needs API key)
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from responder import respond

from harness_optimizer import (
    Candidate,
    Dataset,
    Edit,
    EvalRunner,
    LLMGateway,
    MockLLM,
    ModelConfig,
    PromptTemplate,
    paired_bootstrap,
)
from harness_optimizer.eval.metrics import Contains, ExactMatch
from harness_optimizer.llm import MemoryCache


def main() -> None:
    args = sys.argv[1:]
    model = args[args.index("--model") + 1] if "--model" in args else None
    if "--anthropic" in args:
        from harness_optimizer.config import load_env
        from harness_optimizer.llm.anthropic_client import AnthropicClient

        load_env()
        client, target = AnthropicClient(), ModelConfig(
            provider="anthropic", model=model or "claude-opus-5-5", max_tokens=256,
            params={"output_config": {"effort": "low"}},
        )
    elif "--live" in args:
        from harness_optimizer.llm import OllamaClient

        client = OllamaClient()
        target = ModelConfig(max_tokens=64) if model is None else ModelConfig(model=model, max_tokens=64)
    else:
        client, target = MockLLM(respond), ModelConfig(provider="mock", model="mock")
    print(f"target: {target.provider}/{target.model}\n")

    gateway = LLMGateway(client, cache=MemoryCache())
    runner = EvalRunner(gateway, target, metrics=[ExactMatch(), Contains()])
    data = Dataset.from_jsonl(HERE / "data.jsonl")

    v0 = Candidate.root(PromptTemplate.from_yaml(HERE / "prompt_v0.yaml"))
    v1 = v0.child(
        PromptTemplate.from_yaml(HERE / "prompt_v1.yaml"),
        Edit(operator="rewrite_section", target_section="task",
             rationale="name the neutral class and require a bare label"),
        created_by="human",
    )

    r0 = runner.evaluate(v0, data, split="all")
    r1 = runner.evaluate(v1, data, split="all")

    for name, r in [("v0", r0), ("v1", r1)]:
        for m in r.metric_names:
            ci = r.ci(m)
            print(f"{name} {m:12s} {ci.mean:.3f}  95% CI [{ci.low:.3f}, {ci.high:.3f}]")

    cmp = paired_bootstrap(r0.per_example(), r1.per_example())
    print(f"\nv1 - v0 on exact_match: {cmp.mean_diff:+.3f} "
          f"CI [{cmp.low:+.3f}, {cmp.high:+.3f}]  significant={cmp.significant}")

    print("\nTop v1 failures:")
    for f in r1.failures(k=3):
        print(f.to_text(), end="\n\n")
    print("ledger:", {k: v.model_dump() for k, v in gateway.ledger.by_role.items()})


if __name__ == "__main__":
    main()
