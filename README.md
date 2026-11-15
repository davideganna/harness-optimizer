# Harness Optimizer

A research-oriented, open-source Python package for **agentic prompt optimization**, built on
top of a **rigorous evaluation harness**.

The guiding principle: *the optimizer is only as good as the signal it gets back.* So the eval
harness is built first and the agents are clients of it.

> Status: `0.1.0.dev0`. Blocks A–C (core models, LLM gateway, eval harness), **pipeline mode**
> (Failure Analyst → Proposer → guards → optional Critic → screen → paired-bootstrap acceptance →
> Reflector lessons) and successive-halving screening are implemented and tested. Agent mode is
> next. See `MODULES.md` for a guide to every module.

## Architecture

```
                         ┌──────────────────────────────┐
   User / CLI / API ───► │  Experiment (config + run)   │
                         └──────────────┬───────────────┘
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌──────────────────┐     ┌───────────────────────┐    ┌──────────────────┐
   │ A Task & Data    │◄────│ D Optimization Agents │───►│ G Budget/Policy  │
   └────────┬─────────┘     └─────┬───────────┬─────┘    └──────────────────┘
            │                     │ tools     │
            │                     ▼           ▼
            │           ┌────────────────┐  ┌──────────────────────┐
            └──────────►│ C Eval Harness │─►│ F Experiment Memory  │
                        └───────┬────────┘  └──────────────────────┘
                                ▼                 E Search strategy
                     ┌──────────────────────┐     (which candidates
                     │ B LLM Gateway        │      survive / expand)
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │ H Observability      │
                     └──────────────────────┘
```

| Block | Package | Status |
|---|---|---|
| A Core data model: `PromptTemplate` (sectioned), `Candidate` (lineage), `Dataset` + splits, `EvalReport` | `core/` | done |
| B LLM gateway: provider adapters, SQLite cache, retries/backoff, concurrency, per-role cost ledger | `llm/` | done (Ollama default, Anthropic, mock) |
| C Eval harness: runner, deterministic metrics, composite gates, LLM judge, bootstrap stats | `eval/` | done |
| D Agents: failure analyst, proposer, guards, critic, reflector, pipeline loop; few-shot/orchestrator next | `agents/` | pipeline mode done |
| E Search: paired-improvement acceptance, successive halving; beam, evolutionary next | `search/` | partial |
| F Memory: candidate store + lineage, lessons; state summarizer next | `memory/` | partial |
| G Budget & policy: limits, patience stopping; overfitting guards next | `policy/` | partial |
| H Observability: JSONL traces (done), reports, replay | `obs/` | partial |

## Install (development)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,anthropic]"   # Windows
# or: .venv/bin/python -m pip install -e ".[dev,anthropic]"   # Linux/macOS
```

## Models: Ollama by default, Claude optional

By default every LLM call (target model, judges and later the optimizer agents) goes to a
**free local model through [Ollama](https://ollama.com)**. The default is `llama3.2:3b`
(`ModelConfig()` gives `provider="ollama", model="llama3.2:3b"`).

```bash
ollama pull llama3.2:3b          # the Ollama app must be running (it serves http://localhost:11434)
```

- `OllamaClient` uses the native API through the official `ollama` library. Defaults: `think=False`, `num_ctx=4096`, `keep_alive="10m"`.
- Set `OLLAMA_HOST` to use a different server.
- Per-call `params` accepted: `think`, `format`, `keep_alive`, `options` (e.g. `{"options": {"temperature": 0, "seed": 1}}` for more reproducible evals).
- Cost is reported as $0 for Ollama.

Why `llama3.2:3b`: on a 4 GB GPU it fits entirely in VRAM. On the sentiment example it measured
0.35 s per call with 29–30/30 correct. `qwen3:4b` only partially fit (30/37 layers), took 1.6 s per call,
and wrote its reasoning into the answer even with thinking off.

**Older NVIDIA GPUs** (compute capability 5.0–6.2, such as the GTX 10xx series) need driver 570 or newer
for Ollama's CUDA backend. With an older driver the model runner crashes on load. The workaround is the
user environment variable `OLLAMA_LLM_LIBRARY=vulkan`, which only affects Ollama. Updating the NVIDIA
driver is the supported fix.

**Claude (optional).** Use `--provider anthropic` on the CLI, or `ModelConfig(provider="anthropic", model="claude-opus-5-5")` in code.
Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY=...`. The `.env` file is git-ignored and the CLI
loads it automatically; in scripts, call `harness_optimizer.config.load_env()`.

## Quickstart

Offline demo with a mock "model" (no API key needed):

```bash
python examples/sentiment/run_demo.py           # mock model, offline
python examples/sentiment/run_demo.py --live    # local Ollama
```

CLI:

```bash
# evaluate a prompt (mock provider)
harness-opt eval --prompt examples/sentiment/prompt_v0.yaml --data examples/sentiment/data.jsonl \
    --provider mock --mock-responder responder:respond --run-name v0

# real model: local Ollama by default (llama3.2:3b), free
harness-opt eval --prompt examples/sentiment/prompt_v1.yaml --data examples/sentiment/data.jsonl --split val

# Claude instead (needs ANTHROPIC_API_KEY in .env)
harness-opt eval --prompt examples/sentiment/prompt_v1.yaml --data examples/sentiment/data.jsonl \
    --provider anthropic --split val

# is v1 significantly better than v0? (paired bootstrap)
harness-opt compare runs/v0 runs/v1

# optimize the weak prompt automatically (agents: qwen3:4b, target: llama3.2:3b)
harness-opt optimize --prompt examples/sentiment/prompt_v0.yaml --data examples/sentiment/data.jsonl \
    --stratify-by expected --rounds 3

# the same, with successive-halving screening and the LLM critic switched on
harness-opt optimize --prompt examples/sentiment/prompt_v0.yaml --data examples/sentiment/data.jsonl \
    --stratify-by expected --rounds 3 --screen halving --rungs 4 8 15 --critic
```

Every optional component can be switched on or off, so its effect can be measured by ablation:
`--screen minibatch|halving`, `--critic`, `--no-reflector`.

Each optimize run saves `best_prompt.yaml`, `result.json` (rounds, hypotheses, every candidate's
scores and p-values), `candidates.jsonl` (the lineage tree), every eval report, and `trace.jsonl`.

A measured live run (local Ollama, $0, about 3 minutes): on the weak sentiment prompt, the analyst
found that the model answered in sentences instead of bare labels, and the proposer added
"Output only the label…". Val went from 0.00 to 1.00 (accepted, p < 0.001), and the one-time test
evaluation went from 0.00 to 1.00 on 9 held-out examples. A second run with halving screening,
the critic and the reflector also fixed it in one round (about 2.5 minutes; the critic approved the
edit, and the reflector wrote two lessons). The task is easy; it shows the loop working, not
optimizer quality.

Python API:

```python
from harness_optimizer import Candidate, Dataset, EvalRunner, LLMGateway, ModelConfig, PromptTemplate, SQLiteCache
from harness_optimizer.eval.metrics import ExactMatch
from harness_optimizer.llm import OllamaClient

gateway = LLMGateway(OllamaClient(), cache=SQLiteCache())
runner = EvalRunner(gateway, ModelConfig(), [ExactMatch()])   # ollama / llama3.2:3b
splits = Dataset.from_jsonl("data.jsonl").split(0.5, 0.25, 0.25, stratify_by="expected")

report = runner.evaluate(Candidate.root(PromptTemplate.from_yaml("prompt.yaml")), splits.val, split="val")
print(report.summary())
for f in report.failures(k=5):
    print(f.to_text())
```

## File formats

**Prompt (YAML)**: ordered sections; `{{var}}` placeholders; `role` is `system` or `user`.

```yaml
sections:
  - {name: system, role: system, content: You are a sentiment classifier.}
  - {name: task, role: user, content: "Classify: {{text}}"}
  - {name: output_format, role: user, content: Answer with the label only.}
```

**Dataset (JSONL)**: one example per line.

```json
{"id": "s01", "inputs": {"text": "Great phone"}, "expected": "positive", "metadata": {}}
```

## Research-validity rules built into the harness

- **Failed generations are scored 0, never dropped.** Dropping them would inflate scores for prompts that cause errors or refusals.
- **Statistics are per example.** With `samples > 1` the runner averages over samples first and bootstraps over examples.
- **Every comparison is paired.** `paired_bootstrap` compares candidates on the same examples.
- **Splits are explicit.** Agents will see train failures, only aggregate val scores, and never test.
- **Cache keys include `sample_index`.** Repeated samples are real samples, and re-runs are free and reproducible.
- **No silent model fallback.** A refusal is recorded as such instead of being answered by a different model.

## Roadmap

1. ~~Core models, gateway with cache, eval harness with deterministic metrics~~
2. LLM-judge reliability: pairwise and position-swap, agreement with human labels
3. ~~**Pipeline mode**: Failure Analyst, then Proposer, then greedy search (the first optimizer)~~
4. ~~Memory, Critic, Reflector, successive-halving search~~
5. **Agent mode**: an orchestrator with tools (evaluate, get_failures, propose, compare, finish)
6. Benchmarks, baselines and ablations (pipeline vs agent mode, with and without the critic or memory)

## Development

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src tests
```
