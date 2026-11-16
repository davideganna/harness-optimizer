import json
import sys
from pathlib import Path

from harness_optimizer.cli import main

EXAMPLE = Path(__file__).parents[1] / "examples" / "sentiment"


def test_cli_eval_and_compare_with_mock(tmp_path, capsys, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    common = ["--data", str(EXAMPLE / "data.jsonl"), "--provider", "mock",
              "--mock-responder", "responder:respond", "--no-cache", "--out", str(tmp_path)]

    assert main(["eval", "--prompt", str(EXAMPLE / "prompt_v0.yaml"), "--run-name", "v0", *common]) == 0
    out0 = capsys.readouterr().out
    assert json.loads(out0[: out0.index("\n}") + 2])["metrics"]["exact_match"]["mean"] < 0.5

    assert main(["eval", "--prompt", str(EXAMPLE / "prompt_v1.yaml"), "--run-name", "v1", *common]) == 0
    capsys.readouterr()

    assert (tmp_path / "v0" / "trace.jsonl").exists()
    assert main(["compare", str(tmp_path / "v0"), str(tmp_path / "v1")]) == 0
    cmp = json.loads(capsys.readouterr().out)
    assert cmp["significant"] and cmp["mean_diff"] > 0.5
    sys.modules.pop("responder", None)


def test_cli_optimize_with_mock(tmp_path, capsys, monkeypatch):
    from test_pipeline import dataset

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    data = tmp_path / "toy.jsonl"
    dataset().to_jsonl(data)
    prompt = tmp_path / "p.yaml"
    from harness_optimizer.core import PromptTemplate

    PromptTemplate.from_string("Is this review positive or negative?\nReview: {{text}}").to_yaml(prompt)
    rc = main(["optimize", "--prompt", str(prompt), "--data", str(data), "--provider", "mock",
               "--mock-responder", "test_pipeline:cli_responder", "--no-cache", "--stratify-by", "expected",
               "--out", str(tmp_path), "--run-name", "opt"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ACCEPTED" in out and "test baseline 0.000 -> best 1.000" in out
    run = tmp_path / "opt"
    assert (run / "best_prompt.yaml").exists() and (run / "result.json").exists()
    assert (run / "candidates.jsonl").exists()


def test_cli_ablate_with_mock(tmp_path, capsys, monkeypatch):
    from test_pipeline import dataset

    from harness_optimizer.core import PromptTemplate

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    data = tmp_path / "toy.jsonl"
    dataset().to_jsonl(data)
    prompt = tmp_path / "p.yaml"
    PromptTemplate.from_string("Is this review positive or negative?\nReview: {{text}}").to_yaml(prompt)
    rc = main(["ablate", "--prompt", str(prompt), "--data", str(data), "--provider", "mock",
               "--mock-responder", "test_pipeline:cli_responder", "--no-cache", "--stratify-by", "expected",
               "--variants", "base", "halving", "--seeds", "0", "1", "--rounds", "2", "--minibatch", "8",
               "--rungs", "4", "8", "--out", str(tmp_path), "--run-name", "abl"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| base | 2 |" in out and "| halving | 2 |" in out
    assert (tmp_path / "abl" / "rows.jsonl").read_text().count("\n") == 4
