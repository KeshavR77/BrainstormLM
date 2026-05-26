"""CLI integration: the ``sweep`` command's ``sweep_evaluator="two_stage"`` branch — the
exact path the RunPod run drives. ``build_scorers`` is monkeypatched to inject fakes so the
test runs offline (no model downloads, no API keys); everything else is the real CLI.
"""

import hashlib
import json

import numpy as np
from typer.testing import CliRunner

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_steering.cli import app
from creativity_steering.config import (
    ExperimentConfig,
    FilterConfig,
    GenerationConfig,
    PatchingConfig,
)
from creativity_steering.derivation.pipeline import derive_vectors, save_derivation
from creativity_steering.derivation.prompts import (
    CREATIVE_SYSTEM,
    NEUTRAL_SYSTEM,
    TOPICS,
    PromptSet,
)
from creativity_steering.models.mock import MockBackend
from creativity_steering.scoring import HeuristicScorer

runner = CliRunner()


def _fake_embed(text):
    toks = text.split()
    vecs = [
        np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=16)
        for t in toks
    ]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


def _distinct(prompt, response):
    return float(len(set(response.lower().split())))


def _fake_scorers(*_a, **_k):
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=None,
        ppl=None,
        reward=RewardModelScorer(score_fn=_distinct),
        ensemble=None,  # enable_judge is off in this config -> Stage 1 only
    )


def _config(out_dir):
    return ExperimentConfig(
        backend="mock",
        num_layers=8,
        hidden_dim=16,
        mock_creative_layer=4,
        generation=GenerationConfig(samples_per_prompt=3, max_new_tokens=120),
        filtering=FilterConfig(
            method="evaluator_stack", min_per_class=0, enable_judge=False,
            creative_quantile=0.5, neutral_quantile=0.5,
        ),
        patching=PatchingConfig(
            alphas=(-2.0, 2.0), eval_samples=2,
            sweep_evaluator="two_stage", judge_top_k=1,
        ),
        out_dir=str(out_dir),
    )


def test_cli_sweep_two_stage(tmp_path, monkeypatch):
    # build_scorers is imported inside two_stage_cell_sweep at call time, so patching the
    # module attribute is enough.
    import creativity_evaluators.pipeline as ce_pipeline

    monkeypatch.setattr(ce_pipeline, "build_scorers", _fake_scorers)

    cfg = _config(tmp_path / "run")
    backend = MockBackend(num_layers=8, hidden_dim=16, creative_layer=4, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:8])

    # Stage the run dir the CLI sweep expects (vectors on disk), using injected fakes.
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts, eval_scorers=_fake_scorers())
    run_dir = tmp_path / "run"
    save_derivation(out, run_dir)
    cfg_path = tmp_path / "cfg.json"
    cfg.save(cfg_path)

    result = runner.invoke(
        app,
        ["sweep", "--run", str(run_dir), "--config", str(cfg_path),
         "--position", "last_response_token", "--layers", "3,4,5"],
    )
    assert result.exit_code == 0, result.output
    assert "two-stage sweep" in result.output
    assert "best cell" in result.output

    sweep_json = run_dir / "patching" / "last_response_token" / "sweep_results.json"
    data = json.loads(sweep_json.read_text())
    assert data["evaluator"] == "two_stage"
    assert data["best"] is not None
    # 3 layers x 2 alphas = 6 cells
    assert len(data["cells"]) == 6


def test_cli_experiments_two_stage(tmp_path, monkeypatch):
    """The `experiments` command (Exp-2: all-layers + multi-vector on promising layers)."""
    import creativity_evaluators.pipeline as ce_pipeline

    monkeypatch.setattr(ce_pipeline, "build_scorers", _fake_scorers)

    cfg = _config(tmp_path / "run")
    backend = MockBackend(num_layers=8, hidden_dim=16, creative_layer=4, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:8])
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts, eval_scorers=_fake_scorers())
    run_dir = tmp_path / "run"
    save_derivation(out, run_dir)
    cfg_path = tmp_path / "cfg.json"
    cfg.save(cfg_path)

    # analyze -> writes report.json with candidate_layers
    r1 = runner.invoke(app, ["analyze", "--run", str(run_dir), "--position", "last_response_token"])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        app,
        ["experiments", "--run", str(run_dir), "--config", str(cfg_path),
         "--position", "last_response_token", "--n-promising", "3"],
    )
    assert r2.exit_code == 0, r2.output
    assert "Exp-2 on promising layers" in r2.output
    assert "best experiment" in r2.output

    data = json.loads((run_dir / "patching" / "last_response_token" / "experiments_results.json").read_text())
    assert data["evaluator"] == "two_stage"
    assert data["best"] is not None
    # all-layers: 2 sources x 4 alphas = 8; multi-vector: 1 group x 2 alphas = 2 -> 10 cells
    assert len(data["cells"]) == 10
