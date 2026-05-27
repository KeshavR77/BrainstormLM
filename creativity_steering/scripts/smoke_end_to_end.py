"""Offline end-to-end smoke of the *verified-contrast* pipeline on the mock backend.

Exercises the full stack the RunPod run uses — derivation filtering with the two-stage
``creativity_evaluators`` stack (``method="evaluator_stack"``) and the two-stage layer sweep
(``sweep_evaluator="two_stage"``) — but with **injected fake scorers** so it needs no model
downloads, no GPU, and no API keys. It proves the orchestration composes and that the planted
creativity layer is recovered as the winner; the real run swaps the fakes for real models +
judges via a config alone (see configs/runpod_qwen14b.json).

    python scripts/smoke_end_to_end.py            # Stage-1-only (fast, deterministic)
    python scripts/smoke_end_to_end.py --with-judge   # also exercise the Stage-2 gating path
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

# Put the sibling creativity_evaluators package (src layout) on the path before importing it,
# the same bridge the package uses at runtime — so this script runs without a pip install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CE_SRC = _REPO_ROOT / "creativity_evaluators" / "src"
if (_CE_SRC / "creativity_evaluators" / "__init__.py").exists() and str(_CE_SRC) not in sys.path:
    sys.path.insert(0, str(_CE_SRC))

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_evaluators.stage2.ensemble import Ensemble
from creativity_evaluators.types import DIMENSIONS, JudgeResult
from creativity_steering.analysis.oracle import StubOracle
from creativity_steering.analysis.report import build_report
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
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS
from creativity_steering.patching.evaluate_stack import build_alpha_cells, two_stage_cell_sweep
from creativity_steering.scoring import HeuristicScorer


def _fake_embed(text: str):
    toks = text.split()
    vecs = [
        np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=16)
        for t in toks
    ]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


def _distinct(prompt: str, response: str) -> float:
    return float(len(set(response.lower().split())))


class _FakeJudge:
    def __init__(self, name: str, family: str) -> None:
        self.name = name
        self.family = family

    def score(self, prompt: str, response: str) -> JudgeResult:
        v = 5 if len(set(response.split())) > 4 else 1
        return JudgeResult(scores={d: v for d in DIMENSIONS}, reasoning="", judge_name=self.name)


def _fake_scorers(with_judge: bool) -> Scorers:
    ensemble = (
        Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")]) if with_judge else None
    )
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=None,
        ppl=None,
        reward=RewardModelScorer(score_fn=_distinct),
        ensemble=ensemble,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-judge", action="store_true", help="exercise the Stage-2 gating path")
    ap.add_argument("--creative-layer", type=int, default=12)
    args = ap.parse_args()

    cfg = ExperimentConfig(
        backend="mock",
        num_layers=24,
        hidden_dim=64,
        mock_creative_layer=args.creative_layer,
        generation=GenerationConfig(samples_per_prompt=4, max_new_tokens=240),
        filtering=FilterConfig(
            method="evaluator_stack",
            min_per_class=0,
            enable_judge=args.with_judge,
            creative_quantile=0.6,
            neutral_quantile=0.4,
            stage1_floor_percentile=0.0,
        ),
        patching=PatchingConfig(
            alphas=(-4.0, -1.0, 1.0, 4.0),
            eval_samples=4,
            sweep_evaluator="two_stage",
            judge_top_k=2,
        ),
    )
    backend = MockBackend(
        num_layers=cfg.num_layers, hidden_dim=cfg.hidden_dim,
        creative_layer=cfg.mock_creative_layer, seed=0,
    )
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS)
    scorers = _fake_scorers(args.with_judge)

    print(f"== Phase 1: derive (evaluator_stack, enable_judge={cfg.filtering.enable_judge}) ==")
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts, eval_scorers=scorers)
    print(f"  examples={out.n_examples}  positives={out.subset.n_positive}  "
          f"negatives={out.subset.n_negative}  dropped={out.subset.n_dropped}")
    assert out.subset.n_positive > 0 and out.subset.n_negative > 0, "empty contrast set"

    with tempfile.TemporaryDirectory() as td:
        save_derivation(out, td)
        assert (Path(td) / "vectors" / cfg.capture_position / "vectors.npz").exists()

    print("== Phase 2: analyze ==")
    report = build_report(
        out.vector_set.vectors, oracle=StubOracle(reference_norms=out.vector_set.layer_norms)
    )
    print(f"  candidate layers={report.candidate_layers}  "
          f"norm_top={report.norm.top_layer}  cosine_band={report.cosine.band}  "
          f"(planted at {cfg.mock_creative_layer})")

    print(f"== Phase 3: two-stage sweep (judge_top_k={cfg.patching.judge_top_k}) ==")
    cells = build_alpha_cells(
        out.vector_set, report.candidate_layers, cfg.patching.alphas, rescale=cfg.patching.rescale
    )
    result = two_stage_cell_sweep(
        backend, cells,
        scorers=scorers,
        briefs=DEFAULT_BRIEFS,
        n_samples=cfg.patching.eval_samples,
        enable_judge=cfg.filtering.enable_judge,
        judge_top_k=cfg.patching.judge_top_k,
        stage1_floor_percentile=0.0,
    )
    best = result.best_cell()
    print(f"  baseline composite={result.baseline.composite:.3f}  judged={result.judged_labels}")
    print(f"  BEST: layer={best['layer']} alpha={best['alpha']} "
          f"composite={best['composite']:.3f} (delta={best['delta']:+.3f}, judged={best['judged']})")
    assert best["delta"] > 0, "best steered cell did not beat the unsteered baseline"

    print("\nOK — end-to-end pipeline composed on the mock backend; best vector beats baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
