"""Crash-safe + resumable derivation.

The top-up loop must persist the full state (datapoints + metrics + activations) to out_dir
after every round, and resume from that checkpoint on relaunch instead of regenerating — so a
stop/crash/migration loses at most the in-flight round.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_steering.config import ExperimentConfig, FilterConfig, GenerationConfig
from creativity_steering.derivation.pipeline import (
    derive_vectors,
    load_checkpoint_pool,
)
from creativity_steering.derivation.prompts import CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS, PromptSet
from creativity_steering.models.mock import MockBackend
from creativity_steering.scoring import HeuristicScorer


def _fake_embed(text):
    toks = text.split()
    vecs = [np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32)
            .normal(size=16) for t in toks]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


def _distinct(prompt, response):
    return float(len(set(response.lower().split())))


def _scorers():
    return Scorers(dsi=DSIScorer(embed_fn=_fake_embed), lexical=None, ppl=None,
                   reward=RewardModelScorer(score_fn=_distinct), ensemble=None)


class _CountingMock(MockBackend):
    """MockBackend that counts generate_and_capture calls, to prove resume skips regeneration."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.gen_calls = 0

    def generate_and_capture(self, *a, **k):
        self.gen_calls += 1
        return super().generate_and_capture(*a, **k)


def _cfg(target, chunk):
    return ExperimentConfig(
        backend="mock", num_layers=6, hidden_dim=16, mock_creative_layer=3,
        capture_position="mean_response",
        extra_capture_positions=("last_response_token", "assistant_start"),
        generation=GenerationConfig(samples_per_prompt=1, max_new_tokens=120),
        filtering=FilterConfig(
            method="evaluator_stack", enable_judge=False, one_per_prompt=True,
            generator_family="qwen", stage1_floor_percentile=0.0,
            creative_quantile=0.9, neutral_quantile=0.1, length_match=False,
            target_per_class=target, topup_chunk_size=chunk, min_per_class=0,
        ),
    )


def test_checkpoint_persists_pool_and_resumes(tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(target=9999, chunk=15)  # unreachable target -> exhausts the small bank, then saves
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:30])

    b1 = _CountingMock(num_layers=6, hidden_dim=16, creative_layer=3, seed=0)
    out1 = derive_vectors(b1, cfg, HeuristicScorer(), prompt_set=prompts,
                          eval_scorers=_scorers(), out_dir=out)
    # checkpoint files exist with metrics + activations
    rows = [json.loads(x) for x in (out / "scored_pool.jsonl").read_text().splitlines() if x]
    assert len(rows) == out1.n_examples == 60  # 30 topics x 2 framings
    assert {"score", "novelty", "quality_ok", "framing", "prompt", "response"} <= set(rows[0])
    acts = np.load(out / "activations" / "mean_response.npz")["acts"]
    assert acts.shape[0] == 60
    calls_run1 = b1.gen_calls
    assert calls_run1 == 60

    # Resume: same out_dir, same (exhausted) bank -> loads checkpoint, regenerates NOTHING.
    b2 = _CountingMock(num_layers=6, hidden_dim=16, creative_layer=3, seed=0)
    out2 = derive_vectors(b2, cfg, HeuristicScorer(), prompt_set=prompts,
                          eval_scorers=_scorers(), out_dir=out)
    assert b2.gen_calls == 0           # fully resumed from checkpoint
    assert out2.n_examples == 60


def test_load_checkpoint_pool_roundtrip(tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(target=3, chunk=40)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:80])
    backend = MockBackend(num_layers=6, hidden_dim=16, creative_layer=3, seed=0)
    derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts,
                   eval_scorers=_scorers(), out_dir=out)
    loaded = load_checkpoint_pool(out, ("mean_response", "last_response_token", "assistant_start"))
    assert loaded and all(e.novelty is not None for e in loaded)
    assert all("mean_response" in e.activations for e in loaded)
