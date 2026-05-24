"""End-to-end pipeline tests.

We use the real scorer objects but inject their leaf functions (fake embedder, fake NLL,
fake reward, fake judges) so the full wiring — Stage 1 -> gating -> Stage 2 -> composite ->
EvalResult — runs offline.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from creativity_evaluators.pipeline import Scorers, evaluate, evaluate_batch
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.lexical import LexicalScorer
from creativity_evaluators.stage1.perplexity import PerplexityScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_evaluators.stage2.ensemble import Ensemble
from creativity_evaluators.types import DIMENSIONS, EvalConfig, EvalResult, JudgeResult, PromptCategory


def _fake_embed(text: str):
    tokens = text.split()
    vecs = []
    for tok in tokens:
        seed = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % (2**32)
        vecs.append(np.random.default_rng(seed).normal(size=16))
    return tokens, (np.array(vecs) if vecs else np.zeros((0, 16)))


class _FakeJudge:
    def __init__(self, name, family, val):
        self.name = name
        self.family = family
        self._val = val

    def score(self, prompt, response):
        return JudgeResult(scores={d: self._val for d in DIMENSIONS}, reasoning="", judge_name=self.name)


def _scorers():
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=LexicalScorer(),
        ppl=PerplexityScorer(nll_fn=lambda t: (2.0 * len(t.split()), len(t.split()))),
        reward=RewardModelScorer(score_fn=lambda p, r: float(len(r))),
        ensemble=Ensemble([_FakeJudge("a", "openai", 4), _FakeJudge("b", "anthropic", 4)]),
    )


_RECORDS = [
    {"prompt": "Invent a use for a brick.", "response": "a floating lantern for koi ponds at dusk glowing softly", "category": "creative"},
    {"prompt": "Invent a use for a brick.", "response": "you can build a wall with the brick yes a wall", "category": "creative"},
    {"prompt": "What is 2+2?", "response": "the answer is four and that is the standard result here", "category": "neutral"},
    {"prompt": "Define water.", "response": "water is a clear liquid that people drink every single day", "category": "neutral"},
]


def test_returns_one_result_per_record():
    results = evaluate_batch(_RECORDS, EvalConfig(enable_stage2=True), scorers=_scorers(), n_workers=2)
    assert len(results) == len(_RECORDS)
    assert all(isinstance(r, EvalResult) for r in results)


def test_scores_finite_and_categories_coerced():
    results = evaluate_batch(_RECORDS, EvalConfig(enable_stage2=True), scorers=_scorers())
    assert all(math.isfinite(r.creativity_score) for r in results)
    assert results[0].category == PromptCategory.CREATIVE
    assert results[2].category == PromptCategory.NEUTRAL
    assert all(r.n_tokens > 0 for r in results)


def test_config_hash_stamped():
    config = EvalConfig(enable_stage2=False)
    results = evaluate_batch(_RECORDS, config, scorers=_scorers())
    assert all(r.config_hash == config.config_hash for r in results)


def test_stage2_disabled_makes_all_stage1_only():
    results = evaluate_batch(_RECORDS, EvalConfig(enable_stage2=False), scorers=_scorers())
    assert all(r.stage1_only for r in results)
    assert all(r.ensemble is None for r in results)


def test_floor_zero_runs_stage2_for_all():
    config = EvalConfig(enable_stage2=True, stage1_percentile_floor=0.0)
    results = evaluate_batch(_RECORDS, config, scorers=_scorers())
    assert all(r.ensemble is not None for r in results)
    assert all(not r.stage1_only for r in results)


def test_ensemble_presence_matches_stage1_only_flag():
    results = evaluate_batch(_RECORDS, EvalConfig(stage1_percentile_floor=75.0), scorers=_scorers())
    for r in results:
        assert (r.ensemble is None) == r.stage1_only


def test_evaluate_single_returns_eval_result():
    res = evaluate(
        "Invent a use for a brick.",
        "a tiny lighthouse for garden snails to navigate the patio at night",
        PromptCategory.CREATIVE,
        EvalConfig(enable_stage2=False),
        scorers=_scorers(),
    )
    assert isinstance(res, EvalResult)
    assert res.category == PromptCategory.CREATIVE
