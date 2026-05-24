"""Evaluator-stack filtering: score generated responses with the two-stage
``creativity_evaluators`` stack, keep top creative / bottom neutral for DiffMeans.

Runs offline by injecting ``creativity_evaluators.Scorers`` whose leaf functions are fakes
(hash embedder for DSI, distinct-word-count reward, optional fake judge ensemble).
"""

import hashlib

import numpy as np
import pytest

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_evaluators.stage2.ensemble import Ensemble
from creativity_evaluators.types import DIMENSIONS, JudgeResult
from creativity_steering.config import (
    CAPTURE_LAST_RESPONSE,
    ExperimentConfig,
    FilterConfig,
    GenerationConfig,
)
from creativity_steering.derivation.evaluator_filter import (
    evaluate_and_filter,
    select_by_quantile,
)
from creativity_steering.derivation.pipeline import derive_vectors
from creativity_steering.derivation.prompts import (
    CREATIVE_SYSTEM,
    NEUTRAL_SYSTEM,
    TOPICS,
    PromptSet,
)
from creativity_steering.derivation.records import ResponseExample
from creativity_steering.models.mock import MockBackend
from creativity_steering.scoring import HeuristicScorer

DIVERSE = "aurora zephyr lattice ember quartz nimbus drift cobalt"
DIVERSE2 = "marigold thistle harbor glass willow cinder vellum prism"
REPET = "the the the the the the the the"
REPET2 = "and and and and and and and and"
REPET3 = "of of of of of of of of"


def _fake_embed(text):
    toks = text.split()
    vecs = [np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=16) for t in toks]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


def _distinct(prompt, response):  # reward = vocabulary richness, correlates with DSI
    return float(len(set(response.lower().split())))


def _ex(framing, response, layers=3, dim=4):
    return ResponseExample(
        framing=framing,
        system="s",
        prompt="p",
        response=response,
        activations={CAPTURE_LAST_RESPONSE: np.ones((layers, dim), dtype=np.float32)},
    )


class _FakeJudge:
    def __init__(self, name, family):
        self.name = name
        self.family = family

    def score(self, prompt, response):
        v = 5 if len(set(response.split())) > 4 else 1
        return JudgeResult(scores={d: v for d in DIMENSIONS}, reasoning="", judge_name=self.name)


def _scorers(ensemble=None):
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=None,
        ppl=None,
        reward=RewardModelScorer(score_fn=_distinct),
        ensemble=ensemble,
    )


# ---------------------------------------------------------------------------
# Quantile selection
# ---------------------------------------------------------------------------


def test_select_by_quantile_high_and_low():
    pool = [_ex("creative", "x") for _ in range(5)]
    for ex, s in zip(pool, [0.1, 0.2, 0.3, 0.4, 0.5]):
        ex.score = s
    top = select_by_quantile(pool, quantile=0.8, keep="high")
    assert {round(e.score, 1) for e in top} == {0.5}
    bottom = select_by_quantile(pool, quantile=0.2, keep="low")
    assert {round(e.score, 1) for e in bottom} == {0.1}


# ---------------------------------------------------------------------------
# evaluate_and_filter
# ---------------------------------------------------------------------------


def test_keeps_top_creative_and_bottom_neutral():
    examples = [
        _ex("creative", DIVERSE),
        _ex("creative", DIVERSE2),
        _ex("creative", REPET),
        _ex("creative", REPET2),
        _ex("creative", REPET3),
        _ex("neutral", DIVERSE),
        _ex("neutral", DIVERSE2),
        _ex("neutral", REPET),
        _ex("neutral", REPET2),
        _ex("neutral", REPET3),
    ]
    subset = evaluate_and_filter(
        examples, scorers=_scorers(), enable_judge=False,
        creative_quantile=0.8, neutral_quantile=0.2,
    )
    assert all(ex.score is not None for ex in examples)
    assert subset.n_positive >= 1 and subset.n_negative >= 1
    assert all(p.framing == "creative" for p in subset.positives)
    assert all(n.framing == "neutral" for n in subset.negatives)

    creatives = [e for e in examples if e.framing == "creative"]
    neutrals = [e for e in examples if e.framing == "neutral"]
    dropped_c = [c for c in creatives if c not in subset.positives]
    dropped_n = [n for n in neutrals if n not in subset.negatives]
    # kept positives strictly out-score dropped creatives; kept negatives strictly under-score
    # dropped neutrals (the quantile cut leaves no overlap).
    if dropped_c:
        assert min(p.score for p in subset.positives) > max(c.score for c in dropped_c)
    if dropped_n:
        assert max(n.score for n in subset.negatives) < min(n.score for n in dropped_n)
    # the diverse creative responses are the ones kept as positives
    assert all(p.response in (DIVERSE, DIVERSE2) for p in subset.positives)
    assert subset.n_dropped == len(examples) - subset.n_positive - subset.n_negative


def test_both_phases_run_with_judge():
    ensemble = Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")])
    examples = [
        _ex("creative", DIVERSE),
        _ex("creative", REPET),
        _ex("neutral", DIVERSE),
        _ex("neutral", REPET),
    ]
    subset = evaluate_and_filter(
        examples, scorers=_scorers(ensemble=ensemble), enable_judge=True,
        generator_family="qwen", stage1_floor_percentile=0.0,
        creative_quantile=0.5, neutral_quantile=0.5,
    )
    assert all(ex.score is not None for ex in examples)
    creatives = [e for e in examples if e.framing == "creative"]
    assert max(creatives, key=lambda e: e.score).response == DIVERSE
    assert subset.n_positive >= 1 and subset.n_negative >= 1


def test_min_per_class_guardrail_raises():
    examples = [_ex("creative", DIVERSE), _ex("neutral", REPET)]
    with pytest.raises(ValueError, match="too small"):
        evaluate_and_filter(examples, scorers=_scorers(), enable_judge=False, min_per_class=5)


def test_empty_examples_returns_empty_subset():
    subset = evaluate_and_filter([], scorers=_scorers(), enable_judge=False)
    assert subset.n_positive == 0 and subset.n_negative == 0


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_derive_vectors_uses_evaluator_stack():
    cfg = ExperimentConfig(
        backend="mock",
        num_layers=12,
        hidden_dim=32,
        mock_creative_layer=6,
        generation=GenerationConfig(samples_per_prompt=3, max_new_tokens=240),
        filtering=FilterConfig(
            method="evaluator_stack", min_per_class=0, enable_judge=False,
            creative_quantile=0.5, neutral_quantile=0.5,
        ),
    )
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:12])
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts, eval_scorers=_scorers())
    assert out.subset.n_positive > 0 and out.subset.n_negative > 0
    assert out.vector_set.n_layers == 12
