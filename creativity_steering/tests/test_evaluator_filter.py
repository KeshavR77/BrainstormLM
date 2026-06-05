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
    select_by_threshold,
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


def test_select_by_threshold_high_and_low():
    pool = [_ex("creative", "x") for _ in range(5)]
    for ex, s in zip(pool, [0.1, 0.2, 0.3, 0.4, 0.5]):
        ex.score = s
    hi = select_by_threshold(pool, threshold=0.35, keep="high")
    assert sorted(round(e.score, 1) for e in hi) == [0.4, 0.5]
    lo = select_by_threshold(pool, threshold=0.25, keep="low")
    assert sorted(round(e.score, 1) for e in lo) == [0.1, 0.2]


def test_evaluate_and_filter_uses_thresholds_when_both_set():
    # Permissive thresholds keep every creative as positive / every neutral as negative,
    # proving the threshold path (not the quantile path) is taken.
    examples = [
        _ex("creative", DIVERSE),
        _ex("creative", DIVERSE2),
        _ex("neutral", REPET),
        _ex("neutral", REPET2),
    ]
    subset = evaluate_and_filter(
        examples, scorers=_scorers(), enable_judge=False,
        creative_score_min=-1e9, neutral_score_max=1e9,
    )
    assert subset.n_positive == 2 and subset.n_negative == 2 and subset.n_dropped == 0


def test_split_by_score_ignores_framing():
    # High-scoring responses tagged "neutral", low-scoring tagged "creative". With
    # split_by="score" the pools are formed from the whole set by score — framing is ignored —
    # so the high-scoring (neutral-framed) responses must land in positives.
    examples = [
        _ex("neutral", DIVERSE),
        _ex("neutral", DIVERSE2),
        _ex("creative", REPET),
        _ex("creative", REPET2),
    ]
    subset = evaluate_and_filter(
        examples, scorers=_scorers(), enable_judge=False, split_by="score",
        creative_score_min=0.2, neutral_score_max=0.05,
    )
    assert {p.response for p in subset.positives} == {DIVERSE, DIVERSE2}
    assert all(p.framing == "neutral" for p in subset.positives)  # framing ignored
    assert {n.response for n in subset.negatives} == {REPET, REPET2}


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


def _exp(framing, response, prompt, layers=3, dim=4):
    ex = _ex(framing, response, layers, dim)
    ex.prompt = prompt
    return ex


def test_one_per_prompt_caps_each_pole_at_one_response_per_prompt():
    # Two creative-framed samples for prompt "p1" (both diverse, both top-tail). With
    # one_per_prompt the positive pool keeps at most ONE of them (the higher-scoring), so a
    # single prompt cannot dominate the DiffMeans contrast.
    examples = [
        _exp("creative", DIVERSE, "p1"),
        _exp("creative", DIVERSE2, "p1"),
        _exp("creative", DIVERSE, "p2"),
        _exp("neutral", REPET, "p1"),
        _exp("neutral", REPET2, "p2"),
        _exp("neutral", REPET3, "p3"),
    ]
    subset = evaluate_and_filter(
        examples, scorers=_scorers(), enable_judge=False,
        creative_quantile=0.0, neutral_quantile=1.0,  # thresholds keep everything; cap does the work
        one_per_prompt=True,
    )
    # at most one positive per prompt
    pos_prompts = [p.prompt for p in subset.positives]
    assert len(pos_prompts) == len(set(pos_prompts))
    # p1 contributes exactly its single best-scoring creative response
    p1_pos = [p for p in subset.positives if p.prompt == "p1"]
    assert len(p1_pos) == 1
    # likewise the negative pool is capped at one per prompt
    neg_prompts = [n.prompt for n in subset.negatives]
    assert len(neg_prompts) == len(set(neg_prompts))


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


def test_derive_vectors_v3_one_per_prompt_quality_gated():
    # The v3 framed path end to end: 3 samples/prompt, top/bottom-10% novelty with a
    # one-per-prompt cap and a judge quality gate. Asserts each pole keeps at most one response
    # per prompt and every example carries the novelty + quality_ok selection signals.
    ensemble = Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")])
    cfg = ExperimentConfig(
        backend="mock",
        num_layers=12,
        hidden_dim=32,
        mock_creative_layer=6,
        generation=GenerationConfig(samples_per_prompt=3, max_new_tokens=240),
        filtering=FilterConfig(
            method="evaluator_stack", min_per_class=0, enable_judge=True,
            one_per_prompt=True, quality_floor=3.0, generator_family="qwen",
            stage1_floor_percentile=0.0, creative_quantile=0.5, neutral_quantile=0.5,
        ),
    )
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:12])
    out = derive_vectors(
        backend, cfg, HeuristicScorer(), prompt_set=prompts,
        eval_scorers=_scorers(ensemble=ensemble),
    )
    # one-per-prompt: no (framing, prompt) duplicated inside a pole
    assert len({p.prompt for p in out.subset.positives}) == out.subset.n_positive
    assert len({n.prompt for n in out.subset.negatives}) == out.subset.n_negative
    # selection signals populated; positives cleared the quality gate
    assert all(p.novelty is not None for p in out.subset.positives)
    assert all(p.quality_ok for p in out.subset.positives)
    assert out.vector_set.n_layers == 12


def test_topup_loop_generates_until_target_reached():
    # target_per_class drives a generate→filter loop: it should stop as soon as BOTH pools hit
    # the target, without consuming the whole topic bank.
    ensemble = Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")])
    cfg = ExperimentConfig(
        backend="mock", num_layers=8, hidden_dim=16, mock_creative_layer=4,
        generation=GenerationConfig(samples_per_prompt=1, max_new_tokens=240),
        filtering=FilterConfig(
            method="evaluator_stack", enable_judge=True, one_per_prompt=True,
            quality_floor=3.0, generator_family="qwen", stage1_floor_percentile=0.0,
            creative_quantile=0.9, neutral_quantile=0.1, length_match=False,
            target_per_class=3, topup_chunk_size=60, min_per_class=0,
        ),
    )
    backend = MockBackend(num_layers=8, hidden_dim=16, creative_layer=4, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:120])
    out = derive_vectors(
        backend, cfg, HeuristicScorer(), prompt_set=prompts,
        eval_scorers=_scorers(ensemble=ensemble),
    )
    assert out.subset.n_positive >= 3 and out.subset.n_negative >= 3
    # stopped early — did NOT generate the whole bank (120 topics x 2 framings = 240)
    assert out.n_examples < 240
    assert out.examples is not None  # full scored pool retained for saving


def test_topup_loop_exhausts_bank_without_raising():
    # The hardening fix: when the bank can't reach the target, derive must NOT raise (that's what
    # lost 6 hr of generation in v3 run #1) — it returns what it has so the pool can still be saved.
    cfg = ExperimentConfig(
        backend="mock", num_layers=8, hidden_dim=16, mock_creative_layer=4,
        generation=GenerationConfig(samples_per_prompt=1, max_new_tokens=240),
        filtering=FilterConfig(
            method="evaluator_stack", enable_judge=False, one_per_prompt=True,
            generator_family="qwen", stage1_floor_percentile=0.0,
            creative_quantile=0.9, neutral_quantile=0.1, length_match=False,
            target_per_class=9999, topup_chunk_size=10, min_per_class=0,
        ),
    )
    backend = MockBackend(num_layers=8, hidden_dim=16, creative_layer=4, seed=0)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:20])
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts, eval_scorers=_scorers())
    assert out.subset.n_positive >= 1 and out.subset.n_negative >= 1  # whatever the bank yielded
    assert out.examples is not None and out.n_examples == 40  # consumed all 20 topics x 2 framings
