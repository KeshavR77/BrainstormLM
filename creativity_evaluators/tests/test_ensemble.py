"""Tests for the 3-judge ensemble: family exclusion, aggregation math, order-invariance
(the single-scoring analogue of position-swap), and inter-judge Spearman + low-agreement
warning."""

from __future__ import annotations

import logging

from creativity_evaluators.stage2.ensemble import Ensemble, exclude_same_family
from creativity_evaluators.types import DIMENSIONS, EnsembleResult, JudgeResult


def _sd(o, f, fl, e, p) -> dict[str, int]:
    return dict(zip(DIMENSIONS, (o, f, fl, e, p)))


class FakeJudge:
    """Satisfies the Judge protocol; returns a fixed score dict."""

    def __init__(self, name: str, family: str, scores: dict[str, int]):
        self.name = name
        self.family = family
        self._scores = scores

    def score(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(scores=dict(self._scores), reasoning="", judge_name=self.name)


class BoomJudge:
    name = "boom"
    family = "z"

    def score(self, prompt: str, response: str) -> JudgeResult:
        raise RuntimeError("transient")


# ---------------------------------------------------------------------------
# Family exclusion
# ---------------------------------------------------------------------------


def test_exclude_same_family_helper():
    judges = [FakeJudge("a", "openai", _sd(3, 3, 3, 3, 3)), FakeJudge("b", "qwen", _sd(3, 3, 3, 3, 3))]
    kept = exclude_same_family(judges, "openai")
    assert [j.name for j in kept] == ["b"]


def test_ensemble_excludes_generator_family():
    judges = [
        FakeJudge("openai_j", "openai", _sd(4, 4, 4, 4, 4)),
        FakeJudge("anthropic_j", "anthropic", _sd(2, 2, 2, 2, 2)),
    ]
    res = Ensemble(judges, generator_family="openai").score("p", "r")
    assert res.n_judges_succeeded == 1
    assert all(jr.judge_name == "anthropic_j" for jr in res.per_judge)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_per_dimension_mean_averages_across_judges():
    j1 = FakeJudge("j1", "x", _sd(2, 2, 2, 2, 2))
    j2 = FakeJudge("j2", "y", _sd(4, 4, 4, 4, 4))
    res = Ensemble([j1, j2]).score("p", "r")
    assert isinstance(res, EnsembleResult)
    for dim in DIMENSIONS:
        assert res.per_dimension_mean[dim] == 3.0
    assert res.overall == 3.0


def test_overall_is_mean_of_dimension_means():
    res = Ensemble([FakeJudge("j", "x", _sd(1, 2, 3, 4, 5))]).score("p", "r")
    assert res.overall == 3.0  # mean(1,2,3,4,5)


def test_aggregation_is_order_invariant():
    # Mean aggregation removes any ordering effect — the single-scoring stand-in for the
    # pairwise position-swap bias control.
    j1 = FakeJudge("j1", "x", _sd(1, 2, 3, 4, 5))
    j2 = FakeJudge("j2", "y", _sd(5, 4, 3, 2, 1))
    a = Ensemble([j1, j2]).score("p", "r")
    b = Ensemble([j2, j1]).score("p", "r")
    assert a.per_dimension_mean == b.per_dimension_mean
    assert a.overall == b.overall


def test_failed_judge_skipped_and_counted():
    res = Ensemble([BoomJudge(), FakeJudge("g", "y", _sd(3, 3, 3, 3, 3))]).score("p", "r")
    assert res.n_judges_succeeded == 1
    assert res.per_judge[0].judge_name == "g"


# ---------------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------------


def test_perfect_agreement_spearman_is_one():
    j1 = FakeJudge("j1", "x", _sd(1, 2, 3, 4, 5))
    j2 = FakeJudge("j2", "y", _sd(1, 2, 3, 4, 5))
    res = Ensemble([j1, j2]).score("p", "r")
    assert abs(res.inter_judge_spearman - 1.0) < 1e-9


def test_low_agreement_logs_warning(caplog):
    j1 = FakeJudge("j1", "x", _sd(1, 2, 3, 4, 5))
    j2 = FakeJudge("j2", "y", _sd(5, 4, 3, 2, 1))  # reversed -> spearman -1
    with caplog.at_level(logging.WARNING):
        res = Ensemble([j1, j2], warn_threshold=0.5).score("p", "r")
    assert res.inter_judge_spearman < 0.5
    assert any("agree" in r.message.lower() for r in caplog.records)
