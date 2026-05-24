"""Tests for the quality reward-model scorer.

The Skywork model is far too large to load here, so we inject ``score_fn`` and assert the
caching + model-selection behavior (the parts that are ours, not the model's).
"""

from __future__ import annotations

from creativity_evaluators.stage1.reward_model import (
    SKYWORK_27B,
    SKYWORK_8B,
    RewardModelScorer,
)
from creativity_evaluators.utils.caching import DiskCache


def test_returns_injected_score():
    s = RewardModelScorer(score_fn=lambda p, r: 0.7)
    assert s.score("prompt", "response") == 0.7


def test_caches_to_disk_and_avoids_recompute(tmp_path):
    calls = []

    def fn(p, r):
        calls.append((p, r))
        return 1.23

    s = RewardModelScorer(score_fn=fn, cache=DiskCache(tmp_path))
    first = s.score("p", "r")
    second = s.score("p", "r")
    assert first == second == 1.23
    assert len(calls) == 1  # second call was served from cache


def test_cache_key_depends_on_inputs(tmp_path):
    calls = []

    def fn(p, r):
        calls.append((p, r))
        return float(len(r))

    s = RewardModelScorer(score_fn=fn, cache=DiskCache(tmp_path))
    s.score("p", "aa")
    s.score("p", "bbb")  # different response -> different key -> recompute
    assert len(calls) == 2


def test_default_model_is_27b():
    assert RewardModelScorer(score_fn=lambda p, r: 0.0).active_model == SKYWORK_27B


def test_fallback_flag_selects_8b():
    s = RewardModelScorer(use_fallback=True, score_fn=lambda p, r: 0.0)
    assert s.active_model == SKYWORK_8B
