"""Paired-bootstrap p-value: edge cases + sanity checks."""

from __future__ import annotations

import random

from brainstormlm.eval.stats import paired_bootstrap_p_value


def test_empty_returns_one() -> None:
    assert paired_bootstrap_p_value([]) == 1.0


def test_clear_positive_signal_low_p() -> None:
    deltas = [1.0] * 30  # every prompt favors the treatment arm
    p = paired_bootstrap_p_value(deltas, n_iterations=2000, rng=random.Random(0))
    assert p < 0.05


def test_zero_mean_high_p() -> None:
    deltas = [-1.0, 1.0] * 25
    p = paired_bootstrap_p_value(deltas, n_iterations=2000, rng=random.Random(0))
    # Symmetric around 0 → roughly half the resamples ≥ observed.
    assert 0.3 < p < 0.7


def test_clear_negative_signal_high_p() -> None:
    deltas = [-1.0] * 30
    p = paired_bootstrap_p_value(deltas, n_iterations=2000, rng=random.Random(0))
    assert p > 0.95


def test_seeded_rng_is_deterministic() -> None:
    deltas = [0.3, -0.1, 0.5, 0.2, 0.0, 0.4, -0.2, 0.6, 0.1, 0.3]
    p1 = paired_bootstrap_p_value(deltas, n_iterations=1000, rng=random.Random(42))
    p2 = paired_bootstrap_p_value(deltas, n_iterations=1000, rng=random.Random(42))
    assert p1 == p2
