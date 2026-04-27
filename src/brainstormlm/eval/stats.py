"""Eval statistics: paired bootstrap shared by memory and ablation evals."""

from __future__ import annotations

import random
from collections.abc import Sequence


def paired_bootstrap_p_value(
    deltas: Sequence[float],
    *,
    n_iterations: int = 10_000,
    rng: random.Random | None = None,
) -> float:
    """One-sided paired bootstrap p-value for ``H1: mean(deltas) > 0``.

    Resamples ``deltas`` (centered under H0) and counts the fraction of
    resampled means at least as extreme as the observed mean. Empty input
    returns 1.0 (no evidence).
    """
    n = len(deltas)
    if n == 0:
        return 1.0
    rng = rng or random.Random()
    mean_obs = sum(deltas) / n
    centered = [d - mean_obs for d in deltas]
    geq = 0
    for _ in range(n_iterations):
        total = 0.0
        for _ in range(n):
            total += centered[rng.randrange(n)]
        if total / n >= mean_obs:
            geq += 1
    return geq / n_iterations
