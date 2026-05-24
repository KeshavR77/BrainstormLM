"""Token-level perplexity from per-token negative log-likelihoods (V8.1 manifold audit)."""

from __future__ import annotations

import math
from typing import Sequence


def perplexity_stats_from_nlls(
    token_nlls: Sequence[float],
    *,
    sliding_window: int = 32,
) -> dict[str, float]:
    """
    ``token_nlls[i] = -log P(x_i | x_<i)`` under the sampler's distribution.

    Returns ``ppl_mean`` and the per-window extrema (V8.2.1):
    - ``ppl_sliding_max``: maximum window PPL (catches Condition A — noisy decode).
    - ``ppl_sliding_min``: minimum window PPL (catches Condition B — degenerate repetition;
      a loop like "the the the..." has window PPL near 1.0 which is invisible to the max-only check).
    """
    xs = [float(x) for x in token_nlls]
    if not xs:
        return {
            "ppl_mean": float("nan"),
            "ppl_sliding_max": float("nan"),
            "ppl_sliding_min": float("nan"),
        }
    ppl_mean = math.exp(sum(xs) / len(xs))
    w = max(1, int(sliding_window))
    w_eff = min(w, len(xs))
    sliding_max = 0.0
    sliding_min = float("inf")
    for i in range(0, len(xs) - w_eff + 1):
        chunk = xs[i : i + w_eff]
        ppl_win = math.exp(sum(chunk) / len(chunk))
        if ppl_win > sliding_max:
            sliding_max = ppl_win
        if ppl_win < sliding_min:
            sliding_min = ppl_win
    if sliding_max <= 0.0:
        sliding_max = ppl_mean
    if not math.isfinite(sliding_min) or sliding_min <= 0.0:
        sliding_min = ppl_mean
    return {
        "ppl_mean": float(ppl_mean),
        "ppl_sliding_max": float(sliding_max),
        "ppl_sliding_min": float(sliding_min),
    }
