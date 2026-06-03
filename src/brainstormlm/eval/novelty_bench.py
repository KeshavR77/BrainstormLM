"""NoveltyBench-Distinct — the north-star diversity metric.

Partitions a set of outputs into semantic equivalence classes and reports how
many distinct classes there are (Zhang et al. 2025, "NoveltyBench"). A set that
collapses to one idea scores 1; a set of N genuinely-distinct ideas scores N.

We approximate the paper's learned pairwise-equivalence classifier with greedy
cosine clustering over OpenAI embeddings (the shared
:func:`brainstormlm.eval.mode_collapse.cluster_labels` primitive). This keeps
the metric offline-testable and deterministic; swap in a trained equivalence
model later without changing the call site.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD, cluster_labels


def _embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised OpenAI embeddings via the cached client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix


def novelty_bench_distinct(
    texts: Sequence[str],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, float]:
    """Count semantic equivalence classes in ``texts``.

    Returns ``{n_distinct, n_total, distinct_ratio}`` where ``distinct_ratio``
    is ``n_distinct / n_total`` (1.0 = every output is its own class, lower =
    more collapse). Empty input → all zeros.
    """
    n_total = len(texts)
    if n_total == 0:
        return {"n_distinct": 0, "n_total": 0, "distinct_ratio": 0.0}
    labels = cluster_labels(_embed(texts), threshold=threshold)
    n_distinct = len(set(labels))
    return {
        "n_distinct": int(n_distinct),
        "n_total": int(n_total),
        "distinct_ratio": float(n_distinct) / float(n_total),
    }
