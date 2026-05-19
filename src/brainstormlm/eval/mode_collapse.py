"""Mode-Collapse@K — semantic-cluster count for repeated-prompt outputs.

For a single prompt regenerated K times, embed every output and greedy-cluster
them by cosine similarity. The number of resulting clusters is a direct
measure of how many distinct semantic modes the model produced.

Pure mode collapse → 1 cluster. Healthy diversity → K clusters. Reported as
both the raw cluster count and the *coverage ratio* clusters / K so values
are comparable across different K choices.

We use greedy single-pass clustering at a cosine threshold (matching the
existing ``_archetype_count`` helper in ``scripts/summarize_benchmark.py``)
rather than HDBSCAN because we want deterministic, interpretable behavior on
the degenerate edge cases (all-identical → 1 cluster, all-orthogonal → K
clusters).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_CLUSTER_THRESHOLD = 0.85


def _embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised OpenAI embeddings via the cached client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix


def cluster_count(
    embeddings: np.ndarray,
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> int:
    """Greedy cluster count: a new point joins the first centroid it matches
    (cosine ≥ threshold), otherwise becomes a new centroid.
    """
    n = len(embeddings)
    if n == 0:
        return 0
    if n == 1:
        return 1

    v = embeddings.astype(np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    centroids: list[np.ndarray] = []
    for i in range(n):
        matched = False
        for c in centroids:
            if float(np.dot(v[i], c)) >= threshold:
                matched = True
                break
        if not matched:
            centroids.append(v[i])
    return len(centroids)


def mode_collapse_at_k(
    texts: Sequence[str],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, float]:
    """Return {n_clusters, k, coverage_ratio} for a list of K outputs."""
    k = len(texts)
    if k == 0:
        return {"n_clusters": 0, "k": 0, "coverage_ratio": 0.0}
    emb = _embed(texts)
    n_clusters = cluster_count(emb, threshold=threshold)
    return {
        "n_clusters": int(n_clusters),
        "k": int(k),
        "coverage_ratio": float(n_clusters) / float(k),
    }


def mode_collapse_corpus(
    per_prompt_texts: Sequence[Sequence[str]],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, float]:
    """Mean cluster count + coverage ratio across multiple prompts."""
    results = [mode_collapse_at_k(t, threshold=threshold) for t in per_prompt_texts]
    if not results:
        return {"mean_n_clusters": 0.0, "mean_coverage_ratio": 0.0, "n_prompts": 0}
    return {
        "mean_n_clusters": float(np.mean([r["n_clusters"] for r in results])),
        "mean_coverage_ratio": float(np.mean([r["coverage_ratio"] for r in results])),
        "n_prompts": len(results),
    }
