"""
Cluster displacement metrics for long-chain creative evaluation (Conceptial V8 exhaustion gauntlet).

Given a sequence of L2-normalized embeddings (one row per idea), compare how far the final idea
drifts from the centroid of the first *k* ideas in Euclidean space.
"""

from __future__ import annotations

import numpy as np


def initial_centroid(emb: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean embedding of the first ``k`` rows (shape ``(d,)``)."""
    if emb.ndim != 2:
        raise ValueError("emb must have shape (n, d)")
    n, _d = emb.shape
    if n < k:
        raise ValueError(f"need at least {k} embedding rows, got {n}")
    if k < 1:
        raise ValueError("k must be >= 1")
    return emb[:k].astype(np.float64).mean(axis=0)


def cluster_displacement_last_vs_initial_centroid(
    emb: np.ndarray,
    *,
    k_initial: int = 3,
    last_row_index: int = -1,
) -> float:
    """
    Cluster displacement :math:`D = \\|E_{\\text{last}} - C\\|_2` with :math:`C` the centroid of the
    first ``k_initial`` rows.

    For a 10-idea chain, pass ``emb`` with 10 rows and use defaults so the last row is the 10th idea
    and :math:`C` is the centroid of ideas 1–3 (0-based rows ``0:3``).
    """
    if emb.ndim != 2:
        raise ValueError("emb must have shape (n, d)")
    n = emb.shape[0]
    if n < k_initial:
        raise ValueError(f"need at least {k_initial} embedding rows, got {n}")
    idx = last_row_index + n if last_row_index < 0 else last_row_index
    if not 0 <= idx < n:
        raise ValueError(f"last_row_index {last_row_index} resolves to invalid row {idx} for n={n}")
    c = initial_centroid(emb, k_initial)
    last = emb[idx].astype(np.float64)
    return float(np.linalg.norm(last - c))
