"""Cosine analysis — layer-wise direction geometry.

Two questions, both from the conformity paper's §4:

  * Which layers point the *same* way? A contiguous band of mutually-aligned DiffMeans
    vectors marks where the trait's direction is stable. We reproduce the paper's
    per-layer top-k neighbour cosine ``s_L`` and detect the high-cosine band.
  * Which layers point the *furthest apart*? The smallest pairwise cosines flag layers
    whose directions are unrelated (often early layers vs the trait band).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from creativity_steering.derivation.diffmeans import unit_normalize


def pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """``(n_layers, n_layers)`` cosine-similarity matrix between unit layer directions."""
    u = unit_normalize(np.asarray(vectors, dtype=np.float32))
    return u @ u.T


def top_k_neighbor_cosine(vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """Paper's ``s_L``: mean cosine to the ``k`` most-similar *other* layers, per layer."""
    sims = pairwise_cosine_matrix(vectors)
    n = sims.shape[0]
    np.fill_diagonal(sims, -np.inf)  # exclude self-comparison
    kk = min(k, n - 1)
    if kk <= 0:
        return np.zeros(n, dtype=np.float32)
    # top-kk per row
    topk = np.sort(sims, axis=1)[:, -kk:]
    return topk.mean(axis=1).astype(np.float32)


def furthest_layer_pairs(vectors: np.ndarray, n_pairs: int = 5) -> list[tuple[int, int, float]]:
    """The ``n_pairs`` layer pairs with the smallest cosine (directions furthest apart)."""
    sims = pairwise_cosine_matrix(vectors)
    n = sims.shape[0]
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sims[i, j])))
    pairs.sort(key=lambda t: t[2])
    return pairs[:n_pairs]


@dataclass
class CosineAnalysis:
    matrix: np.ndarray  # (L, L)
    neighbor_cosine: np.ndarray  # (L,) s_L
    band: tuple[int, int]  # inclusive [start, end] high-cosine band
    furthest_pairs: list[tuple[int, int, float]]

    def to_dict(self) -> dict:
        return {
            "neighbor_cosine": [float(x) for x in self.neighbor_cosine],
            "band": list(self.band),
            "furthest_pairs": [[i, j, c] for (i, j, c) in self.furthest_pairs],
        }


def detect_high_cosine_band(neighbor_cosine: np.ndarray) -> tuple[int, int]:
    """Longest contiguous run of layers whose ``s_L`` is above the global mean."""
    s = np.asarray(neighbor_cosine)
    above = s > s.mean()
    best = (int(np.argmax(s)), int(np.argmax(s)))  # fallback: the single best layer
    best_len = 0
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best_len, best = i - start, (start, i - 1)
            start = None
    if start is not None and len(above) - start > best_len:
        best = (start, len(above) - 1)
    return best


def analyze_cosine(vectors: np.ndarray, *, k: int = 5, n_furthest: int = 5) -> CosineAnalysis:
    matrix = pairwise_cosine_matrix(vectors)
    s = top_k_neighbor_cosine(vectors, k=k)
    return CosineAnalysis(
        matrix=matrix,
        neighbor_cosine=s,
        band=detect_high_cosine_band(s),
        furthest_pairs=furthest_layer_pairs(vectors, n_pairs=n_furthest),
    )
