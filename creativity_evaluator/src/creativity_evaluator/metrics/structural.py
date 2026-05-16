"""
Embedding-based structural diversity metrics.

Math extracted from MAS_Diversity (Chen et al., ACL 2026):
  - compute_vendi_and_order.py  →  vendi_score, order_parameter
  - compute_vendi_extended.py   →  pairwise_cosine_distance

All functions expect L2-normalized embeddings of shape (n, d).
Normalization is the caller's responsibility (handled by embedders).
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist


def vendi_score(emb: np.ndarray) -> float:
    """
    Vendi Score: effective number of distinct concepts in the embedding space.
    Higher = more diverse. Minimum 1.0 (all identical), maximum n (all orthogonal).

    Computes exp(entropy of eigenvalue distribution of the Gram matrix).
    Requires: vendi-score>=0.0.3
    """
    from vendi_score import vendi  # type: ignore

    emb = emb.astype(np.float64)
    K = emb @ emb.T
    K = (K + K.T) / 2.0  # symmetrize against floating-point drift
    return float(vendi.score_K(K))


def order_parameter(emb: np.ndarray) -> float:
    """
    Order Parameter φ ∈ [-1, 1]: measures directional alignment of embeddings.
    0 = isotropically distributed (diverse), 1 = fully collapsed to one direction.

    Equivalent to the mean projection of all embeddings onto the mean embedding direction.
    """
    emb = emb.astype(np.float64)
    v_avg = emb.mean(axis=0)
    norm = np.linalg.norm(v_avg)
    if norm < 1e-12:
        return 0.0
    v_avg = v_avg / norm
    return float((emb @ v_avg).mean())


def pairwise_cosine_distance(emb: np.ndarray) -> float:
    """
    Mean pairwise cosine distance across all pairs.
    Higher = more spread in embedding space (more diverse).
    Range: [0, 2] for L2-normalized embeddings (0 = identical, 2 = antipodal).
    """
    if len(emb) < 2:
        return 0.0
    distances = pdist(emb.astype(np.float64), metric="cosine")
    return float(distances.mean())
