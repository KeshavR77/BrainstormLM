"""Semantic anti-mode / embedding-repulsion selector.

Greedily pick up to ``n`` candidates that are mutually distinct *and* distinct
from a prior set of "stored" ideas (the anti-mode). A candidate is rejected if
its cosine similarity to any already-kept candidate, or to any prior idea,
reaches ``epsilon`` (a similarity *ceiling* — "within epsilon" means
``cos >= epsilon``). Candidates are visited best-first by ``quality`` so ties
resolve toward the higher-scoring idea.

Embeddings are resolved in priority order: an explicit ``candidate_embeddings``
matrix, then ``embed_fn(candidates)``, then the cached OpenAI client. Rows are
L2-normalised internally, so callers may pass raw native-latent hiddens (which
are not unit-norm) straight from the idea store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD


@dataclass(frozen=True)
class SelectionResult:
    kept_indices: list[int]
    kept_texts: list[str]
    dropped_indices: list[int]
    reasons: dict[int, str]  # dropped index → "near_prior" | "near_kept" | "budget_full"


def _embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised OpenAI embeddings via the cached client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    v = np.asarray(matrix, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def select_by_repulsion(
    candidates: Sequence[str],
    *,
    n: int,
    epsilon: float = DEFAULT_CLUSTER_THRESHOLD,
    prior_embeddings: np.ndarray | None = None,
    quality: Sequence[float] | None = None,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    candidate_embeddings: np.ndarray | None = None,
) -> SelectionResult:
    """Greedy max-spread selection with anti-mode repulsion.

    Returns at most ``n`` mutually-distinct candidates, dropping any within
    ``epsilon`` cosine of a kept candidate (``near_kept``) or a prior idea
    (``near_prior``), or once the budget is exhausted (``budget_full``).
    """
    cand = list(candidates)
    m = len(cand)
    if m == 0:
        return SelectionResult([], [], [], {})
    if n <= 0:
        reasons = {i: "budget_full" for i in range(m)}
        return SelectionResult([], [], list(range(m)), reasons)

    if candidate_embeddings is not None:
        emb = _normalize(candidate_embeddings)
    elif embed_fn is not None:
        emb = _normalize(embed_fn(cand))
    else:
        emb = _normalize(_embed(cand))

    prior = _normalize(prior_embeddings) if prior_embeddings is not None else None
    q = list(quality) if quality is not None else [0.0] * m
    order = sorted(range(m), key=lambda i: (-q[i], i))

    kept_indices: list[int] = []
    kept_vecs: list[np.ndarray] = []
    reasons: dict[int, str] = {}

    for i in order:
        if len(kept_indices) >= n:
            reasons[i] = "budget_full"
            continue
        if prior is not None and float(np.max(emb[i] @ prior.T)) >= epsilon:
            reasons[i] = "near_prior"
            continue
        if kept_vecs and float(np.max(emb[i] @ np.array(kept_vecs).T)) >= epsilon:
            reasons[i] = "near_kept"
            continue
        kept_indices.append(i)
        kept_vecs.append(emb[i])

    return SelectionResult(
        kept_indices=kept_indices,
        kept_texts=[cand[i] for i in kept_indices],
        dropped_indices=sorted(reasons),
        reasons=reasons,
    )
