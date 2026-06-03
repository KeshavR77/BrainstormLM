"""Refiner-as-diversity-critic (X4) — arXiv 2604.01029.

Given a set of candidate ideas, compute pairwise embedding similarity,
identify the LEAST-distinct idea (the one most similar to its neighbours),
and return a replacement *request* (NOT a rewrite). Authorship of new ideas
always stays with the generator (diverger), never the critic.

Reference: "Revision or Re-Solving?" (arXiv 2604.01029) — refiners that
rewrite ideas erase diversity; the critic's job is only to flag, not author.

Usage::

    from brainstormlm.generation.diversity_critic import find_least_distinct

    result = find_least_distinct(ideas, embedder=embed_fn)
    if result is not None:
        print(f"Replace idea at index {result.index}: {result.reason}")
        # Pass the request back to the generator; critic does NOT rewrite.

Inject ``embedder`` for tests (no API calls):

    result = find_least_distinct(["A", "B", "C"], embedder=fake_embedder)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Data model — no "replacement" field (design invariant)
# ---------------------------------------------------------------------------


@dataclass
class ReplacementRequest:
    """A request to replace an idea — the critic does NOT write the replacement.

    Attributes:
        index: 0-based position of the least-distinct idea in the input list.
        reason: brief human-readable explanation of why this idea was flagged.
        mean_sim_to_others: average cosine similarity of this idea to all others;
                            higher = more redundant.
    """

    index: int
    reason: str
    mean_sim_to_others: float


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------


def pairwise_similarity(ideas: list[str], *, embedder: EmbedCallable) -> np.ndarray:
    """Compute the (N, N) cosine-similarity matrix for a list of ideas.

    Returns a symmetric float32 matrix with 1.0 on the diagonal.
    Embedder must return L2-normalised vectors of shape (N, D).
    """
    if not ideas:
        return np.zeros((0, 0), dtype=np.float32)

    embs = embedder(ideas).astype(np.float32)  # (N, D) L2-normalised
    sim = embs @ embs.T                         # (N, N) cosine similarities
    # Clip to [-1, 1] to handle floating-point noise
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def mean_distinctness(ideas: list[str], *, embedder: EmbedCallable) -> float:
    """Return the mean pairwise *distinctness* (1 - cosine_sim) of the idea set.

    Higher = more diverse. Returns 0.0 for a single idea, 1.0 for perfectly
    orthogonal embeddings.
    """
    n = len(ideas)
    if n <= 1:
        return 0.0

    sim = pairwise_similarity(ideas, embedder=embedder)
    # Off-diagonal elements only
    mask = ~np.eye(n, dtype=bool)
    mean_sim = float(sim[mask].mean())
    return 1.0 - mean_sim


def find_least_distinct(
    ideas: list[str],
    *,
    embedder: EmbedCallable,
) -> ReplacementRequest | None:
    """Identify the idea most redundant with the rest of the set.

    The 'least distinct' idea is the one with the highest mean cosine
    similarity to all other ideas in the set.

    Args:
        ideas: list of idea strings to evaluate.
        embedder: callable ``(texts: list[str]) -> np.ndarray`` (L2-normalised).

    Returns:
        A ReplacementRequest with the 0-based index of the least-distinct idea
        and a brief reason string, or None if the set has fewer than 2 ideas.

    Note:
        The caller is responsible for replacing the idea by invoking the
        generator again. This function intentionally has no ``replacement``
        field — see arXiv 2604.01029 on why the critic must not re-author.
    """
    n = len(ideas)
    if n < 2:
        return None

    sim = pairwise_similarity(ideas, embedder=embedder)  # (N, N)

    # Mean similarity to *other* ideas (exclude self on diagonal)
    mean_sim_to_others = np.array([
        float((sim[i].sum() - sim[i, i]) / (n - 1))
        for i in range(n)
    ])

    least_distinct_idx = int(np.argmax(mean_sim_to_others))
    score = float(mean_sim_to_others[least_distinct_idx])

    reason = (
        f"Idea at index {least_distinct_idx} has a mean cosine similarity of "
        f"{score:.3f} to the other {n - 1} idea(s) — it is the least distinct "
        "in the set. Request the generator to produce a replacement on a "
        "different conceptual axis."
    )

    return ReplacementRequest(
        index=least_distinct_idx,
        reason=reason,
        mean_sim_to_others=score,
    )


__all__ = [
    "ReplacementRequest",
    "pairwise_similarity",
    "mean_distinctness",
    "find_least_distinct",
]
