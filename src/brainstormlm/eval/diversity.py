"""Diversity metrics for evaluating output variety across a set of text responses.

Metrics:
  - near_duplicate_rate: fraction of response pairs that are near-duplicates
    (cosine similarity >= threshold).
  - distinct_bigrams: ratio of unique bigrams to total bigrams (Distinct-2).
  - repetition_index: 1 - distinct_bigrams (fraction of repeated bigrams).
  - vendi_score: embedding-based diversity (Vendi Score, Friedman & Dieng 2022).

All functions accept a list of strings and return a float.
Embeddings are computed via BrainstormLM's cached OpenAI client so repeated
calls on the same text set are free.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Lexical metrics (no embeddings needed)
# ---------------------------------------------------------------------------


def _bigrams(text: str) -> list[tuple[str, str]]:
    tokens = re.findall(r"\w+", text.lower())
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def distinct_bigrams(texts: Sequence[str]) -> float:
    """Fraction of unique bigrams over total bigrams (Distinct-2).

    Returns 1.0 when there are no bigrams (degenerate input).
    """
    total: list[tuple[str, str]] = []
    for t in texts:
        total.extend(_bigrams(t))
    if not total:
        return 1.0
    return len(set(total)) / len(total)


def repetition_index(texts: Sequence[str]) -> float:
    """Fraction of non-unique bigrams (1 - Distinct-2).

    Returns 0.0 when there are no bigrams.
    """
    return 1.0 - distinct_bigrams(texts)


# ---------------------------------------------------------------------------
# Embedding-based metrics
# ---------------------------------------------------------------------------


def _embed(texts: Sequence[str]) -> np.ndarray:
    """Return L2-normalised embedding matrix shape (n, d) via cached OpenAI client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix  # already L2-normalised by embed()


def near_duplicate_rate(texts: Sequence[str], *, threshold: float = 0.92) -> float:
    """Fraction of response pairs with cosine similarity >= threshold.

    A rate near 0 means outputs are all distinct; near 1 means near-identical.
    """
    n = len(texts)
    if n < 2:
        return 0.0
    emb = _embed(texts)
    K = emb @ emb.T  # cosine similarity matrix (already L2-normalised)
    # Count strictly upper-triangle pairs above threshold.
    total_pairs = n * (n - 1) // 2
    dup_pairs = int(np.sum(np.triu(K >= threshold, k=1)))
    return dup_pairs / total_pairs


def vendi_score(texts: Sequence[str]) -> float:
    """Vendi Score: effective number of distinct outputs (Friedman & Dieng 2022).

    Uses cosine similarity of OpenAI embeddings as the kernel. A score of 1
    means all outputs are identical; higher values indicate more diversity.
    """
    from vendi_score import vendi  # already installed via shrooms-llm dep chain

    n = len(texts)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    emb = _embed(texts)
    K = (emb @ emb.T).astype(np.float64)
    # Vendi score_K expects a positive-semidefinite similarity matrix.
    return float(vendi.score_K(K))


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------


def diversity_report(texts: Sequence[str]) -> dict[str, float]:
    """Return all diversity metrics for a list of texts.

    Keys: distinct2, repetition_index, near_duplicate_rate, vendi_score.
    The near_duplicate_rate and vendi_score require network calls (cached).
    """
    d2 = distinct_bigrams(texts)
    rep = 1.0 - d2
    ndr = near_duplicate_rate(texts)
    vs = vendi_score(texts)
    return {
        "distinct2": round(d2, 6),
        "repetition_index": round(rep, 6),
        "near_duplicate_rate": round(ndr, 6),
        "vendi_score": round(vs, 6),
    }
