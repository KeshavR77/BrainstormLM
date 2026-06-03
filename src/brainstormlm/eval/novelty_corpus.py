"""Novelty-vs-corpus — anti-homogenization guard against a reference corpus.

Given fresh candidates and a corpus of prior outputs, score each candidate by
its cosine distance to the *nearest* corpus item: ``novelty = 1 - max_sim``.
Candidates whose nearest-neighbour similarity meets ``threshold`` are flagged as
too close to something already produced — the signal that ideation is drifting
back toward the existing corpus rather than expanding it.

An empty corpus means nothing has been produced yet, so every candidate is
maximally novel (and the embedding calls are skipped entirely).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD


def _embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised OpenAI embeddings via the cached client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix


def novelty_vs_corpus(
    candidates: Sequence[str],
    corpus: Sequence[str],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Score each candidate's novelty relative to ``corpus``.

    Returns ``{per_candidate: [{max_sim, novelty, flag}], flags, mean_novelty}``.
    ``flag`` is True when the candidate is within ``threshold`` cosine of some
    corpus item (too close). Empty candidates → zeros; empty corpus → all novel.
    """
    cand = list(candidates)
    if not cand:
        return {"per_candidate": [], "flags": [], "mean_novelty": 0.0}

    corp = list(corpus)
    if not corp:
        per = [{"max_sim": 0.0, "novelty": 1.0, "flag": False} for _ in cand]
        return {"per_candidate": per, "flags": [False] * len(cand), "mean_novelty": 1.0}

    sims = _embed(cand) @ _embed(corp).T  # (n_cand, n_corpus)
    max_sim = sims.max(axis=1)
    novelty = np.clip(1.0 - max_sim, 0.0, None)
    flags = max_sim >= threshold

    per = [
        {
            "max_sim": float(max_sim[i]),
            "novelty": float(novelty[i]),
            "flag": bool(flags[i]),
        }
        for i in range(len(cand))
    ]
    return {
        "per_candidate": per,
        "flags": [bool(f) for f in flags],
        "mean_novelty": float(np.mean(novelty)),
    }
