"""Tests for Divergent Semantic Integration (Johnson et al. 2022).

The heavy path (BERT layers 6+7, mean-pooled) is not exercised here — that needs a model
download. Instead we test the pure pairwise-distance core directly and drive ``DSIScorer``
with an injected deterministic fake embedder, so the suite runs offline.
"""

from __future__ import annotations

import hashlib

import numpy as np

from creativity_evaluators.stage1.dsi import (
    DSIScorer,
    mean_pairwise_cosine_distance,
)
from creativity_evaluators.types import DSIResult


# --- a deterministic, offline stand-in for BERT: same token -> same vector ---
def _fake_embed(text: str):
    tokens = text.split()
    vecs = []
    for tok in tokens:
        seed = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vecs.append(rng.normal(size=16))
    emb = np.array(vecs, dtype=np.float64) if vecs else np.zeros((0, 16))
    return tokens, emb


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


def test_identical_rows_give_zero_distance():
    emb = np.ones((6, 8), dtype=np.float64)
    assert mean_pairwise_cosine_distance(emb) < 1e-9


def test_orthogonal_rows_give_distance_one():
    emb = np.eye(4, dtype=np.float64)  # rows are orthonormal -> cos sim 0 -> distance 1
    assert abs(mean_pairwise_cosine_distance(emb) - 1.0) < 1e-9


def test_varied_rows_strictly_positive():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(10, 8))
    assert mean_pairwise_cosine_distance(emb) > 0.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_identity_string_scores_zero():
    scorer = DSIScorer(embed_fn=_fake_embed)
    res = scorer.score("delve delve delve delve delve delve")
    assert isinstance(res, DSIResult)
    assert res.score is not None and res.score < 1e-9


def test_varied_prose_scores_positive():
    scorer = DSIScorer(embed_fn=_fake_embed)
    res = scorer.score("a curious otter juggled three luminous river stones at dawn")
    assert res.score is not None and res.score > 0.0


def test_too_short_returns_none():
    scorer = DSIScorer(embed_fn=_fake_embed, min_tokens=5)
    res = scorer.score("just three words")  # 3 tokens < 5
    assert res.score is None
    assert res.reason == "too_short"
    assert res.n_tokens == 3


def test_deterministic():
    scorer = DSIScorer(embed_fn=_fake_embed)
    text = "the inventor wired a kettle to whistle in morse code"
    assert scorer.score(text).score == scorer.score(text).score


def test_code_response_flagged_but_scored():
    scorer = DSIScorer(embed_fn=_fake_embed)
    code = "def add(a, b):\n    return a + b\n\nfor i in range(10):\n    print(add(i, i))"
    res = scorer.score(code)
    assert res.is_code is True
    assert res.score is not None  # still computes


def test_layer_indices_reported():
    scorer = DSIScorer(embed_fn=_fake_embed, layers=(6, 7))
    res = scorer.score("a curious otter juggled three luminous river stones at dawn")
    assert res.layer_indices == (6, 7)


def test_chunking_averages_chunk_scores():
    scorer_chunked = DSIScorer(embed_fn=_fake_embed, chunk_size=5)
    text = " ".join(f"word{i}" for i in range(20))  # 20 distinct tokens -> 4 chunks of 5
    res = scorer_chunked.score(text)
    assert res.n_chunks == 4
    # manual: mean of per-chunk DSI over the same embeddings
    _, emb = _fake_embed(text)
    manual = np.mean([mean_pairwise_cosine_distance(emb[i : i + 5]) for i in range(0, 20, 5)])
    assert abs(res.score - manual) < 1e-9
