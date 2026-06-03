"""Tests for the semantic anti-mode / embedding-repulsion selector.

Most cases pass candidate_embeddings directly (deterministic, offline). One
case exercises the default _embed path via monkeypatch.
"""

from __future__ import annotations

import numpy as np

from brainstormlm.pipeline.select import SelectionResult, select_by_repulsion


def test_returns_selection_result() -> None:
    res = select_by_repulsion(["a", "b", "c"], n=3, candidate_embeddings=np.eye(3, dtype=np.float32))
    assert isinstance(res, SelectionResult)
    assert res.kept_indices == [0, 1, 2]
    assert res.kept_texts == ["a", "b", "c"]
    assert res.dropped_indices == []


def test_rejects_near_duplicate_keeps_distant() -> None:
    emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    res = select_by_repulsion(["x", "x2", "y"], n=3, candidate_embeddings=emb)
    assert res.kept_indices == [0, 2]
    assert res.reasons[1] == "near_kept"


def test_budget_full_when_n_smaller() -> None:
    res = select_by_repulsion(["a", "b", "c", "d"], n=2, candidate_embeddings=np.eye(4, dtype=np.float32))
    assert res.kept_indices == [0, 1]
    assert res.reasons[2] == "budget_full"
    assert res.reasons[3] == "budget_full"


def test_quality_orders_selection() -> None:
    res = select_by_repulsion(
        ["a", "b", "c"], n=2, quality=[0.1, 0.9, 0.5], candidate_embeddings=np.eye(3, dtype=np.float32)
    )
    assert res.kept_indices == [1, 2]


def test_keeps_higher_quality_of_duplicate() -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    res = select_by_repulsion(["lo", "hi"], n=2, quality=[0.2, 0.8], candidate_embeddings=emb)
    assert res.kept_indices == [1]
    assert res.reasons[0] == "near_kept"


def test_prior_repulsion() -> None:
    prior = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    res = select_by_repulsion(
        ["a", "b", "c"], n=3, prior_embeddings=prior, candidate_embeddings=np.eye(3, dtype=np.float32)
    )
    assert res.reasons[0] == "near_prior"
    assert set(res.kept_indices) == {1, 2}


def test_normalizes_unnormalized_hiddens() -> None:
    # rows 0 and 2 are colinear once normalized → 2 is a near-duplicate of 0.
    emb = np.array([[2.0, 0.0], [0.0, 3.0], [2.0, 0.01]], dtype=np.float32)
    res = select_by_repulsion(["a", "b", "c"], n=3, candidate_embeddings=emb)
    assert 0 in res.kept_indices
    assert res.reasons[2] == "near_kept"


def test_n_zero_keeps_nothing() -> None:
    res = select_by_repulsion(["a", "b"], n=0, candidate_embeddings=np.eye(2, dtype=np.float32))
    assert res.kept_indices == []


def test_empty_candidates() -> None:
    res = select_by_repulsion([], n=3)
    assert res.kept_indices == []
    assert res.dropped_indices == []


def test_default_embed_path(monkeypatch) -> None:
    import brainstormlm.pipeline.select as sel

    monkeypatch.setattr(sel, "_embed", lambda texts: np.eye(3, dtype=np.float32)[: len(texts)])
    res = select_by_repulsion(["a", "b", "c"], n=3)
    assert res.kept_indices == [0, 1, 2]
