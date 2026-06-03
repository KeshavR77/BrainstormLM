"""Tests for novelty-vs-corpus — distance of each candidate to a prior corpus.

novelty_vs_corpus._embed is called twice per non-empty call (candidates, then
corpus), so tests use an alternating fake: odd calls return the candidate
matrix, even calls return the corpus matrix.
"""

from __future__ import annotations

import numpy as np

from brainstormlm.eval.novelty_corpus import novelty_vs_corpus


def _patch_alternating(monkeypatch, cand_m: np.ndarray, corp_m: np.ndarray) -> dict:
    import brainstormlm.eval.novelty_corpus as nc

    calls = {"i": 0}

    def fake(texts):
        calls["i"] += 1
        m = cand_m if calls["i"] % 2 == 1 else corp_m
        return m[: len(texts)]

    monkeypatch.setattr(nc, "_embed", fake)
    return calls


def test_keys_present(monkeypatch) -> None:
    _patch_alternating(
        monkeypatch,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    result = novelty_vs_corpus(["a"], ["b"])
    assert set(result.keys()) == {"per_candidate", "flags", "mean_novelty"}
    assert set(result["per_candidate"][0].keys()) == {"max_sim", "novelty", "flag"}


def test_duplicate_of_corpus_is_flagged(monkeypatch) -> None:
    _patch_alternating(
        monkeypatch,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = novelty_vs_corpus(["dup"], ["original"])
    c = result["per_candidate"][0]
    assert abs(c["max_sim"] - 1.0) < 1e-6
    assert abs(c["novelty"] - 0.0) < 1e-6
    assert c["flag"] is True


def test_orthogonal_to_corpus_is_novel(monkeypatch) -> None:
    _patch_alternating(
        monkeypatch,
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = novelty_vs_corpus(["fresh"], ["original"])
    c = result["per_candidate"][0]
    assert abs(c["max_sim"] - 0.0) < 1e-6
    assert abs(c["novelty"] - 1.0) < 1e-6
    assert c["flag"] is False


def test_mixed_flags_and_mean(monkeypatch) -> None:
    _patch_alternating(
        monkeypatch,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = novelty_vs_corpus(["dup", "fresh"], ["original"])
    assert result["flags"] == [True, False]
    assert abs(result["mean_novelty"] - 0.5) < 1e-6


def test_threshold_controls_flagging(monkeypatch) -> None:
    cand = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    corp = np.array([[0.9, 0.436, 0.0]], dtype=np.float32)
    corp = corp / np.linalg.norm(corp, axis=1, keepdims=True)
    _patch_alternating(monkeypatch, cand, corp)
    assert novelty_vs_corpus(["x"], ["y"], threshold=0.85)["flags"] == [True]
    assert novelty_vs_corpus(["x"], ["y"], threshold=0.95)["flags"] == [False]


def test_empty_corpus_everything_novel(monkeypatch) -> None:
    calls = _patch_alternating(
        monkeypatch,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    result = novelty_vs_corpus(["a", "b"], [])
    assert result["flags"] == [False, False]
    assert result["mean_novelty"] == 1.0
    assert all(c["novelty"] == 1.0 for c in result["per_candidate"])
    # empty corpus short-circuits before any embedding call
    assert calls["i"] == 0


def test_empty_candidates() -> None:
    assert novelty_vs_corpus([], ["a", "b"]) == {
        "per_candidate": [],
        "flags": [],
        "mean_novelty": 0.0,
    }
