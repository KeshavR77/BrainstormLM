"""Tests for Mode-Collapse@K metric — no network, stub embeddings."""

from __future__ import annotations

import numpy as np

from brainstormlm.eval.mode_collapse import (
    cluster_count,
    cluster_labels,
    mode_collapse_at_k,
    mode_collapse_corpus,
)


def _stub_embed(monkeypatch, matrix: np.ndarray) -> None:
    import brainstormlm.eval.mode_collapse as mc

    monkeypatch.setattr(mc, "_embed", lambda texts: matrix[: len(texts)])


def test_cluster_count_identical_embeddings_is_one() -> None:
    """All embeddings identical → one cluster."""
    emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1))
    assert cluster_count(emb) == 1


def test_cluster_count_orthogonal_embeddings_each_a_mode() -> None:
    """Orthogonal embeddings → K distinct modes."""
    emb = np.eye(4, dtype=np.float64)
    assert cluster_count(emb) == 4


def test_cluster_count_two_blobs() -> None:
    """Two tight clusters of three each → 2 clusters."""
    blob_a = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    blob_b = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
    emb = np.vstack([blob_a, blob_b])
    assert cluster_count(emb) == 2


def test_cluster_count_empty() -> None:
    assert cluster_count(np.zeros((0, 3))) == 0


def test_cluster_count_single_embedding() -> None:
    assert cluster_count(np.array([[1.0, 0.0]])) == 1


def test_cluster_count_threshold_controls_merging() -> None:
    """A higher threshold means stricter matching → more clusters."""
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.436, 0.0],  # cos ≈ 0.9 with the first
        ],
        dtype=np.float64,
    )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    assert cluster_count(emb, threshold=0.85) == 1
    assert cluster_count(emb, threshold=0.95) == 2


def test_mode_collapse_at_k_returns_keys(monkeypatch) -> None:
    emb = np.eye(5, dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    result = mode_collapse_at_k(["a", "b", "c", "d", "e"])
    assert set(result.keys()) == {"n_clusters", "k", "coverage_ratio"}
    assert result["k"] == 5
    assert 0.0 <= result["coverage_ratio"] <= 1.0


def test_mode_collapse_at_k_full_collapse(monkeypatch) -> None:
    """All-identical embeddings → coverage_ratio = 1/K."""
    emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (10, 1))
    _stub_embed(monkeypatch, emb)
    result = mode_collapse_at_k(["x"] * 10)
    assert result["n_clusters"] == 1
    assert abs(result["coverage_ratio"] - 0.1) < 1e-9


def test_mode_collapse_corpus_averages(monkeypatch) -> None:
    """Corpus aggregates one collapsed prompt and one fully diverse prompt."""
    collapsed = np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]), (5, 1))
    orthogonal = np.eye(5, dtype=np.float32)
    batches = [collapsed, orthogonal]
    calls = {"i": 0}

    def fake_embed(texts):
        out = batches[calls["i"]]
        calls["i"] += 1
        return out

    import brainstormlm.eval.mode_collapse as mc

    monkeypatch.setattr(mc, "_embed", fake_embed)
    summary = mode_collapse_corpus([["a"] * 5, ["b"] * 5])
    assert summary["n_prompts"] == 2
    # Collapsed prompt: 1/5 = 0.2. Orthogonal: 5/5 = 1.0. Mean = 0.6.
    assert abs(summary["mean_coverage_ratio"] - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# cluster_labels — per-point centroid assignment (shared primitive)
# ---------------------------------------------------------------------------


def test_cluster_labels_identical_all_zero() -> None:
    """Identical embeddings all join the first centroid → all label 0."""
    emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1))
    assert cluster_labels(emb) == [0, 0, 0, 0, 0]


def test_cluster_labels_orthogonal_each_new() -> None:
    """Orthogonal embeddings each open a new centroid, numbered in order."""
    emb = np.eye(4, dtype=np.float64)
    assert cluster_labels(emb) == [0, 1, 2, 3]


def test_cluster_labels_two_blobs() -> None:
    """Two tight blobs → contiguous label runs [0,0,0,1,1,1]."""
    blob_a = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    blob_b = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
    emb = np.vstack([blob_a, blob_b])
    assert cluster_labels(emb) == [0, 0, 0, 1, 1, 1]


def test_cluster_labels_empty() -> None:
    assert cluster_labels(np.zeros((0, 3))) == []


def test_cluster_labels_single() -> None:
    assert cluster_labels(np.array([[1.0, 0.0]])) == [0]


def test_cluster_labels_length_equals_n() -> None:
    emb = np.eye(5, dtype=np.float64)
    assert len(cluster_labels(emb)) == 5


def test_cluster_count_matches_distinct_labels() -> None:
    """cluster_count is exactly the number of distinct labels."""
    blob_a = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    blob_b = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
    emb = np.vstack([blob_a, blob_b])
    labels = cluster_labels(emb)
    assert cluster_count(emb) == len(set(labels))


def test_cluster_labels_threshold_controls_merging() -> None:
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.436, 0.0],  # cos ≈ 0.9 with the first
        ],
        dtype=np.float64,
    )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    assert cluster_labels(emb, threshold=0.85) == [0, 0]
    assert cluster_labels(emb, threshold=0.95) == [0, 1]
