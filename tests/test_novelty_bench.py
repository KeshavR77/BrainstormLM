"""Tests for NoveltyBench-Distinct — north-star metric. Stubbed embeddings."""

from __future__ import annotations

import numpy as np

from brainstormlm.eval.novelty_bench import novelty_bench_distinct


def _stub_embed(monkeypatch, matrix: np.ndarray) -> None:
    import brainstormlm.eval.novelty_bench as nb

    monkeypatch.setattr(nb, "_embed", lambda texts: matrix[: len(texts)])


def test_keys_present(monkeypatch) -> None:
    _stub_embed(monkeypatch, np.eye(3, dtype=np.float32))
    result = novelty_bench_distinct(["a", "b", "c"])
    assert set(result.keys()) == {"n_distinct", "n_total", "distinct_ratio"}


def test_all_distinct(monkeypatch) -> None:
    _stub_embed(monkeypatch, np.eye(4, dtype=np.float32))
    result = novelty_bench_distinct(["a", "b", "c", "d"])
    assert result["n_distinct"] == 4
    assert result["n_total"] == 4
    assert result["distinct_ratio"] == 1.0


def test_full_collapse(monkeypatch) -> None:
    emb = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (5, 1))
    _stub_embed(monkeypatch, emb)
    result = novelty_bench_distinct(["x"] * 5)
    assert result["n_distinct"] == 1
    assert result["n_total"] == 5
    assert abs(result["distinct_ratio"] - 0.2) < 1e-9


def test_two_equivalence_classes(monkeypatch) -> None:
    blob_a = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
    blob_b = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    _stub_embed(monkeypatch, np.vstack([blob_a, blob_b]))
    result = novelty_bench_distinct(["a"] * 3 + ["b"] * 3)
    assert result["n_distinct"] == 2
    assert abs(result["distinct_ratio"] - (2 / 6)) < 1e-9


def test_empty_list() -> None:
    assert novelty_bench_distinct([]) == {
        "n_distinct": 0,
        "n_total": 0,
        "distinct_ratio": 0.0,
    }


def test_single_text(monkeypatch) -> None:
    _stub_embed(monkeypatch, np.array([[1.0, 0.0]], dtype=np.float32))
    result = novelty_bench_distinct(["only one"])
    assert result["n_distinct"] == 1
    assert result["distinct_ratio"] == 1.0


def test_threshold_sensitivity(monkeypatch) -> None:
    emb = np.array([[1.0, 0.0, 0.0], [0.9, 0.436, 0.0]], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _stub_embed(monkeypatch, emb)
    assert novelty_bench_distinct(["a", "b"], threshold=0.85)["n_distinct"] == 1
    assert novelty_bench_distinct(["a", "b"], threshold=0.95)["n_distinct"] == 2
