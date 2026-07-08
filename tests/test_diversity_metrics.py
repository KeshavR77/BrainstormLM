"""Tests for diversity metrics — no network, no embeddings API."""

from __future__ import annotations

import numpy as np
import pytest

from brainstormlm.eval.diversity import (
    _bigrams,
    distinct_bigrams,
    diversity_report,
    near_duplicate_rate,
    repetition_index,
    vendi_score,
)


def _stub_embed(monkeypatch: pytest.MonkeyPatch, matrix: np.ndarray) -> None:
    """Patch _embed to return a fixed normalised matrix without hitting OpenAI."""
    import brainstormlm.eval.diversity as div_mod

    monkeypatch.setattr(div_mod, "_embed", lambda texts: matrix[: len(texts)])


# ---------------------------------------------------------------------------
# Lexical metrics (no stubs needed)
# ---------------------------------------------------------------------------


def test_bigrams_empty() -> None:
    assert _bigrams("") == []


def test_bigrams_single_word() -> None:
    assert _bigrams("hello") == []


def test_bigrams_two_words() -> None:
    assert _bigrams("hello world") == [("hello", "world")]


def test_distinct_bigrams_identical_texts() -> None:
    texts = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    d2 = distinct_bigrams(texts)
    assert d2 < 0.5  # lots of repeated bigrams across texts


def test_distinct_bigrams_all_unique() -> None:
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    d2 = distinct_bigrams(texts)
    assert d2 == 1.0  # all bigrams are distinct


def test_distinct_bigrams_empty_list() -> None:
    assert distinct_bigrams([]) == 1.0


def test_repetition_index_complements_distinct2() -> None:
    texts = ["one two three", "four five six"]
    assert abs(repetition_index(texts) + distinct_bigrams(texts) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Embedding-based metrics (stub _embed)
# ---------------------------------------------------------------------------


def test_near_duplicate_rate_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    rate = near_duplicate_rate(["a", "b", "c"], threshold=0.92)
    assert rate == 1.0  # all pairs are near-identical


def test_near_duplicate_rate_orthogonal(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    rate = near_duplicate_rate(["a", "b", "c"], threshold=0.92)
    assert rate == 0.0  # no near-duplicates


def test_near_duplicate_rate_single_text(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    assert near_duplicate_rate(["only one"]) == 0.0


def test_vendi_score_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    score = vendi_score(["a", "b"])
    assert abs(score - 1.0) < 0.01  # all identical → effective diversity = 1


def test_vendi_score_orthogonal(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    score = vendi_score(["a", "b"])
    assert score > 1.5  # orthogonal → higher diversity


def test_vendi_score_single() -> None:
    assert vendi_score(["just one"]) == 1.0


def test_vendi_score_empty() -> None:
    assert vendi_score([]) == 0.0


# ---------------------------------------------------------------------------
# diversity_report
# ---------------------------------------------------------------------------


def test_diversity_report_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _stub_embed(monkeypatch, emb)
    report = diversity_report(["the quick fox", "a lazy dog"])
    assert set(report.keys()) == {"distinct2", "repetition_index", "near_duplicate_rate", "vendi_score"}


def test_diversity_report_values_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    emb = np.array([[1.0, 0.0], [0.5, 0.866]], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _stub_embed(monkeypatch, emb)
    report = diversity_report(["one two three", "four five six"])
    assert 0.0 <= report["distinct2"] <= 1.0
    assert 0.0 <= report["repetition_index"] <= 1.0
    assert 0.0 <= report["near_duplicate_rate"] <= 1.0
    assert report["vendi_score"] >= 1.0
