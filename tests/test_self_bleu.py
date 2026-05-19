"""Tests for Self-BLEU diversity metric (Zhu et al., 2018)."""

from __future__ import annotations

from brainstormlm.eval.self_bleu import self_bleu, self_bleu_corpus


def test_self_bleu_identical_texts_is_high() -> None:
    """When all outputs are identical, Self-BLEU ≈ 1.0 (maximum mode collapse)."""
    texts = ["the quick brown fox jumps over the lazy dog"] * 5
    score = self_bleu(texts, n=4)
    assert score > 0.9


def test_self_bleu_distinct_texts_is_low() -> None:
    """When outputs share no n-grams, Self-BLEU is low (high diversity)."""
    texts = [
        "alpha beta gamma delta epsilon zeta",
        "one two three four five six",
        "red orange yellow green blue purple",
        "winter spring summer autumn season change",
        "guitar piano violin trumpet drum bass",
    ]
    score = self_bleu(texts, n=4)
    assert score < 0.05


def test_self_bleu_single_text_returns_zero() -> None:
    """With <2 texts, Self-BLEU is undefined; return 0.0."""
    assert self_bleu(["only one"], n=4) == 0.0
    assert self_bleu([], n=4) == 0.0


def test_self_bleu_n_smoothing_handles_short_texts() -> None:
    """Short texts (fewer tokens than n) should not crash."""
    texts = ["hi there", "hello world", "good morning"]
    score = self_bleu(texts, n=4)
    assert 0.0 <= score <= 1.0


def test_self_bleu_corpus_aggregates_per_prompt() -> None:
    """self_bleu_corpus returns mean Self-BLEU across multiple prompts."""
    per_prompt = [
        ["the cat sat", "the cat sat", "the cat sat"],  # high
        ["winter sun", "purple moon", "red ocean"],  # low
    ]
    score = self_bleu_corpus(per_prompt, n=4)
    assert 0.0 <= score <= 1.0


def test_self_bleu_lower_means_more_diverse() -> None:
    """Identical set should score strictly higher than diverse set."""
    identical = ["the quick brown fox jumps"] * 5
    diverse = [
        "alpha beta gamma delta epsilon",
        "winter spring autumn summer rain",
        "guitar trumpet violin cello flute",
        "ocean mountain forest valley desert",
        "morning evening twilight midnight noon",
    ]
    assert self_bleu(identical, n=4) > self_bleu(diverse, n=4)
