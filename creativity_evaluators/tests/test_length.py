"""Tests for token counting and the length-matched contrast partition."""

from __future__ import annotations

from creativity_evaluators.types import EvalResult, LexicalResult, PromptCategory
from creativity_evaluators.utils.length import count_tokens, length_matched_partition


def _result(score: float, n_tokens: int, category: PromptCategory) -> EvalResult:
    return EvalResult(
        prompt="p",
        response="r",
        category=category,
        dsi=0.0,
        lexical=LexicalResult(0.0, 0.0, 0.0),
        ppl=0.0,
        reward=0.0,
        ensemble=None,
        creativity_score=score,
        n_tokens=n_tokens,
        stage1_only=True,
        timestamp="",
        config_hash="",
    )


def test_count_tokens_words_and_punctuation():
    assert count_tokens("a b c") == 3
    assert count_tokens("hello, world") == 3  # hello + , + world


def test_count_tokens_custom_tokenizer():
    assert count_tokens("anything here", tokenizer=lambda s: s.split()) == 2


def test_partition_selects_quintiles_and_length_matches():
    # creative pool: scores 1..10; top quintile -> scores 9,10
    creative = [_result(float(s), 0, PromptCategory.CREATIVE) for s in range(1, 11)]
    creative[9].n_tokens = 100  # score 10
    creative[8].n_tokens = 50  # score 9
    # neutral pool: scores 1..10; bottom quintile -> scores 1,2
    neutral = [_result(float(s), 0, PromptCategory.NEUTRAL) for s in range(1, 11)]
    neutral[0].n_tokens = 200  # score 1
    neutral[1].n_tokens = 102  # score 2

    cre, neu = length_matched_partition(creative, neutral, tolerance=0.10)
    # creative 100 matches neutral 102 (within 10%); creative 50 has no neutral in range -> dropped
    assert len(cre) == len(neu) == 1
    assert cre[0].n_tokens == 100
    assert neu[0].n_tokens == 102


def test_partition_empty_pools():
    cre, neu = length_matched_partition([], [])
    assert cre == [] and neu == []
