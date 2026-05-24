"""Tests for length-normalized perplexity.

The gpt2-medium path needs a download, so we inject a fake ``nll_fn`` returning
``(total_nll, n_tokens)`` and test the length-normalization (the only non-trivial logic).
"""

from __future__ import annotations

import math

from creativity_evaluators.stage1.perplexity import PerplexityScorer
from creativity_evaluators.types import PPLResult


def _fake_nll(text: str):
    n = len(text.split())
    return 2.0 * n, n  # exactly 2.0 nll per token


def test_nll_per_token_is_length_normalized():
    res = PerplexityScorer(nll_fn=_fake_nll).score("one two three four")
    assert isinstance(res, PPLResult)
    assert abs(res.nll_per_token - 2.0) < 1e-9
    assert res.n_tokens == 4


def test_default_model_name():
    assert PerplexityScorer(nll_fn=_fake_nll).score("a b c").model == "gpt2-medium"


def test_custom_model_name_propagates():
    res = PerplexityScorer(model_name="gpt2", nll_fn=_fake_nll).score("a b c d")
    assert res.model == "gpt2"


def test_empty_text_is_nan():
    res = PerplexityScorer(nll_fn=lambda t: (0.0, 0)).score("")
    assert math.isnan(res.nll_per_token)
    assert res.n_tokens == 0


def test_deterministic():
    s = PerplexityScorer(nll_fn=_fake_nll)
    assert s.score("alpha beta gamma").nll_per_token == s.score("alpha beta gamma").nll_per_token
