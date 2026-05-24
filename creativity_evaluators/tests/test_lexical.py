"""Tests for lexical/syntactic redundancy (Shaib et al. 2024 + EQ-Bench v3 slop).

The slop sub-score needs no model and is the reliable half (weighted 2x), so it carries
the assertions. POS-template overlap needs spaCy; when spaCy is absent the scorer must
degrade gracefully (overlap 0.0, ``template_overlap_available=False``), which these tests
also assert so the suite passes with or without spaCy installed.
"""

from __future__ import annotations

from creativity_evaluators.stage1.lexical import LexicalScorer, slop_rate
from creativity_evaluators.stage2.slop_list import SLOP_PHRASES

SLOP_TEXT = (
    "In the realm of innovation, we must delve into the rich tapestry of ideas and "
    "navigate the complexities. It's worth noting that this is a testament to synergy."
)
CLEAN_TEXT = "The cat sat on a warm windowsill and watched two sparrows hop across the fence."


def test_slop_rate_high_for_sloppy_text():
    rate, hits = slop_rate(SLOP_TEXT, SLOP_PHRASES)
    assert rate > 0.1
    assert any("delve" in h for h in hits)


def test_slop_rate_zero_for_clean_text():
    rate, hits = slop_rate(CLEAN_TEXT, SLOP_PHRASES)
    assert rate == 0.0
    assert hits == []


def test_occurrence_counted_per_match():
    rate, hits = slop_rate("a tapestry and another tapestry", ["tapestry"])
    assert len(hits) == 2
    assert abs(rate - 2 / 5) < 1e-9  # 5 whitespace tokens, 2 hits


def test_scorer_combined_higher_for_slop():
    scorer = LexicalScorer()
    slop = scorer.score(SLOP_TEXT)
    clean = scorer.score(CLEAN_TEXT)
    assert slop.combined > clean.combined
    assert slop.slop_rate > clean.slop_rate


def test_clean_text_combined_near_zero():
    scorer = LexicalScorer()
    res = scorer.score(CLEAN_TEXT)
    assert res.combined < 0.05


def test_slop_hits_recorded():
    scorer = LexicalScorer()
    res = scorer.score(SLOP_TEXT)
    assert len(res.slop_hits) >= 3


def test_template_overlap_flag_is_bool():
    scorer = LexicalScorer()
    res = scorer.score(CLEAN_TEXT)
    assert isinstance(res.template_overlap_available, bool)
    # When unavailable, overlap must be a benign 0.0 (not None / not NaN).
    if not res.template_overlap_available:
        assert res.template_overlap == 0.0


def test_deterministic():
    scorer = LexicalScorer()
    assert scorer.score(SLOP_TEXT).combined == scorer.score(SLOP_TEXT).combined
