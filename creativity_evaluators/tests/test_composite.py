"""Tests for the composite scorer.

Every z-score is computed against the current batch, so the composite is a batch operation.
With a 2-item batch each z is exactly +/-1, which lets us hand-compute an exact expected score.
"""

from __future__ import annotations

import math

from creativity_evaluators.composite import (
    compute_composite_batch,
    stage1_composite_batch,
    zscores,
)


def _rec(dsi, ppl, lex, reward, orig, prag, elab) -> dict:
    return dict(
        dsi=dsi,
        ppl=ppl,
        lexical_redundancy=lex,
        reward=reward,
        originality_s2=orig,
        pragmaticality_s2=prag,
        elaboration_s2=elab,
    )


# ---------------------------------------------------------------------------
# z-scoring
# ---------------------------------------------------------------------------


def test_zscores_population_standardization():
    z = zscores([1, 2, 3])
    assert abs(float(sum(z))) < 1e-9
    assert abs(z[2] - 1.224744871) < 1e-6


def test_zscores_impute_none_to_mean():
    z = zscores([1, None, 3])
    assert z[1] == 0.0
    assert abs(z[0] + 1.0) < 1e-9 and abs(z[2] - 1.0) < 1e-9


def test_zscores_constant_is_all_zero():
    assert list(zscores([5, 5, 5])) == [0, 0, 0]


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def test_two_item_exact_composite():
    # A dominates B on every signal -> A z=+1, B z=-1 everywhere (lexical is inverted).
    a = _rec(1, 1, 0, 1, 5, 5, 5)
    b = _rec(0, 0, 1, 0, 1, 1, 1)
    scores, _comps = compute_composite_batch([a, b])
    # A: creativity = sqrt(1*1)=1, elab z=+1 -> 0.85*1 + 0.15*1 = 1.0
    assert abs(scores[0] - 1.0) < 1e-9
    # B: creativity = sqrt(0*0)=0, elab z=-1 -> 0.85*0 + 0.15*(-1) = -0.15
    assert abs(scores[1] - (-0.15)) < 1e-9


def test_quality_negative_side_clips_creativity_to_zero():
    a = _rec(1, 1, 0, 0, 5, 1, 3)  # high novelty, low quality
    b = _rec(0, 0, 1, 1, 1, 5, 3)
    _scores, comps = compute_composite_batch([a, b])
    assert abs(comps[0]["creativity"]) < 1e-9  # quality_combined < 0 -> clipped


def test_raw_components_exposed():
    a = _rec(1, 1, 0, 1, 5, 5, 5)
    b = _rec(0, 0, 1, 0, 1, 1, 1)
    _scores, comps = compute_composite_batch([a, b])
    for key in ("novelty", "novelty_combined", "quality_combined", "creativity", "dsi_z"):
        assert key in comps[0]


def test_missing_dsi_does_not_crash():
    a = _rec(None, 1, 0, 1, 5, 5, 5)
    b = _rec(None, 0, 1, 0, 1, 1, 1)
    scores, _comps = compute_composite_batch([a, b])
    assert all(math.isfinite(s) for s in scores)


# ---------------------------------------------------------------------------
# Stage-1-only gating composite
# ---------------------------------------------------------------------------


def test_stage1_composite_is_monotonic():
    hi = _rec(1, 1, 0, 1, None, None, None)
    lo = _rec(0, 0, 1, 0, None, None, None)
    gate = stage1_composite_batch([hi, lo])
    assert gate[0] > gate[1]
