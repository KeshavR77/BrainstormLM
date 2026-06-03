"""Tests for the A/B diversity validation harness analysis logic.

Exercises bootstrap_ci, verdict, coherence_rate_from_texts, quality_pass_rate,
score_all_texts (mock), and the MockGenerator — all without hitting a real model
or the OpenAI embeddings / judge API.

The metric pipeline (novelty_bench_distinct, conditional_vendi) is NOT called
here because it requires the OpenAI embeddings API and is covered by its own
dedicated test files (test_novelty_bench.py, test_conditional_vendi.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the scripts directory is importable
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from validate_diversity import (
    ArmResult,
    MockGenerator,
    bootstrap_ci,
    coherence_rate_from_texts,
    quality_pass_rate,
    score_all_texts,
    verdict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_arm_result(arm: str, n_briefs: int = 5, k: int = 4) -> ArmResult:
    """Build an ArmResult from the MockGenerator with deterministic seed."""
    gen = MockGenerator(arm=arm, seed=42)
    briefs = [f"brief {i}" for i in range(n_briefs)]
    texts_per_brief = [gen.generate_k(f"brief {i}", k) for i in range(n_briefs)]
    # Attach mock judge scores
    scores = score_all_texts(texts_per_brief, briefs, mock=True)
    return ArmResult(
        arm=arm,
        texts_per_brief=texts_per_brief,
        judge_scores_per_brief=scores,
    )


def _dummy_metric_fn(arm_result: ArmResult) -> dict[str, float]:
    """Deterministic fake metric: counts total texts as n_distinct proxy."""
    all_texts = arm_result.all_texts
    n = len(all_texts)
    qpr = (
        quality_pass_rate(arm_result.judge_scores_per_brief)
        if arm_result.judge_scores_per_brief is not None
        else float("nan")
    )
    return {
        "n_distinct": float(n) * 0.75,  # some diversity
        "distinct_ratio": 0.75,
        "mean_per_prompt_vendi": float(n) * 0.1 + 1.0,
        "salad_rate": 0.0,
        "coherence_rate": 1.0,
        "quality_pass_rate": qpr,
    }


def _constant_metric_fn(arm_result: ArmResult) -> dict[str, float]:
    """Returns the same value regardless of the sample — CI should be tight."""
    return {
        "n_distinct": 5.0,
        "distinct_ratio": 0.625,
        "mean_per_prompt_vendi": 2.5,
        "salad_rate": 0.0,
        "coherence_rate": 1.0,
        "quality_pass_rate": 0.9,
    }


# ---------------------------------------------------------------------------
# MockGenerator
# ---------------------------------------------------------------------------


class TestMockGenerator:
    def test_returns_exactly_k_texts(self):
        for arm in ["greedy", "temp", "antimode", "ortho", "ortho_temp", "ortho_L17", "vs", "repulsion", "full"]:
            gen = MockGenerator(arm=arm, seed=0)
            texts = gen.generate_k("test brief", k=8)
            assert len(texts) == 8, f"arm={arm} returned {len(texts)} texts"

    def test_greedy_all_identical(self):
        gen = MockGenerator(arm="greedy", seed=0)
        texts = gen.generate_k("some brief", k=4)
        assert len(set(texts)) == 1, "greedy should produce all-identical texts (mode collapse floor)"

    def test_antimode_all_distinct(self):
        gen = MockGenerator(arm="antimode", seed=0)
        texts = gen.generate_k("brief", k=6)
        # Should be fully distinct (mocked diverse pool)
        assert len(set(texts)) == len(texts), "antimode should produce distinct texts"

    def test_ortho_L17_arm_returns_diverse(self):
        """ortho_L17 mock should return fully diverse texts (same pool as ortho_temp)."""
        gen = MockGenerator(arm="ortho_L17", seed=0)
        texts = gen.generate_k("brief", k=6)
        assert len(texts) == 6
        assert len(set(texts)) == len(texts), "ortho_L17 mock should return distinct texts"

    def test_alpha_suffix_stripped(self):
        """MockGenerator strips '@alpha' suffix for arm dispatch."""
        gen = MockGenerator(arm="ortho_L17@1.5", seed=0)
        # Should dispatch as ortho_L17 without raising
        texts = gen.generate_k("brief", k=4)
        assert len(texts) == 4

    def test_unknown_arm_raises(self):
        gen = MockGenerator(arm="nonexistent", seed=0)
        with pytest.raises(ValueError, match="Unknown arm"):
            gen.generate_k("brief", k=4)

    def test_different_seeds_dont_change_arm_diversity_shape(self):
        """Seed changes content but not structural properties of the mock."""
        gen_a = MockGenerator(arm="greedy", seed=0)
        gen_b = MockGenerator(arm="greedy", seed=99)
        a = gen_a.generate_k("brief", k=4)
        b = gen_b.generate_k("brief", k=4)
        # Both still all-identical (seed affects brief suffix, not collapse)
        assert len(set(a)) == 1
        assert len(set(b)) == 1


# ---------------------------------------------------------------------------
# ArmResult
# ---------------------------------------------------------------------------


class TestArmResult:
    def test_all_texts_flattens_correctly(self):
        arm = ArmResult(
            arm="test",
            texts_per_brief=[["a", "b"], ["c", "d"], ["e", "f"]],
        )
        assert arm.all_texts == ["a", "b", "c", "d", "e", "f"]

    def test_empty_texts_per_brief(self):
        arm = ArmResult(arm="test", texts_per_brief=[])
        assert arm.all_texts == []

    def test_judge_scores_default_none(self):
        arm = ArmResult(arm="test", texts_per_brief=[["a"]])
        assert arm.judge_scores_per_brief is None


# ---------------------------------------------------------------------------
# coherence_rate_from_texts
# ---------------------------------------------------------------------------


class TestCoherenceRate:
    def test_clean_texts_all_coherent(self):
        texts = [
            "A well-formed idea about product naming for coffee.",
            "Another coherent concept for a sustainable brand.",
            "Third distinctive creative direction for packaging.",
        ]
        rate = coherence_rate_from_texts(texts)
        assert rate == 1.0

    def test_token_loop_counts_as_incoherent(self):
        # 4+ identical consecutive tokens trigger token_loop
        texts = [
            "good text here",
            "the the the the end",  # token_loop
        ]
        rate = coherence_rate_from_texts(texts)
        assert rate == 0.5  # 1 of 2 flagged

    def test_non_ascii_heavy_counts_as_incoherent(self):
        # > 5% non-ASCII characters triggers non_ascii
        texts = [
            "clean text here",
            "テキストテキストテキストテキスト",  # all non-ASCII
        ]
        rate = coherence_rate_from_texts(texts)
        assert rate == 0.5

    def test_empty_input(self):
        assert coherence_rate_from_texts([]) == 1.0

    def test_comma_bleed_does_not_affect_coherence_rate(self):
        """coherence_rate uses only token_loop + non_ascii, not comma_bleed."""
        texts = [
            "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p",  # comma_bleed but not token_loop
        ]
        rate = coherence_rate_from_texts(texts)
        # comma_bleed is excluded from coherence_rate by design
        assert rate == 1.0

    def test_micro_output_does_not_affect_coherence_rate(self):
        """coherence_rate uses only token_loop + non_ascii, not micro_output."""
        texts = ["tiny"]  # micro_output (< 20 tokens) but no loop/non_ascii
        rate = coherence_rate_from_texts(texts)
        assert rate == 1.0


# ---------------------------------------------------------------------------
# FIX (b): quality_pass_rate and score_all_texts (mock)
# ---------------------------------------------------------------------------


class TestQualityPassRate:
    def test_all_pass(self):
        scores = [[{"coherence": 4, "on_topic": 5}], [{"coherence": 3, "on_topic": 4}]]
        assert quality_pass_rate(scores) == 1.0

    def test_none_pass(self):
        scores = [[{"coherence": 2, "on_topic": 3}], [{"coherence": 1, "on_topic": 2}]]
        assert quality_pass_rate(scores) == 0.0

    def test_partial(self):
        scores = [
            [{"coherence": 4, "on_topic": 4}, {"coherence": 2, "on_topic": 3}],
        ]
        result = quality_pass_rate(scores)
        assert result == pytest.approx(0.5)

    def test_boundary_coherence_3(self):
        """coherence=3 is sufficient (>=3)."""
        scores = [[{"coherence": 3, "on_topic": 4}]]
        assert quality_pass_rate(scores) == 1.0

    def test_boundary_on_topic_4(self):
        """on_topic=4 is sufficient (>=4)."""
        scores = [[{"coherence": 5, "on_topic": 4}]]
        assert quality_pass_rate(scores) == 1.0

    def test_boundary_on_topic_3_fails(self):
        """on_topic=3 is NOT sufficient (needs >=4)."""
        scores = [[{"coherence": 5, "on_topic": 3}]]
        assert quality_pass_rate(scores) == 0.0

    def test_empty(self):
        assert quality_pass_rate([]) == 1.0


class TestScoreAllTexts:
    def test_mock_greedy_gets_low_scores(self):
        """Mock: 'Standard solution' texts get coherence=2, on_topic=2."""
        texts_per_brief = [["Standard solution for: some brief"]]
        briefs = ["some brief"]
        scores = score_all_texts(texts_per_brief, briefs, mock=True)
        assert len(scores) == 1
        assert len(scores[0]) == 1
        s = scores[0][0]
        assert s["coherence"] == 2 and s["on_topic"] == 2

    def test_mock_diverse_texts_get_high_scores(self):
        """Mock: normal texts get coherence=4, on_topic=4."""
        texts_per_brief = [["Aether Hush — biomimetic noise-canceling concept."]]
        briefs = ["Create a product name for headphones."]
        scores = score_all_texts(texts_per_brief, briefs, mock=True)
        s = scores[0][0]
        assert s["coherence"] == 4 and s["on_topic"] == 4

    def test_shape_preserved(self):
        n_briefs, k = 3, 4
        texts_per_brief = [["text"] * k for _ in range(n_briefs)]
        briefs = [f"brief {i}" for i in range(n_briefs)]
        scores = score_all_texts(texts_per_brief, briefs, mock=True)
        assert len(scores) == n_briefs
        for group in scores:
            assert len(group) == k


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_ci_for_all_metric_keys(self):
        arm = _make_arm_result("antimode", n_briefs=6, k=4)
        cis = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=200)
        expected_keys = {
            "n_distinct", "distinct_ratio", "mean_per_prompt_vendi",
            "salad_rate", "coherence_rate", "quality_pass_rate",
        }
        assert set(cis.keys()) == expected_keys

    def test_ci_is_tuple_of_two_floats(self):
        arm = _make_arm_result("antimode", n_briefs=5, k=4)
        cis = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=100)
        for key, (lo, hi) in cis.items():
            assert isinstance(lo, float), f"{key} lower bound is not float"
            assert isinstance(hi, float), f"{key} upper bound is not float"

    def test_lower_le_upper(self):
        arm = _make_arm_result("antimode", n_briefs=5, k=4)
        cis = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=200)
        for key, (lo, hi) in cis.items():
            if not (math.isnan(lo) or math.isnan(hi)):
                assert lo <= hi, f"{key}: lower={lo} > upper={hi}"

    def test_constant_metric_gives_tight_ci(self):
        """When the metric never varies across resamples, CI should collapse to a point."""
        arm = _make_arm_result("antimode", n_briefs=10, k=4)
        cis = bootstrap_ci(arm, _constant_metric_fn, n_resamples=500)
        for key, (lo, hi) in cis.items():
            if not (math.isnan(lo) or math.isnan(hi)):
                assert abs(hi - lo) < 1e-9, f"{key} CI should be zero-width for constant metric"

    def test_wider_ci_with_more_variance(self):
        """More variance in the sample → wider CI (probabilistic check over many resamples)."""
        rng_state = np.random.default_rng(7)

        def variable_metric(arm_result: ArmResult) -> dict[str, float]:
            # Returns a value proportional to a random draw
            return {
                "n_distinct": float(rng_state.uniform(1, 10)),
                "distinct_ratio": float(rng_state.uniform(0, 1)),
                "mean_per_prompt_vendi": float(rng_state.uniform(1, 5)),
                "salad_rate": float(rng_state.uniform(0, 0.1)),
                "coherence_rate": float(rng_state.uniform(0.8, 1.0)),
                "quality_pass_rate": float(rng_state.uniform(0.5, 1.0)),
            }

        arm = _make_arm_result("antimode", n_briefs=8, k=4)
        cis_variable = bootstrap_ci(arm, variable_metric, n_resamples=300)
        cis_constant = bootstrap_ci(arm, _constant_metric_fn, n_resamples=300)
        # Variable metric CI must be wider than constant for at least the main metrics
        assert (cis_variable["n_distinct"][1] - cis_variable["n_distinct"][0]) > \
               (cis_constant["n_distinct"][1] - cis_constant["n_distinct"][0])

    def test_seed_reproducibility(self):
        arm = _make_arm_result("antimode", n_briefs=6, k=4)
        cis_a = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=100, seed=42)
        cis_b = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=100, seed=42)
        for key in cis_a:
            assert cis_a[key] == cis_b[key], f"{key}: CIs differ between same-seed runs"

    def test_different_seeds_give_close_but_possibly_different_cis(self):
        arm = _make_arm_result("antimode", n_briefs=8, k=4)
        cis_a = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=500, seed=1)
        cis_b = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=500, seed=2)
        # They should be close (large n_resamples) but not necessarily identical
        # This just checks they run without error
        assert set(cis_a.keys()) == set(cis_b.keys())

    def test_judge_scores_propagate_through_bootstrap(self):
        """judge_scores_per_brief must be resampled together with texts_per_brief."""
        arm = _make_arm_result("antimode", n_briefs=6, k=4)
        cis = bootstrap_ci(arm, _dummy_metric_fn, n_resamples=100)
        lo, hi = cis["quality_pass_rate"]
        # With mock scores all high, QPR should be well above 0
        assert not math.isnan(lo), "quality_pass_rate CI should not be NaN when scores provided"
        assert hi > 0.0


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _make_verdict_inputs(
    arm_n_distinct: float = 6.0,
    arm_vendi: float = 3.0,
    arm_vendi_ci: tuple[float, float] = (2.5, 3.5),
    arm_coherence: float = 1.0,
    arm_salad: float = 0.0,
    arm_qpr: float = 0.90,
    base_n_distinct: float = 4.0,
    base_vendi: float = 2.0,
    base_vendi_ci: tuple[float, float] = (1.5, 2.5),
    base_coherence: float = 1.0,
    base_qpr: float = 0.90,
    k: int = 8,
):
    metrics = {
        "n_distinct": arm_n_distinct,
        "distinct_ratio": arm_n_distinct / k,
        "mean_per_prompt_vendi": arm_vendi,
        "salad_rate": arm_salad,
        "coherence_rate": arm_coherence,
        "quality_pass_rate": arm_qpr,
    }
    cis = {
        "n_distinct": (arm_n_distinct - 0.5, arm_n_distinct + 0.5),
        "distinct_ratio": (arm_n_distinct / k - 0.05, arm_n_distinct / k + 0.05),
        "mean_per_prompt_vendi": arm_vendi_ci,
        "salad_rate": (0.0, arm_salad + 0.01),
        "coherence_rate": (arm_coherence - 0.01, arm_coherence),
        "quality_pass_rate": (arm_qpr - 0.05, arm_qpr + 0.05),
    }
    baseline_metrics = {
        "n_distinct": base_n_distinct,
        "distinct_ratio": base_n_distinct / k,
        "mean_per_prompt_vendi": base_vendi,
        "salad_rate": 0.0,
        "coherence_rate": base_coherence,
        "quality_pass_rate": base_qpr,
    }
    baseline_cis = {
        "n_distinct": (base_n_distinct - 0.5, base_n_distinct + 0.5),
        "distinct_ratio": (base_n_distinct / k - 0.05, base_n_distinct / k + 0.05),
        "mean_per_prompt_vendi": base_vendi_ci,
        "salad_rate": (0.0, 0.01),
        "coherence_rate": (base_coherence - 0.01, base_coherence),
        "quality_pass_rate": (base_qpr - 0.05, base_qpr + 0.05),
    }
    return metrics, cis, baseline_metrics, baseline_cis


class TestVerdict:
    def test_win_when_all_criteria_met(self):
        # arm: n_distinct=6 >= base(4)+1=5 ✓; vendi arm_lo(2.5) > base_hi(2.5) → need strictly >
        # Use non-overlapping: arm_lo=3.0 > base_hi=2.5
        m, c, bm, bc = _make_verdict_inputs(
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "WIN", f"Expected WIN, got {v}. Notes: {notes}"

    def test_fail_nb_not_met(self):
        # arm n_distinct=4 < base(4)+1=5
        m, c, bm, bc = _make_verdict_inputs(
            arm_n_distinct=4.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "FAIL"
        assert any("NoveltyBench FAIL" in n for n in notes)

    def test_fail_vendi_overlapping_ci(self):
        # Vendi CIs overlap: arm [2.0, 3.0] vs base [2.2, 3.2]
        m, c, bm, bc = _make_verdict_inputs(
            arm_n_distinct=6.0,
            arm_vendi=2.5,
            arm_vendi_ci=(2.0, 3.0),
            base_vendi=2.7,
            base_vendi_ci=(2.2, 3.2),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "FAIL"
        assert any("Vendi FAIL" in n for n in notes)

    def test_fail_vendi_arm_lower_than_base(self):
        # Arm vendi lower than baseline
        m, c, bm, bc = _make_verdict_inputs(
            arm_n_distinct=6.0,
            arm_vendi=1.5,
            arm_vendi_ci=(1.0, 2.0),
            base_vendi=2.5,
            base_vendi_ci=(2.0, 3.0),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "FAIL"

    def test_void_coherence_drop(self):
        # Coherence drops 5pp (> 2pp threshold)
        m, c, bm, bc = _make_verdict_inputs(
            arm_coherence=0.93,
            base_coherence=1.0,
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "VOID"
        assert any("coherence_rate dropped" in n for n in notes)

    def test_void_salad_rate_high(self):
        m, c, bm, bc = _make_verdict_inputs(
            arm_salad=0.15,  # > 0.10 threshold
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "VOID"
        assert any("salad_rate" in n for n in notes)

    def test_void_quality_pass_rate_drop(self):
        """FIX (b): quality_pass_rate drop >5pp triggers VOID."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_qpr=0.80,     # arm QPR
            base_qpr=0.90,    # 10pp drop → VOID
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "VOID"
        assert any("quality_pass_rate dropped" in n for n in notes)

    def test_void_quality_pass_rate_under_5pp_is_not_void(self):
        """A drop clearly under 5pp (3pp) should NOT be void from QPR alone."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_qpr=0.87,
            base_qpr=0.90,    # 3pp drop — clearly below >5pp threshold
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        # Should NOT be VOID from QPR
        assert not any("quality_pass_rate dropped" in n for n in notes)

    def test_void_when_qpr_is_nan_skipped(self):
        """If QPR is NaN (judge not wired), the QPR void gate should be skipped."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_qpr=float("nan"),
            base_qpr=float("nan"),
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        # Should not be VOID due to QPR NaN
        assert not any("quality_pass_rate dropped" in n for n in notes)

    def test_void_takes_priority_over_win(self):
        """Even when diversity criteria are met, VOID if quality breaks."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_coherence=0.90,  # void trigger
            arm_n_distinct=8.0,  # would be a win
            arm_vendi=5.0,
            arm_vendi_ci=(4.0, 6.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, _ = verdict("test_arm", m, c, bm, bc, k=8)
        assert v == "VOID"

    def test_exactly_at_boundary_nb(self):
        """n_distinct = base + 1 exactly should be a win (>= not >)."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_n_distinct=5.0,
            base_n_distinct=4.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        nb_ok = any("NoveltyBench OK" in n for n in notes)
        assert nb_ok, f"n_distinct exactly at baseline+1 should pass NB criterion. Notes: {notes}"

    def test_coherence_drop_under_2pp_is_not_void(self):
        """A drop of < 2pp should not trigger void (threshold is strictly >2pp)."""
        m, c, bm, bc = _make_verdict_inputs(
            arm_coherence=0.985,
            base_coherence=1.00,  # 1.5pp drop — clearly below threshold
            arm_n_distinct=6.0,
            arm_vendi=3.5,
            arm_vendi_ci=(3.0, 4.0),
            base_vendi=2.0,
            base_vendi_ci=(1.5, 2.5),
        )
        v, _ = verdict("test_arm", m, c, bm, bc, k=8)
        assert v != "VOID", "A 1.5pp coherence drop should not trigger void (threshold is >2pp)"

    def test_notes_always_returned_as_list(self):
        m, c, bm, bc = _make_verdict_inputs()
        v, notes = verdict("test_arm", m, c, bm, bc, k=8)
        assert isinstance(notes, list)
        assert len(notes) >= 1
