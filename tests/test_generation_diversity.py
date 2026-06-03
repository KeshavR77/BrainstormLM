"""Tests for three generation-side diversity features (N1, N2, X4).

All tests are fully deterministic — LLM client and embedder are injected
as fakes; no API calls, no model downloads.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------


def _norm(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Fixed unit vectors for 3-D embedding space; each is orthogonal to the others
# except where we want near-duplicates.
EMBED_MAP: dict[str, np.ndarray] = {
    # Three clearly distinct angles
    "typography": _norm([1.0, 0.0, 0.0]),
    "mascot": _norm([0.0, 1.0, 0.0]),
    "abstract": _norm([0.0, 0.0, 1.0]),
    # Near-duplicate of "typography" (cosine sim ≈ 0.998)
    "typographic": _norm([1.0, 0.05, 0.0]),
    # Another near-duplicate of "typography"
    "letterform": _norm([1.0, 0.03, 0.0]),
    # Ideas for X4 critic tests
    "idea_A": _norm([1.0, 0.0, 0.0]),
    "idea_B": _norm([0.0, 1.0, 0.0]),
    "idea_C": _norm([1.0, 0.05, 0.0]),   # very close to idea_A → least distinct
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Return fixed L2-normalised vectors; unknown texts get a random-seed vector."""
    out = []
    for t in texts:
        if t in EMBED_MAP:
            out.append(EMBED_MAP[t])
        else:
            # Deterministic fallback based on hash
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(3).astype(np.float32)
            out.append(v / np.linalg.norm(v))
    return np.stack(out)


# ---------------------------------------------------------------------------
# N1 — Verbalized Sampling
# ---------------------------------------------------------------------------


class TestVerbalizedSampling:
    """Feature N1: one LLM call → N ideas with self-estimated probs; tail-sample."""

    # ---- fake LLM responses ------------------------------------------------

    CLEAN_RESPONSE = """
1. Blockchain loyalty (0.40)
2. Subscription bundle (0.25)
3. Community co-op (0.20)
4. Flash auction model (0.10)
5. Freemium tiers (0.05)
"""

    MESSY_RESPONSE = """
Here are my ideas with probability estimates:

1. **Blockchain loyalty** — probability: 40%
2. Subscription bundle (prob ≈ 25%)
3. community co-op [0.20]
4. Flash auction model — 10%
5. Freemium tiers: 5%
"""

    UNNORMALIZED_RESPONSE = """
1. Idea Alpha (0.50)
2. Idea Beta (0.50)
3. Idea Gamma (0.50)
"""

    def fake_llm(self, response_text: str) -> Callable:
        """Return a fake LLM callable that always returns the given text."""
        def _call(prompt: str, **kwargs) -> str:  # noqa: ARG001
            return response_text
        return _call

    def _module(self):
        from brainstormlm.generation import verbalized_sampling as vs
        return vs

    # ---- parsing -----------------------------------------------------------

    def test_parse_clean(self):
        vs = self._module()
        items = vs.parse_vs_response(self.CLEAN_RESPONSE)
        assert len(items) == 5
        texts = [i.text for i in items]
        assert "Blockchain loyalty" in texts
        probs = [i.prob for i in items]
        assert abs(sum(probs) - 1.0) < 0.01, "probabilities should sum to ~1"

    def test_parse_messy(self):
        """Parser handles varied formatting: %, 'prob ≈', [0.20], '— 10%'."""
        vs = self._module()
        items = vs.parse_vs_response(self.MESSY_RESPONSE)
        assert len(items) == 5
        probs = [i.prob for i in items]
        assert abs(sum(probs) - 1.0) < 0.01

    def test_parse_unnormalized_renormalizes(self):
        """Probabilities that don't sum to 1 are renormalized."""
        vs = self._module()
        items = vs.parse_vs_response(self.UNNORMALIZED_RESPONSE)
        probs = [i.prob for i in items]
        assert abs(sum(probs) - 1.0) < 0.01

    def test_parse_minimum_one_item(self):
        """Even a single-item response parses cleanly."""
        vs = self._module()
        items = vs.parse_vs_response("1. Solo idea (1.0)")
        assert len(items) == 1
        assert items[0].prob == pytest.approx(1.0, abs=0.01)

    # ---- tail sampling -----------------------------------------------------

    def test_tail_sample_never_returns_modal(self):
        """With floor=0 the modal candidate should be down-weighted to zero prob."""
        vs = self._module()
        items = vs.parse_vs_response(self.CLEAN_RESPONSE)
        # Modal = "Blockchain loyalty" (prob 0.40)
        modal = max(items, key=lambda x: x.prob).text
        rng = np.random.default_rng(42)
        results = [vs.tail_sample(items, rng=rng, modal_floor=0.0).text for _ in range(200)]
        assert modal not in results, "modal item should have zero weight at floor=0"

    def test_tail_sample_respects_distribution(self):
        """Lower-prob items should appear; total coverage > 1 distinct item."""
        vs = self._module()
        items = vs.parse_vs_response(self.CLEAN_RESPONSE)
        rng = np.random.default_rng(0)
        sampled = {vs.tail_sample(items, rng=rng, modal_floor=0.1).text for _ in range(500)}
        assert len(sampled) > 1, "tail sampling should yield more than one distinct idea"

    def test_full_pipeline_calls_llm_once(self):
        """verbalized_sample() issues exactly one LLM call and returns parsed items."""
        vs = self._module()
        call_count = [0]

        def counting_llm(prompt: str, **kwargs) -> str:  # noqa: ARG001
            call_count[0] += 1
            return self.CLEAN_RESPONSE

        items = vs.verbalized_sample("brainstorm business models", llm=counting_llm, n=5)
        assert call_count[0] == 1
        assert len(items) >= 1


# ---------------------------------------------------------------------------
# N2 — Planner → K orthogonal angles + dedupe
# ---------------------------------------------------------------------------


class TestPlannerAngles:
    """Feature N2: generate K angle strings, dedupe by embedding proximity."""

    ANGLE_RESPONSE_6 = "\n".join([
        "1. typographic letter-based wordmark",
        "2. mascot character",
        "3. abstract geometric shape",
        "4. heritage / vintage badge",
        "5. negative-space minimalist mark",
        "6. icon + wordmark lockup",
    ])

    def fake_llm(self, response_text: str) -> Callable:
        def _call(prompt: str, **kwargs) -> str:  # noqa: ARG001
            return response_text
        return _call

    def _module(self):
        from brainstormlm.generation import planner_angles as pa
        return pa

    # ---- parsing -----------------------------------------------------------

    def test_parse_returns_list_of_strings(self):
        pa = self._module()
        angles = pa.parse_angles(self.ANGLE_RESPONSE_6)
        assert isinstance(angles, list)
        assert all(isinstance(a, str) for a in angles)
        assert len(angles) == 6

    def test_parse_strips_numbering(self):
        pa = self._module()
        angles = pa.parse_angles(self.ANGLE_RESPONSE_6)
        for a in angles:
            assert not a[0].isdigit(), f"Angle should not start with a digit: {a!r}"

    # ---- deduplication -------------------------------------------------------

    def test_dedupe_removes_near_duplicates(self):
        """Angles with cos-sim > threshold are dropped; orthogonal ones kept."""
        pa = self._module()
        # "typographic", "typography", "letterform" are near-duplicates in EMBED_MAP;
        # "mascot" and "abstract" are orthogonal.
        angles = ["typography", "typographic", "letterform", "mascot", "abstract"]
        kept = pa.dedupe_angles(angles, embedder=fake_embedder, threshold=0.95)
        kept_set = set(kept)
        # The mascot and abstract should survive
        assert "mascot" in kept_set
        assert "abstract" in kept_set
        # At most one of the near-duplicate trio should remain
        near_dup_kept = kept_set & {"typography", "typographic", "letterform"}
        assert len(near_dup_kept) <= 1, f"Near-duplicate trio over-represented: {near_dup_kept}"

    def test_dedupe_keeps_all_when_all_distinct(self):
        pa = self._module()
        angles = ["typography", "mascot", "abstract"]
        kept = pa.dedupe_angles(angles, embedder=fake_embedder, threshold=0.95)
        assert set(kept) == {"typography", "mascot", "abstract"}

    def test_dedupe_preserves_order(self):
        """First-encountered angle wins; duplicates are dropped, not reordered."""
        pa = self._module()
        angles = ["mascot", "abstract", "typography", "typographic"]
        kept = pa.dedupe_angles(angles, embedder=fake_embedder, threshold=0.95)
        assert kept[0] == "mascot"
        assert kept[1] == "abstract"
        assert kept[2] == "typography"
        assert "typographic" not in kept

    # ---- full pipeline -------------------------------------------------------

    def test_plan_angles_returns_k_or_fewer(self):
        pa = self._module()
        angles = pa.plan_angles(
            brief="Design a logo for a tech startup",
            k=6,
            llm=self.fake_llm(self.ANGLE_RESPONSE_6),
            embedder=fake_embedder,
            dedupe_threshold=0.95,
        )
        assert 1 <= len(angles) <= 6

    def test_plan_angles_fewer_after_dedupe(self):
        """When LLM returns near-duplicates, the returned list is shorter than k."""
        pa = self._module()
        # All near-duplicates of "typography" plus two distinct ones
        response = "\n".join([
            "1. typography approach",
            "2. typographic approach",
            "3. letterform approach",
            "4. mascot approach",
            "5. abstract approach",
        ])
        # Map the exact strings to near-duplicate vectors
        custom_map = {
            "typography approach": _norm([1.0, 0.0, 0.0]),
            "typographic approach": _norm([1.0, 0.04, 0.0]),
            "letterform approach": _norm([1.0, 0.03, 0.0]),
            "mascot approach": _norm([0.0, 1.0, 0.0]),
            "abstract approach": _norm([0.0, 0.0, 1.0]),
        }

        def custom_embedder(texts):
            out = []
            for t in texts:
                v = custom_map.get(t, _norm([0.5, 0.5, 0.0]))
                out.append(v)
            return np.stack(out)

        angles = pa.plan_angles(
            brief="Logo design",
            k=5,
            llm=self.fake_llm(response),
            embedder=custom_embedder,
            dedupe_threshold=0.95,
        )
        # 3 near-dups → only first survives; 2 orthogonal → survive; total ≤ 3
        assert len(angles) <= 3


# ---------------------------------------------------------------------------
# X4 — Refiner-as-diversity-critic
# ---------------------------------------------------------------------------


class TestDiversityCritic:
    """Feature X4: score pairwise distinctness; return least-distinct idea index."""

    IDEAS_DIVERSE = ["idea_A", "idea_B"]   # orthogonal in EMBED_MAP
    IDEAS_WITH_CLONE = ["idea_A", "idea_B", "idea_C"]   # idea_C ≈ idea_A

    def _module(self):
        from brainstormlm.generation import diversity_critic as dc
        return dc

    # ---- pairwise matrix ---------------------------------------------------

    def test_pairwise_sim_shape(self):
        dc = self._module()
        sim = dc.pairwise_similarity(self.IDEAS_DIVERSE, embedder=fake_embedder)
        assert sim.shape == (2, 2)

    def test_pairwise_sim_diagonal_is_one(self):
        dc = self._module()
        sim = dc.pairwise_similarity(self.IDEAS_WITH_CLONE, embedder=fake_embedder)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_pairwise_sim_symmetric(self):
        dc = self._module()
        sim = dc.pairwise_similarity(self.IDEAS_WITH_CLONE, embedder=fake_embedder)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    # ---- least-distinct detection ------------------------------------------

    def test_finds_least_distinct_idea(self):
        """idea_C is very close to idea_A → it should be flagged as least distinct."""
        dc = self._module()
        result = dc.find_least_distinct(self.IDEAS_WITH_CLONE, embedder=fake_embedder)
        assert result.index == 2, (
            f"Expected idea_C (index 2) to be least distinct, got index {result.index}"
        )

    def test_returns_replace_request_not_rewrite(self):
        """The critic must NOT return a rewritten idea — only a replacement request."""
        dc = self._module()
        result = dc.find_least_distinct(self.IDEAS_WITH_CLONE, embedder=fake_embedder)
        # The result should have an index and a reason, but no rewrite
        assert hasattr(result, "index")
        assert hasattr(result, "reason")
        assert not hasattr(result, "replacement"), (
            "Critic must not generate replacements — authorship stays in the generator"
        )

    def test_diverse_set_score_high(self):
        """A set of orthogonal ideas should have a high mean distinctness score."""
        dc = self._module()
        score = dc.mean_distinctness(self.IDEAS_DIVERSE, embedder=fake_embedder)
        assert score > 0.5

    def test_cloned_set_score_lower(self):
        """Adding idea_C (clone of idea_A) should lower the distinctness score."""
        dc = self._module()
        score_clean = dc.mean_distinctness(self.IDEAS_DIVERSE, embedder=fake_embedder)
        score_with_clone = dc.mean_distinctness(self.IDEAS_WITH_CLONE, embedder=fake_embedder)
        assert score_with_clone < score_clean

    def test_single_idea_returns_none(self):
        """Can't compute distinctness for a single idea — should return None gracefully."""
        dc = self._module()
        result = dc.find_least_distinct(["idea_A"], embedder=fake_embedder)
        assert result is None

    def test_two_ideas_returns_lower_scoring(self):
        """With two ideas, the one more similar to the other is flagged."""
        dc = self._module()
        # idea_A and idea_C are near-duplicates; both are in the pair
        result = dc.find_least_distinct(["idea_A", "idea_C"], embedder=fake_embedder)
        # Both are equally non-distinct — either could be flagged, but result must not be None
        assert result is not None
        assert result.index in (0, 1)
