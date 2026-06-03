"""Tests for X1: per-sample orthogonal steering (STARS-style).

All tests are pure-geometry on small CPU tensors.  No model or GPU required.
We test:
  - stiefel_basis returns K orthonormal directions (Gram matrix ≈ I_K)
  - each direction has unit L2 norm
  - shape contract: (K, hidden_dim)
  - K==1 recovers a unit vector collinear with the base vector
  - K > hidden_dim raises an error
  - inject_orthogonal_batch applies the right per-sample direction and preserves rescale math

We deliberately do NOT assert semantic outcomes — steering vectors are non-identifiable
(arXiv:2602.06801): orthogonal hidden-space directions do not guarantee orthogonal outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from creativity_steering.patching.orthogonal import (
    distinctness_order,
    inject_orthogonal_batch,
    orthonormalize_rows,
    stiefel_basis,
)


# ---------------------------------------------------------------------------
# stiefel_basis geometry tests
# ---------------------------------------------------------------------------

class TestStiefelBasis:
    def test_shape(self):
        base = np.random.default_rng(0).normal(size=32).astype(np.float32)
        dirs = stiefel_basis(base, k=4, seed=42)
        assert dirs.shape == (4, 32)

    def test_each_direction_unit_norm(self):
        base = np.random.default_rng(1).normal(size=64).astype(np.float32)
        dirs = stiefel_basis(base, k=6, seed=7)
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)

    def test_gram_matrix_is_identity(self):
        """All K directions must be mutually orthonormal."""
        base = np.random.default_rng(2).normal(size=64).astype(np.float32)
        k = 5
        dirs = stiefel_basis(base, k=k, seed=99)
        gram = dirs @ dirs.T  # (K, K)
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-5)

    def test_k1_aligns_with_base(self):
        """K=1 should return the normalised base vector."""
        base = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        dirs = stiefel_basis(base, k=1, seed=0)
        assert dirs.shape == (1, 3)
        expected = base / np.linalg.norm(base)
        # direction is either +base or -base; accept both
        assert (
            np.allclose(dirs[0], expected, atol=1e-5)
            or np.allclose(dirs[0], -expected, atol=1e-5)
        )

    def test_different_seeds_give_different_bases(self):
        base = np.ones(32, dtype=np.float32)
        d1 = stiefel_basis(base, k=3, seed=0)
        d2 = stiefel_basis(base, k=3, seed=1)
        # Rows 1..K-1 (the random perturbations) should differ; row 0 is always
        # aligned with base so may coincide — compare the full matrix.
        assert not np.allclose(d1, d2, atol=1e-3)

    def test_k_exceeds_hidden_dim_raises(self):
        base = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError):
            stiefel_basis(base, k=5)

    def test_zero_base_raises(self):
        with pytest.raises(ValueError):
            stiefel_basis(np.zeros(8, dtype=np.float32), k=2)

    def test_large_k_equal_to_hidden_dim(self):
        """K == hidden_dim is the full orthogonal group — should still succeed."""
        d = 8
        base = np.random.default_rng(3).normal(size=d).astype(np.float32)
        dirs = stiefel_basis(base, k=d, seed=5)
        gram = dirs @ dirs.T
        np.testing.assert_allclose(gram, np.eye(d), atol=1e-4)

    def test_float64_input_is_cast(self):
        base = np.random.default_rng(4).normal(size=16).astype(np.float64)
        dirs = stiefel_basis(base, k=3, seed=0)
        assert dirs.dtype == np.float32


# ---------------------------------------------------------------------------
# inject_orthogonal_batch tests
# ---------------------------------------------------------------------------

class TestInjectOrthogonalBatch:
    """Test that inject_orthogonal_batch applies the right per-row direction."""

    def _make_dirs(self, k: int, hidden: int, seed: int = 0) -> np.ndarray:
        base = np.random.default_rng(seed).normal(size=hidden).astype(np.float32)
        return stiefel_basis(base, k=k, seed=seed)

    def test_output_shape_unchanged(self):
        batch, seq, hidden = 3, 5, 32
        hidden_states = torch.randn(batch, seq, hidden)
        dirs = self._make_dirs(k=3, hidden=hidden)
        out = inject_orthogonal_batch(hidden_states, dirs, alpha=1.0, rescale=False)
        assert out.shape == hidden_states.shape

    def test_each_sample_receives_its_own_direction_no_rescale(self):
        """With rescale=False, sample i should shift by alpha * dirs[i] (broadcast over seq)."""
        batch, seq, hidden = 4, 1, 16
        hidden_states = torch.zeros(batch, seq, hidden)
        dirs = self._make_dirs(k=4, hidden=hidden)
        alpha = 2.0
        out = inject_orthogonal_batch(hidden_states, dirs, alpha=alpha, rescale=False)
        for i in range(batch):
            expected = alpha * torch.tensor(dirs[i])
            np.testing.assert_allclose(
                out[i, 0].numpy(), expected.numpy(), atol=1e-5,
                err_msg=f"sample {i} did not receive its direction"
            )

    def test_rescale_matches_existing_convention(self):
        """With rescale=True: delta = (v/|v|) * |h_token| * alpha.
        Because dirs are already unit-norm, the formula simplifies to delta = dirs[i] * |h| * alpha.
        """
        batch, seq, hidden = 2, 3, 16
        torch.manual_seed(0)
        hidden_states = torch.randn(batch, seq, hidden)
        dirs = self._make_dirs(k=2, hidden=hidden)
        alpha = 0.5

        out = inject_orthogonal_batch(hidden_states, dirs, alpha=alpha, rescale=True)

        for i in range(batch):
            h = hidden_states[i]  # (seq, hidden)
            h_norm = h.norm(dim=-1, keepdim=True)  # (seq, 1)
            v = torch.tensor(dirs[i])  # unit-norm
            expected = h + v * h_norm * alpha
            np.testing.assert_allclose(
                out[i].numpy(), expected.numpy(), atol=1e-5,
                err_msg=f"rescale convention violated for sample {i}"
            )

    def test_batch_direction_mismatch_raises(self):
        hidden_states = torch.randn(3, 5, 16)
        dirs = self._make_dirs(k=2, hidden=16)  # k=2 but batch=3
        with pytest.raises(ValueError):
            inject_orthogonal_batch(hidden_states, dirs, alpha=1.0)

    def test_hidden_dim_mismatch_raises(self):
        hidden_states = torch.randn(2, 5, 16)
        dirs = self._make_dirs(k=2, hidden=32)  # wrong hidden dim
        with pytest.raises(ValueError):
            inject_orthogonal_batch(hidden_states, dirs, alpha=1.0)

    def test_returns_tensor_not_in_place(self):
        """inject_orthogonal_batch must not modify the input tensor."""
        batch, seq, hidden = 2, 4, 8
        h = torch.ones(batch, seq, hidden)
        h_copy = h.clone()
        dirs = self._make_dirs(k=2, hidden=hidden)
        _ = inject_orthogonal_batch(h, dirs, alpha=1.0, rescale=False)
        assert torch.equal(h, h_copy), "input hidden_states was mutated"

    def test_alpha_zero_is_identity(self):
        batch, seq, hidden = 3, 6, 12
        h = torch.randn(batch, seq, hidden)
        dirs = self._make_dirs(k=3, hidden=hidden)
        out = inject_orthogonal_batch(h, dirs, alpha=0.0, rescale=True)
        np.testing.assert_allclose(out.numpy(), h.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# orthonormalize_rows — Gram–Schmidt decorrelation of a set of directions (X1×X2)
# ---------------------------------------------------------------------------

class TestOrthonormalizeRows:
    def test_gram_matrix_is_identity(self):
        M = np.random.default_rng(0).normal(size=(5, 16)).astype(np.float32)
        out = orthonormalize_rows(M)
        np.testing.assert_allclose(out @ out.T, np.eye(5), atol=1e-5)

    def test_row0_collinear_with_input_row0(self):
        M = np.random.default_rng(1).normal(size=(4, 12)).astype(np.float32)
        out = orthonormalize_rows(M)
        expected = M[0] / np.linalg.norm(M[0])
        np.testing.assert_allclose(out[0], expected, atol=1e-5)

    def test_each_row_sign_aligned_with_input(self):
        M = np.random.default_rng(2).normal(size=(6, 32)).astype(np.float32)
        out = orthonormalize_rows(M)
        for i in range(6):
            assert float(np.dot(out[i], M[i])) >= -1e-6, f"row {i} not sign-aligned"

    def test_span_preserved(self):
        """For full-rank input, projecting the input onto the orthonormal rows is the identity."""
        M = np.random.default_rng(3).normal(size=(4, 16)).astype(np.float32)
        out = orthonormalize_rows(M)
        recon = (M @ out.T) @ out
        np.testing.assert_allclose(recon, M, atol=1e-4)

    def test_k_exceeds_dim_raises(self):
        with pytest.raises(ValueError):
            orthonormalize_rows(np.ones((5, 4), dtype=np.float32))

    def test_dtype_float32(self):
        out = orthonormalize_rows(np.random.default_rng(4).normal(size=(3, 8)))
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# distinctness_order — most-distinct-first ordering (guardrail 1)
# ---------------------------------------------------------------------------

class TestDistinctnessOrder:
    def test_returns_permutation(self):
        V = np.random.default_rng(0).normal(size=(5, 16)).astype(np.float32)
        order = distinctness_order(V)
        assert sorted(order.tolist()) == list(range(5))

    def test_distinct_first_redundant_last(self):
        """A vector orthogonal to the rest ranks first; a near-duplicate ranks last."""
        d = 8
        v0 = np.zeros(d, dtype=np.float32); v0[0] = 1.0
        v2 = np.zeros(d, dtype=np.float32); v2[1] = 1.0  # orthogonal to v0 -> most distinct
        v1 = v0 + 1e-3 * np.random.default_rng(1).normal(size=d).astype(np.float32)  # ~v0
        V = np.stack([v0, v1, v2])
        order = distinctness_order(V)
        assert order[0] == 2, f"orthogonal vector should rank first; got {order}"
        assert order[-1] in (0, 1), f"a near-duplicate should rank last; got {order}"
