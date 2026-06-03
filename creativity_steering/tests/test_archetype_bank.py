"""Tests for X2: archetype-indexed DiffMeans bank.

All tests are pure-math / shape tests on small CPU numpy arrays.  No model
or GPU required.

We test:
  - ArchetypeBank stores K vectors of shape (hidden_dim,)
  - get(k) returns the correct sub-vector
  - vectors are distinct (not all identical)
  - from_diffmeans_result correctly slices layer vectors as archetype modes
  - from_stacks convenience constructor runs end-to-end
  - out-of-range index raises IndexError
  - k=1 edge case works
  - assign_to_batch maps batch indices correctly (round-robin when batch > K)
  - unit_vectors returns unit-normed rows matching DiffMeans convention

We do NOT assert semantic outcomes — see non-identifiability caveat (2602.06801).
"""

from __future__ import annotations

import numpy as np
import pytest

from creativity_steering.derivation.archetype_bank import (
    ArchetypeBank,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_bank(k: int = 4, hidden: int = 16, seed: int = 0) -> ArchetypeBank:
    """Build a bank from random distinct vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(k, hidden)).astype(np.float32)
    return ArchetypeBank(vectors=vecs, archetype_names=None)


def _make_diffmeans_stacks(
    n_pos: int = 10,
    n_neg: int = 10,
    n_layers: int = 6,
    hidden: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n_pos, n_layers, hidden)).astype(np.float32)
    neg = rng.normal(size=(n_neg, n_layers, hidden)).astype(np.float32)
    return pos, neg


# ---------------------------------------------------------------------------
# ArchetypeBank construction
# ---------------------------------------------------------------------------

class TestArchetypeBankConstruction:
    def test_shape_stored(self):
        bank = _make_bank(k=5, hidden=32)
        assert bank.k == 5
        assert bank.hidden_dim == 32
        assert bank.vectors.shape == (5, 32)

    def test_k1_works(self):
        bank = _make_bank(k=1, hidden=8)
        assert bank.k == 1
        assert bank.vectors.shape == (1, 8)

    def test_vectors_distinct(self):
        bank = _make_bank(k=4, hidden=16, seed=42)
        for i in range(bank.k):
            for j in range(i + 1, bank.k):
                assert not np.allclose(bank.vectors[i], bank.vectors[j])

    def test_optional_names_stored(self):
        vecs = np.eye(3, dtype=np.float32)
        names = ["structural", "metaphorical", "contrarian"]
        bank = ArchetypeBank(vectors=vecs, archetype_names=names)
        assert bank.archetype_names == names

    def test_name_count_mismatch_raises(self):
        vecs = np.eye(3, dtype=np.float32)
        with pytest.raises(ValueError):
            ArchetypeBank(vectors=vecs, archetype_names=["a", "b"])  # only 2 names for 3 vecs

    def test_vectors_not_1d_raises(self):
        with pytest.raises(ValueError):
            ArchetypeBank(vectors=np.ones((3,), dtype=np.float32), archetype_names=None)

    def test_from_diffmeans_result_shape(self):
        """from_diffmeans_result slices the requested layers → one sub-vector per archetype."""
        pos, neg = _make_diffmeans_stacks(n_layers=8, hidden=16)
        bank = ArchetypeBank.from_diffmeans_result(pos, neg, archetype_layers=[1, 3, 5])
        assert bank.k == 3
        assert bank.hidden_dim == 16

    def test_from_diffmeans_result_vectors_match_diffmeans(self):
        """Each archetype vector must equal the DiffMeans direction at the chosen layer."""
        from creativity_steering.derivation.diffmeans import diffmeans_per_layer
        pos, neg = _make_diffmeans_stacks(n_layers=6, hidden=16)
        layers = [0, 2, 4]
        bank = ArchetypeBank.from_diffmeans_result(pos, neg, archetype_layers=layers)
        dm = diffmeans_per_layer(pos, neg)  # (n_layers, hidden)
        for idx, L in enumerate(layers):
            np.testing.assert_allclose(
                bank.vectors[idx], dm[L], atol=1e-5,
                err_msg=f"archetype {idx} does not match DiffMeans layer {L}"
            )

    def test_from_stacks_convenience(self):
        """from_stacks is a thin wrapper — just check it returns a valid bank."""
        pos, neg = _make_diffmeans_stacks(n_layers=4, hidden=8)
        bank = ArchetypeBank.from_stacks(pos, neg)
        assert bank.k == 4  # defaults to one archetype per layer
        assert bank.hidden_dim == 8

    def test_from_stacks_custom_layers(self):
        pos, neg = _make_diffmeans_stacks(n_layers=6, hidden=12)
        bank = ArchetypeBank.from_stacks(pos, neg, archetype_layers=[2, 4])
        assert bank.k == 2


# ---------------------------------------------------------------------------
# ArchetypeBank.get
# ---------------------------------------------------------------------------

class TestArchetypeBankGet:
    def test_get_returns_correct_vector(self):
        vecs = np.arange(12, dtype=np.float32).reshape(3, 4)
        bank = ArchetypeBank(vectors=vecs, archetype_names=None)
        np.testing.assert_array_equal(bank.get(0), vecs[0])
        np.testing.assert_array_equal(bank.get(1), vecs[1])
        np.testing.assert_array_equal(bank.get(2), vecs[2])

    def test_get_out_of_range_raises(self):
        bank = _make_bank(k=3)
        with pytest.raises(IndexError):
            bank.get(3)
        with pytest.raises(IndexError):
            bank.get(-1)

    def test_get_returns_copy_not_view(self):
        bank = _make_bank(k=2, hidden=8)
        v = bank.get(0)
        v[:] = 99.0
        assert not np.allclose(bank.vectors[0], 99.0), "get() returned a live view"


# ---------------------------------------------------------------------------
# ArchetypeBank.unit_vectors
# ---------------------------------------------------------------------------

class TestArchetypeBankUnitVectors:
    def test_unit_norm(self):
        bank = _make_bank(k=4, hidden=16)
        uvecs = bank.unit_vectors()
        norms = np.linalg.norm(uvecs, axis=-1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)

    def test_shape_unchanged(self):
        bank = _make_bank(k=5, hidden=20)
        uvecs = bank.unit_vectors()
        assert uvecs.shape == (5, 20)

    def test_zero_vector_stays_zero(self):
        vecs = np.zeros((3, 8), dtype=np.float32)
        vecs[1] = 1.0  # only index 1 non-zero
        bank = ArchetypeBank(vectors=vecs, archetype_names=None)
        uvecs = bank.unit_vectors()
        np.testing.assert_allclose(uvecs[0], np.zeros(8), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(uvecs[1]), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# ArchetypeBank.orthonormalized — plain Gram–Schmidt decorrelation of the bank
# ---------------------------------------------------------------------------

class TestArchetypeBankOrthonormalized:
    def test_orthonormal_rows(self):
        bank = _make_bank(k=4, hidden=16, seed=7)
        o = bank.orthonormalized()
        assert o.shape == (4, 16)
        np.testing.assert_allclose(o @ o.T, np.eye(4), atol=1e-4)


# ---------------------------------------------------------------------------
# ArchetypeBank.assign_to_batch
# ---------------------------------------------------------------------------

class TestArchetypeBankAssignToBatch:
    def test_batch_equals_k(self):
        bank = _make_bank(k=4, hidden=8)
        assigned = bank.assign_to_batch(batch_size=4)
        assert assigned.shape == (4, 8)
        for i in range(4):
            np.testing.assert_array_equal(assigned[i], bank.vectors[i])

    def test_batch_less_than_k(self):
        bank = _make_bank(k=6, hidden=8)
        assigned = bank.assign_to_batch(batch_size=3)
        assert assigned.shape == (3, 8)
        for i in range(3):
            np.testing.assert_array_equal(assigned[i], bank.vectors[i])

    def test_batch_greater_than_k_wraps_round_robin(self):
        """When batch_size > k, indices wrap round-robin: sample i → archetype i % k."""
        bank = _make_bank(k=3, hidden=8)
        assigned = bank.assign_to_batch(batch_size=7)
        assert assigned.shape == (7, 8)
        for i in range(7):
            np.testing.assert_array_equal(
                assigned[i], bank.vectors[i % bank.k],
                err_msg=f"sample {i}: expected archetype {i % bank.k}"
            )

    def test_batch_size_zero_raises(self):
        bank = _make_bank(k=2)
        with pytest.raises(ValueError):
            bank.assign_to_batch(batch_size=0)
