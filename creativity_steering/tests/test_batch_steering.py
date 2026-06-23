"""Tests for B2: per-sample batch steering hook in ``models/hf.py``.

Pure-tensor tests on the forward-hook factory ``_make_batch_hook`` — no model load,
no GPU.  The hook must apply ``inject_orthogonal_batch(hidden, dirs, alpha, rescale)``
to a decoder block's output, preserving the ``(hidden, *tail)`` tuple convention used
by the existing ``_make_hook`` (so the KV-cache tail rides through untouched).

Per the non-identifiability caveat (arXiv:2602.06801) we test plumbing/geometry only;
semantic spread is validated by the real-model smoke, not here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from creativity_steering.derivation.archetype_bank import ArchetypeBank
from creativity_steering.derivation.diffmeans import unit_normalize
from creativity_steering.models.hf import _make_batch_hook
from creativity_steering.patching.orthogonal import inject_orthogonal_batch, stiefel_basis
from creativity_steering.patching.steering import build_orthogonal_archetype_dirs


def _dirs(b: int, hidden: int, seed: int = 0) -> np.ndarray:
    base = np.random.default_rng(seed).normal(size=hidden).astype(np.float32)
    return stiefel_basis(base, k=b, seed=seed)


class TestMakeBatchHook:
    def test_applies_inject_orthogonal_batch_tuple_output(self):
        b, t, h = 3, 5, 16
        torch.manual_seed(0)
        hidden = torch.randn(b, t, h)
        dirs = _dirs(b, h)
        alpha = 0.7
        hook = _make_batch_hook(dirs, alpha=alpha, rescale=True)
        out = hook(None, None, (hidden, "kv"))
        assert isinstance(out, tuple) and out[1] == "kv", "tuple tail not preserved"
        expected = inject_orthogonal_batch(hidden, dirs, alpha, rescale=True)
        np.testing.assert_allclose(out[0].numpy(), expected.numpy(), atol=1e-5)

    def test_bare_tensor_output(self):
        b, t, h = 2, 4, 16
        hidden = torch.randn(b, t, h)
        dirs = _dirs(b, h)
        hook = _make_batch_hook(dirs, alpha=1.0, rescale=False)
        out = hook(None, None, hidden)  # bare tensor, not a tuple
        assert torch.is_tensor(out)
        expected = inject_orthogonal_batch(hidden, dirs, 1.0, rescale=False)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5)

    def test_does_not_mutate_input(self):
        b, t, h = 2, 3, 8
        hidden = torch.randn(b, t, h)
        hidden_copy = hidden.clone()
        dirs = _dirs(b, h)
        hook = _make_batch_hook(dirs, alpha=2.0, rescale=True)
        _ = hook(None, None, (hidden,))
        assert torch.equal(hidden, hidden_copy), "hook mutated the input hidden state"

    def test_no_nan_and_zero_direction_is_graceful(self):
        """A zero direction row must yield no change for that sample and never NaN."""
        b, t, h = 3, 4, 12
        hidden = torch.randn(b, t, h)
        dirs = _dirs(b, h)
        dirs[1] = 0.0  # sample 1 gets a degenerate (zero) direction
        hook = _make_batch_hook(dirs, alpha=1.5, rescale=True)
        out = hook(None, None, hidden)  # bare tensor in -> bare tensor out
        assert not torch.isnan(out).any(), "hook produced NaNs"
        np.testing.assert_allclose(out[1].numpy(), hidden[1].numpy(), atol=1e-5)

    def test_shape_preserved(self):
        b, t, h = 4, 6, 16
        hidden = torch.randn(b, t, h)
        dirs = _dirs(b, h)
        hook = _make_batch_hook(dirs, alpha=1.0, rescale=True)
        out = hook(None, None, hidden)  # bare tensor in -> bare tensor out
        assert out.shape == hidden.shape


# ---------------------------------------------------------------------------
# build_orthogonal_archetype_dirs — X1×X2 fusion ("orthogonalize the bank")
# ---------------------------------------------------------------------------

class TestBuildOrthogonalArchetypeDirs:
    def test_orthonormal_when_distinct_no_fallback(self):
        rng = np.random.default_rng(4)
        Q, _ = np.linalg.qr(rng.normal(size=(16, 16)).astype(np.float32))
        base = Q[:, :4].T  # 4 exactly-orthonormal rows
        vecs = (base + 0.03 * rng.normal(size=(4, 16)).astype(np.float32)).astype(np.float32)
        bank = ArchetypeBank(vectors=vecs, archetype_names=None)
        res = build_orthogonal_archetype_dirs(bank, batch_size=4)
        assert res.dirs.shape == (4, 16)
        np.testing.assert_allclose(res.dirs @ res.dirs.T, np.eye(4), atol=1e-4)
        assert res.fell_back == []
        assert (res.retention_cos > 0.5).all()
        assert sorted(res.order) == [0, 1, 2, 3]

    def test_round_robin_wraps(self):
        rng = np.random.default_rng(5)
        bank = ArchetypeBank(vectors=rng.normal(size=(4, 16)).astype(np.float32))
        res = build_orthogonal_archetype_dirs(bank, batch_size=7)
        assert res.dirs.shape == (7, 16)
        for i in range(7):
            np.testing.assert_array_equal(res.dirs[i], res.dirs[i % 4])

    def test_near_collinear_mode_falls_back_to_raw(self):
        rng = np.random.default_rng(3)
        d = 8
        m0 = np.zeros(d, dtype=np.float32)
        m0[0] = 1.0     # distinct
        m1 = rng.normal(size=d).astype(np.float32)
        m2 = (m1 + 1e-3 * rng.normal(size=d).astype(np.float32)).astype(np.float32)  # ~parallel to m1
        bank = ArchetypeBank(vectors=np.stack([m0, m1, m2]), archetype_names=None)
        res = build_orthogonal_archetype_dirs(bank, batch_size=3, retention_threshold=0.5)
        assert len(res.fell_back) >= 1, f"a collinear mode should fall back; cos={res.retention_cos}"
        unit = unit_normalize(bank.vectors)
        for i in res.fell_back:
            np.testing.assert_allclose(res.dirs[i], unit[i], atol=1e-5)

    def test_batch_size_zero_raises(self):
        bank = ArchetypeBank(vectors=np.eye(3, dtype=np.float32))
        with pytest.raises(ValueError):
            build_orthogonal_archetype_dirs(bank, batch_size=0)
