import numpy as np
import pytest

from creativity_steering.derivation.diffmeans import (
    DiffMeansResult,
    diffmeans_per_layer,
    unit_normalize,
)


def test_diffmeans_is_difference_of_per_layer_means():
    # 2 positives, 2 negatives, 3 layers, 4 dims.
    pos = np.array(
        [
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]],
            [[3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 5, 0]],
        ],
        dtype=np.float32,
    )
    neg = np.zeros((2, 3, 4), dtype=np.float32)
    out = diffmeans_per_layer(pos, neg)
    # mean over the 2 positives, minus zero.
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[0], [2, 0, 0, 0])
    np.testing.assert_allclose(out[1], [0, 3, 0, 0])
    np.testing.assert_allclose(out[2], [0, 0, 4, 0])


def test_diffmeans_equals_mean_of_paired_differences():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(5, 4, 8)).astype(np.float32)
    neg = rng.normal(size=(5, 4, 8)).astype(np.float32)
    direct = diffmeans_per_layer(pos, neg)
    paired = (pos - neg).mean(axis=0)  # only equal because n_pos == n_neg
    np.testing.assert_allclose(direct, paired, atol=1e-5)


def test_result_carries_layer_norms_and_counts():
    pos = np.ones((3, 2, 4), dtype=np.float32)
    neg = np.zeros((7, 2, 4), dtype=np.float32)
    res = DiffMeansResult.from_stacks(pos, neg)
    assert res.n_positive == 3
    assert res.n_negative == 7
    assert res.vectors.shape == (2, 4)
    # each layer vector is [1,1,1,1] → L2 norm 2.0
    np.testing.assert_allclose(res.layer_norms, [2.0, 2.0])


def test_unit_normalize_makes_each_layer_unit_length():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = unit_normalize(vecs)
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
    # zero vector stays zero (no divide-by-zero blow-up)
    np.testing.assert_allclose(out[1], [0.0, 0.0])


def test_empty_stack_raises():
    with pytest.raises(ValueError):
        diffmeans_per_layer(np.zeros((0, 2, 4)), np.ones((3, 2, 4)))
