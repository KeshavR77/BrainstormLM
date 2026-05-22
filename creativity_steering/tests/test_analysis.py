import numpy as np

from creativity_steering.analysis.cosine import analyze_cosine, furthest_layer_pairs
from creativity_steering.analysis.norm import analyze_norms
from creativity_steering.analysis.oracle import (
    LatentQAOracle,
    StubOracle,
    oracle_layer_scan,
)
from creativity_steering.analysis.report import build_report


def _planted_vectors(L=20, H=16, peak=10, seed=0):
    """Layers peak-2..peak+2 form an aligned band with norm peaking at `peak`."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(scale=0.1, size=(L, H)).astype(np.float32)
    d = rng.normal(size=H)
    d = d / np.linalg.norm(d)
    for off in range(-2, 3):
        layer = peak + off
        vecs[layer] = (3 - abs(off)) * d + 0.03 * rng.normal(size=H)
    return vecs.astype(np.float32), d


def test_norm_ranking_peaks_at_planted_layer():
    vecs, _ = _planted_vectors()
    norm = analyze_norms(vecs)
    assert norm.top_layer == 10
    assert norm.ranked_layers[:3] == [10, 11, 9] or norm.ranked_layers[:3] == [10, 9, 11]


def test_cosine_band_brackets_the_aligned_layers():
    vecs, _ = _planted_vectors()
    cos = analyze_cosine(vecs)
    lo, hi = cos.band
    assert lo <= 10 <= hi
    # neighbour cosine is higher inside the band than outside.
    assert cos.neighbor_cosine[10] > cos.neighbor_cosine[0]


def test_furthest_pairs_are_low_cosine():
    vecs, _ = _planted_vectors()
    pairs = furthest_layer_pairs(vecs, n_pairs=3)
    # furthest pairs have smaller cosine than a within-band pair
    within_band_cos = float(np.dot(vecs[9] / np.linalg.norm(vecs[9]), vecs[11] / np.linalg.norm(vecs[11])))
    assert all(c < within_band_cos for (_, _, c) in pairs)


def test_stub_oracle_scan_peaks_in_band():
    vecs, _ = _planted_vectors()
    norms = np.linalg.norm(vecs, axis=-1)
    oracle = StubOracle(reference_norms=norms)
    scan = oracle_layer_scan(oracle, vecs)
    assert scan.shape == (20,)
    assert scan[10] >= scan[0]
    assert 0.0 <= scan.min() and scan.max() <= 1.0


def test_report_candidates_include_planted_layer():
    vecs, _ = _planted_vectors()
    oracle = StubOracle(reference_norms=np.linalg.norm(vecs, axis=-1))
    report = build_report(vecs, oracle=oracle, n_candidates=5)
    assert 10 in report.candidate_layers
    assert "candidate layers" in report.to_markdown().lower()


def test_latentqa_oracle_is_a_documented_slot():
    import pytest

    with pytest.raises(NotImplementedError):
        LatentQAOracle()
