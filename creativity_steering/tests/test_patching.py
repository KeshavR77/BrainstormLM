import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.config import CAPTURE_LAST_RESPONSE
from creativity_steering.models.mock import MockBackend
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS
from creativity_steering.patching.multilayer import (
    all_layers_single_vector,
    multi_vector_combo,
)
from creativity_steering.patching.sweep import per_layer_alpha_sweep
from creativity_steering.scoring import HeuristicScorer


def _backend_and_vectors():
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=12, seed=0)
    # Direction = planted axis, per-layer norm = salience (peaks at 12).
    vectors = (b.creativity_axis * b.layer_salience[:, None]).astype(np.float32)
    vs = SteeringVectorSet(
        vectors=vectors,
        model_id=b.model_id,
        capture_position=CAPTURE_LAST_RESPONSE,
        n_positive=100,
        n_negative=100,
    )
    return b, vs


BRIEFS = DEFAULT_BRIEFS[:3]


def test_sweep_finds_planted_layer_with_positive_alpha():
    b, vs = _backend_and_vectors()
    result = per_layer_alpha_sweep(
        b, vs, HeuristicScorer(),
        alphas=[-4.0, -1.0, 1.0, 4.0],
        layers=[6, 9, 12, 15, 18],
        briefs=BRIEFS, n_samples=3,
    )
    best = result.best_cell()
    assert best["alpha"] > 0
    assert abs(best["layer"] - 12) <= 3
    assert best["composite"] > result.baseline.composite
    # the composite matrix is fully populated
    assert not np.isnan(result.composite_matrix()).any()


def test_negative_alpha_suppresses_creativity():
    b, vs = _backend_and_vectors()
    result = per_layer_alpha_sweep(
        b, vs, HeuristicScorer(),
        alphas=[-4.0, 4.0], layers=[12], briefs=BRIEFS, n_samples=3,
    )
    neg = next(c for c in result.cells if c["alpha"] == -4.0)
    assert neg["composite"] < result.baseline.composite


def test_all_layers_injection_beats_baseline():
    b, vs = _backend_and_vectors()
    res = all_layers_single_vector(
        b, vs, HeuristicScorer(), source_layer=12, alpha=0.3, briefs=BRIEFS, n_samples=3,
    )
    assert res.delta > 0


def test_multi_vector_combo_beats_baseline():
    b, vs = _backend_and_vectors()
    res = multi_vector_combo(
        b, vs, HeuristicScorer(), layers=[9, 12, 15], alpha=2.0, briefs=BRIEFS, n_samples=3,
    )
    assert res.delta > 0
