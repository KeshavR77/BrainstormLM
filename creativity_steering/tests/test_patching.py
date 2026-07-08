import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.config import CAPTURE_LAST_RESPONSE
from creativity_steering.models.mock import MockBackend
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS
from creativity_steering.patching.multilayer import (
    all_layers_single_vector,
    multi_vector_combo,
)
from creativity_steering.patching.sweep import (
    per_layer_alpha_sweep,
    select_promising_layers,
)
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


def test_select_promising_layers_per_item_ranks_by_best_composite():
    # Per-item sweep: each layer scores its best cell over alphas; rank by that composite.
    sweep = {
        "cells": [
            {"layer": 3, "alpha": 1.0, "composite": 0.5},
            {"layer": 3, "alpha": -1.0, "composite": 0.2},  # layer 3 best = 0.5
            {"layer": 7, "alpha": 1.0, "composite": 0.9},  # layer 7 best = 0.9
            {"layer": 7, "alpha": -1.0, "composite": 0.1},
            {"layer": 5, "alpha": 1.0, "composite": 0.7},  # layer 5 best = 0.7
        ],
    }
    assert select_promising_layers(sweep, 2) == [7, 5]
    assert select_promising_layers(sweep, 1) == [7]


def test_select_promising_layers_two_stage_ranks_judged_first():
    # Judged cells carry the two-stage composite (its own scale); un-judged keep Stage-1.
    # Judged layers must rank above any un-judged layer regardless of Stage-1, then the
    # remaining slots fill from un-judged layers by Stage-1.
    sweep = {
        "evaluator": "two_stage",
        "cells": [
            {"layer": 4, "alpha": 2.0, "composite": 0.30, "stage1": 0.10, "judged": True},
            {"layer": 4, "alpha": -2.0, "composite": 0.05, "stage1": 0.05, "judged": False},
            {"layer": 1, "alpha": 2.0, "composite": 0.40, "stage1": 0.20, "judged": True},
            {"layer": 7, "alpha": 2.0, "composite": 0.95, "stage1": 0.95, "judged": False},
            {"layer": 9, "alpha": 2.0, "composite": 0.50, "stage1": 0.50, "judged": False},
        ],
    }
    # judged by two-stage composite: 1 (0.40) > 4 (0.30); then un-judged by Stage-1: 7 > 9.
    assert select_promising_layers(sweep, 4) == [1, 4, 7, 9]
    assert select_promising_layers(sweep, 2) == [1, 4]
    assert select_promising_layers(sweep, 3) == [1, 4, 7]


def test_select_promising_layers_two_stage_no_judge_uses_stage1():
    # Judging off: no cell judged, so rank by the consistent Stage-1 score.
    sweep = {
        "evaluator": "two_stage",
        "cells": [
            {"layer": 1, "alpha": 2.0, "composite": 0.1, "stage1": 0.1, "judged": False},
            {"layer": 2, "alpha": 2.0, "composite": 0.9, "stage1": 0.9, "judged": False},
            {"layer": 3, "alpha": 2.0, "composite": 0.5, "stage1": 0.5, "judged": False},
        ],
    }
    assert select_promising_layers(sweep, 2) == [2, 3]


def test_select_promising_layers_empty():
    assert select_promising_layers({"cells": []}, 3) == []
    assert select_promising_layers({}, 3) == []


def test_select_promising_layers_recovers_planted_layer():
    b, vs = _backend_and_vectors()
    result = per_layer_alpha_sweep(
        b, vs, HeuristicScorer(),
        alphas=[-4.0, -1.0, 1.0, 4.0], layers=[6, 9, 12, 15, 18],
        briefs=BRIEFS, n_samples=3,
    )
    promising = select_promising_layers(result.to_dict(), 3)
    assert len(promising) == 3
    assert 12 in promising
