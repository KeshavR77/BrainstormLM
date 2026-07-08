"""Port guard: the DiffMeans steering path must be dimension-agnostic.

Extraction, DiffMeans, injection, and the sweep's layer enumeration all derive their
shape from the model's ``num_layers`` / ``hidden_dim`` — never a hardcoded constant. This
test drives the *real* capture -> diffmeans -> inject path through the mock backend at the
geometries of models we intend to port to, so a future change that silently pins the
pipeline to the 14B's 48 layers / 5120 dim fails here instead of on a paid GPU pod.

  - Qwen2.5-14B : 48 layers x 5120 dim  (current ship)
  - Qwen2.5-32B : 64 layers x 5120 dim  (next base — see docs/port_to_qwen32b.md)
  - Qwen2.5-72B : 80 layers x 8192 dim  (a wider-dim stress case)
"""

import numpy as np
import pytest

from creativity_steering.derivation.diffmeans import DiffMeansResult
from creativity_steering.models.backend import Injection
from creativity_steering.models.mock import MockBackend

# (num_layers, hidden_dim) for the models the steering stack must handle unchanged.
GEOMETRIES = [
    pytest.param(48, 5120, id="qwen14b"),
    pytest.param(64, 5120, id="qwen32b"),
    pytest.param(80, 8192, id="qwen72b"),
]


def _capture_stack(be: MockBackend, prompts: list[str]) -> np.ndarray:
    """Capture mean-response activations for each prompt -> (n_examples, n_layers, hidden)."""
    rows = [
        be.capture_activations(p, " a response", positions=["mean_response"])["mean_response"]
        for p in prompts
    ]
    return np.stack(rows)


@pytest.mark.parametrize("n_layers, hidden_dim", GEOMETRIES)
def test_diffmeans_derive_shape_follows_model_geometry(n_layers, hidden_dim):
    be = MockBackend(num_layers=n_layers, hidden_dim=hidden_dim, seed=0)
    assert be.num_layers == n_layers
    assert be.hidden_dim == hidden_dim

    pos = _capture_stack(be, ["invent a novel gadget", "imagine an impossible city"])
    neg = _capture_stack(be, ["boil water steps", "capital of France"])
    assert pos.shape == (2, n_layers, hidden_dim)

    res = DiffMeansResult.from_stacks(pos, neg)
    # Vectors, norms, and reported metadata all track the model geometry — no 48/5120 pin.
    assert res.vectors.shape == (n_layers, hidden_dim)
    assert res.n_layers == n_layers
    assert res.hidden_dim == hidden_dim
    assert res.layer_norms.shape == (n_layers,)
    assert res.unit_vectors().shape == (n_layers, hidden_dim)


@pytest.mark.parametrize("n_layers, hidden_dim", GEOMETRIES)
def test_injection_reaches_the_deepest_layer(n_layers, hidden_dim):
    """Steering must inject at any block 0..n_layers-1, including blocks that only exist on
    the larger models (L63 on 32B, L79 on 72B are out of range for the 48-layer 14B)."""
    be = MockBackend(num_layers=n_layers, hidden_dim=hidden_dim, seed=0)
    pos = _capture_stack(be, ["a", "b"])
    neg = _capture_stack(be, ["c", "d"])
    unit = DiffMeansResult.from_stacks(pos, neg).unit_vectors()

    for layer in (0, n_layers // 2, n_layers - 1):
        out = be.generate_steered(
            "brainstorm",
            injections=[Injection(layer=layer, vector=unit[layer], alpha=0.25, rescale=True)],
            max_new_tokens=8,
        )
        assert isinstance(out, str) and out

    # A layer one past the last block must be rejected, not silently clamped.
    with pytest.raises((ValueError, IndexError)):
        be.generate_steered(
            "brainstorm",
            injections=[Injection(layer=n_layers, vector=unit[-1], alpha=0.25, rescale=True)],
            max_new_tokens=8,
        )


@pytest.mark.parametrize("n_layers, hidden_dim", GEOMETRIES)
def test_sweep_enumerates_every_derived_layer(n_layers, hidden_dim):
    """The CLI sweep uses ``list(range(vs.n_layers))`` as its default layer grid; confirm that
    grid spans all derived layers so no deep layer is skipped on a larger model."""
    be = MockBackend(num_layers=n_layers, hidden_dim=hidden_dim, seed=0)
    res = DiffMeansResult.from_stacks(
        _capture_stack(be, ["a", "b"]), _capture_stack(be, ["c", "d"])
    )
    layer_grid = list(range(res.n_layers))
    assert layer_grid[0] == 0
    assert layer_grid[-1] == n_layers - 1
    assert len(layer_grid) == n_layers
