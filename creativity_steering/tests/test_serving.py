"""Serving harness: a placeable {vector, layer, alpha} package + a service that applies it
to steered generation. Runs offline on MockBackend (whose steered output reflects the
injection, so steered != baseline is observable)."""

import json

import numpy as np
import pytest

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.mock import MockBackend
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.service import SteeringService


def _backend_and_vs(creative_layer=4):
    """A mock backend + a vector set whose directions ARE the mock's planted creativity axis,
    so injecting at a salient layer measurably shifts the output."""
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=creative_layer, seed=0)
    vs = SteeringVectorSet(
        vectors=np.asarray(b.creativity_axis, dtype=np.float32),
        model_id="mock-qwen", capture_position="mean_response",
        n_positive=10, n_negative=10,
    )
    return b, vs


def _pkg(layer=4, alpha=8.0):
    return SteeringPackage(name="v3-Ltest", vectors_path="(unused)", layer=layer, alpha=alpha)


def test_package_load_roundtrip(tmp_path):
    p = tmp_path / "pkg.json"
    p.write_text(json.dumps({
        "name": "v3-L17-a0.2", "vectors_path": "/vecs", "layer": 17, "alpha": 0.2,
        "rescale": True, "model_id": "Qwen/Qwen2.5-14B-Instruct", "device": "cuda",
        "dtype": "bf16", "max_new_tokens": 128, "temperature": 0.8, "seed": 3,
    }))
    pkg = SteeringPackage.load(p)
    assert pkg.name == "v3-L17-a0.2" and pkg.layer == 17 and pkg.alpha == 0.2
    assert pkg.rescale is True and pkg.max_new_tokens == 128 and pkg.seed == 3
    # unknown keys in the json are ignored, not fatal
    pkg2 = SteeringPackage.from_dict({"name": "x", "vectors_path": "/v", "layer": 1,
                                      "alpha": 0.3, "future_field": "ignored"})
    assert pkg2.layer == 1


def test_package_builds_single_layer_unit_injection():
    _, vs = _backend_and_vs()
    inj = _pkg(layer=17, alpha=0.2).injections(vs)
    assert len(inj) == 1
    assert inj[0].layer == 17 and inj[0].alpha == 0.2 and inj[0].rescale is True
    assert abs(float(np.linalg.norm(inj[0].vector)) - 1.0) < 1e-5  # unit direction


def test_service_steered_differs_from_baseline():
    b, vs = _backend_and_vs(creative_layer=4)
    svc = SteeringService(b, vs, _pkg(layer=4, alpha=8.0))
    steered = svc.generate("Describe a chair.", steered=True, seed=0)
    baseline = svc.generate("Describe a chair.", steered=False, seed=0)
    assert isinstance(steered, str) and steered
    assert steered != baseline  # the injection changes the output; baseline has none


def test_service_rejects_layer_out_of_range():
    b, vs = _backend_and_vs()  # 24 layers
    with pytest.raises(ValueError):
        SteeringService(b, vs, _pkg(layer=99, alpha=0.2))


def test_service_info_reports_the_shipped_package():
    b, vs = _backend_and_vs()
    info = SteeringService(b, vs, _pkg(layer=17, alpha=0.2)).info()
    assert info["layer"] == 17 and info["alpha"] == 0.2
    assert info["capture_position"] == "mean_response" and info["n_layers"] == 24
