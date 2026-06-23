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


# --- variants: choose just L17 or the bracket ---------------------------------------------

def _bracket_pkg():
    return SteeringPackage(
        name="v3-mid", vectors_path="(unused)", default_variant="L4",
        variants={
            "L4": [{"layer": 4, "alpha": 8.0}],
            "bracket": [{"layer": 2, "alpha": 8.0}, {"layer": 4, "alpha": 8.0},
                        {"layer": 6, "alpha": 8.0}],
        },
    )


def test_single_layer_package_back_compat():
    # a top-level layer/alpha package still works and exposes a one-variant set
    pkg = _pkg(layer=17, alpha=0.2)
    assert pkg.layer == 17 and pkg.default_variant in pkg.variant_names()
    _, vs = _backend_and_vs()
    assert len(pkg.injections(vs)) == 1


def test_variant_selection_builds_right_injections():
    _, vs = _backend_and_vs()
    pkg = _bracket_pkg()
    assert set(pkg.variant_names()) == {"L4", "bracket"}
    assert len(pkg.injections(vs, variant="L4")) == 1
    bracket = pkg.injections(vs, variant="bracket")
    assert len(bracket) == 3 and {i.layer for i in bracket} == {2, 4, 6}
    # each uses its own layer's unit direction
    assert all(abs(float(np.linalg.norm(i.vector)) - 1.0) < 1e-5 for i in bracket)


def test_default_variant_used_when_unspecified():
    _, vs = _backend_and_vs()
    pkg = _bracket_pkg()  # default L4
    assert len(pkg.injections(vs)) == 1  # the default, single layer


def test_unknown_variant_raises():
    _, vs = _backend_and_vs()
    with pytest.raises(KeyError):
        _bracket_pkg().injections(vs, variant="nope")


def test_service_validates_all_variant_layers():
    b, vs = _backend_and_vs()  # 24 layers
    bad = SteeringPackage(name="x", vectors_path="(u)", default_variant="ok",
                          variants={"ok": [{"layer": 4, "alpha": 1.0}],
                                    "bad": [{"layer": 99, "alpha": 1.0}]})
    with pytest.raises(ValueError):
        SteeringService(b, vs, bad)


def test_service_generate_differs_by_variant():
    b, vs = _backend_and_vs(creative_layer=4)
    svc = SteeringService(b, vs, _bracket_pkg())
    single = svc.generate("Describe a chair.", variant="L4", seed=0)
    bracket = svc.generate("Describe a chair.", variant="bracket", seed=0)
    baseline = svc.generate("Describe a chair.", steered=False, seed=0)
    assert single != baseline and bracket != baseline


# --- optional concept-frozen refiner (steer -> refine) ------------------------------------

class _FakeRefiner:
    model = "fake-refiner"

    def __init__(self):
        self.calls = []

    def refine(self, prompt, text):
        self.calls.append((prompt, text))
        return f"REFINED<<{text}>>"

    def info(self):
        return {"model": self.model}


def test_service_refine_text_delegates_to_refiner():
    b, vs = _backend_and_vs()
    ref = _FakeRefiner()
    svc = SteeringService(b, vs, _pkg(layer=4, alpha=8.0), refiner=ref)
    steered = svc.generate("Describe a chair.", seed=0)
    refined = svc.refine_text("Describe a chair.", steered)
    assert refined == f"REFINED<<{steered}>>"
    assert ref.calls == [("Describe a chair.", steered)]
    assert svc.has_refiner is True


def test_service_refine_without_refiner_raises():
    b, vs = _backend_and_vs()
    svc = SteeringService(b, vs, _pkg(layer=4, alpha=8.0))
    assert svc.has_refiner is False
    with pytest.raises(RuntimeError):
        svc.refine_text("p", "steered text")


def test_service_info_reports_refiner():
    b, vs = _backend_and_vs()
    info = SteeringService(b, vs, _pkg(layer=4, alpha=8.0), refiner=_FakeRefiner()).info()
    assert info["refiner"] == {"model": "fake-refiner"}
    no_ref = SteeringService(b, vs, _pkg(layer=4, alpha=8.0)).info()
    assert no_ref["refiner"] is None
