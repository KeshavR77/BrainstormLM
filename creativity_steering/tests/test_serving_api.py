"""FastAPI surface for the steering harness — health/package/generate/compare, driven on a
MockBackend so it runs offline."""

import numpy as np
from fastapi.testclient import TestClient

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.mock import MockBackend
from creativity_steering.serving.app import create_app
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.service import SteeringService


def _svc(package):
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=4, seed=0)
    vs = SteeringVectorSet(vectors=np.asarray(b.creativity_axis, dtype=np.float32),
                           model_id="mock-qwen", capture_position="mean_response",
                           n_positive=10, n_negative=10)
    return SteeringService(b, vs, package)


def _client():
    return TestClient(create_app(_svc(
        SteeringPackage(name="v3-L4-a8", vectors_path="(x)", layer=4, alpha=8.0))))


def _bracket_client():
    pkg = SteeringPackage(
        name="v3-mid", vectors_path="(x)", default_variant="L4",
        variants={"L4": [{"layer": 4, "alpha": 8.0}],
                  "bracket": [{"layer": 2, "alpha": 8.0}, {"layer": 4, "alpha": 8.0},
                              {"layer": 6, "alpha": 8.0}]})
    return TestClient(create_app(_svc(pkg)))


def test_health_reports_package():
    r = _client().get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["layer"] == 4 and j["alpha"] == 8.0


def test_package_endpoint():
    j = _client().get("/package").json()
    assert j["name"] == "v3-L4-a8" and j["layer"] == 4 and j["capture_position"] == "mean_response"


def test_generate_returns_text_and_provenance():
    r = _client().post("/generate", json={"prompt": "Describe a chair."})
    assert r.status_code == 200
    j = r.json()
    assert j["text"] and j["steered"] is True and j["layer"] == 4 and j["package"] == "v3-L4-a8"


def test_generate_baseline_vs_steered_differ():
    c = _client()
    steered = c.post("/generate", json={"prompt": "Describe a chair.", "steered": True, "seed": 0}).json()
    baseline = c.post("/generate", json={"prompt": "Describe a chair.", "steered": False, "seed": 0}).json()
    assert steered["text"] != baseline["text"]
    assert steered["steered"] is True and baseline["steered"] is False


def test_compare_returns_ab_pair():
    j = _client().post("/compare", json={"prompt": "Describe a chair.", "seed": 0}).json()
    assert j["baseline"] and j["steered"] and j["baseline"] != j["steered"]
    assert j["layer"] == 4 and j["alpha"] == 8.0


# --- API user chooses just L17 (default) or the bracket ------------------------------------

def test_variants_endpoint_lists_choices():
    j = _bracket_client().get("/variants").json()
    assert j["default"] == "L4" and set(j["variants"]) == {"L4", "bracket"}


def test_generate_defaults_to_single_variant():
    j = _bracket_client().post("/generate", json={"prompt": "p"}).json()
    assert j["variant"] == "L4" and len(j["layers"]) == 1


def test_generate_can_choose_the_bracket():
    j = _bracket_client().post("/generate", json={"prompt": "p", "variant": "bracket"}).json()
    assert j["variant"] == "bracket"
    assert {x["layer"] for x in j["layers"]} == {2, 4, 6}


def test_unknown_variant_is_400():
    r = _bracket_client().post("/generate", json={"prompt": "p", "variant": "nope"})
    assert r.status_code == 400
