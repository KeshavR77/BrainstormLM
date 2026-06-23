"""FastAPI surface for the steering harness — health/package/generate/compare, driven on a
MockBackend so it runs offline."""

import numpy as np
from fastapi.testclient import TestClient

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.mock import MockBackend
from creativity_steering.serving.app import create_app
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.service import SteeringService


def _client():
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=4, seed=0)
    vs = SteeringVectorSet(vectors=np.asarray(b.creativity_axis, dtype=np.float32),
                           model_id="mock-qwen", capture_position="mean_response",
                           n_positive=10, n_negative=10)
    svc = SteeringService(b, vs, SteeringPackage(name="v3-L4-a8", vectors_path="(x)", layer=4, alpha=8.0))
    return TestClient(create_app(svc))


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
