"""FastAPI server tests.

All tests use ``TestClient`` against the simplified server (demo packs only)
with a temporary ``feedback_human_path`` so no shared state leaks.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from brainstormlm.api import server as server_module
from brainstormlm.api.server import build_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback_human.jsonl"
    log_path = tmp_path / "logs" / "server.jsonl"

    settings = server_module.get_settings()
    monkeypatch.setattr(settings, "feedback_human_path", feedback_path)
    monkeypatch.setattr(settings, "server_log_path", log_path)
    monkeypatch.setattr(settings, "rate_limit_capacity", 1000)

    app = build_app()
    return TestClient(app), feedback_path


def test_healthz(client):
    tc, _ = client
    res = tc.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True


def test_brainstorm_returns_ideas(client):
    tc, _ = client
    res = tc.post("/brainstorm", json={"prompt": "grow to 10M users", "n_ideas": 3})
    assert res.status_code == 200
    ideas = res.json()["ideas"]
    assert len(ideas) >= 3
    for idea in ideas:
        assert idea["trajectory_id"]
        assert idea["summary"]
        assert idea["content"]


def test_brainstorm_persists_trajectories(client):
    tc, feedback_path = client
    tc.post("/brainstorm", json={"prompt": "test persist", "n_ideas": 2})
    assert feedback_path.exists()
    rows = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 2


def test_brainstorm_rejects_oversized_prompt(client):
    tc, _ = client
    res = tc.post("/brainstorm", json={"prompt": "x" * 5000, "n_ideas": 2})
    assert res.status_code == 400


def test_feedback_attaches_rating(client):
    tc, feedback_path = client
    brainstorm_res = tc.post("/brainstorm", json={"prompt": "feedback test", "n_ideas": 1})
    tid = brainstorm_res.json()["ideas"][0]["trajectory_id"]

    res = tc.post(
        "/feedback",
        json={"trajectory_id": tid, "thumbs": "up", "rating": 5, "critique": "nice"},
    )
    assert res.status_code == 200

    rows = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
    rated = [r for r in rows if r.get("human_rating") is not None]
    assert len(rated) == 1
    assert rated[0]["human_rating"]["thumbs"] == "up"


def test_feedback_triggers_memory_ingestion(client, monkeypatch):
    calls: list[dict] = []

    def fake_ingest(trajectory, *, lessons_path=None):
        calls.append({"trajectory_id": trajectory.id, "thumbs": trajectory.human_rating.thumbs})
        return 1

    monkeypatch.setattr(server_module, "ingest_human_feedback", fake_ingest)

    tc, _ = client
    brainstorm_res = tc.post("/brainstorm", json={"prompt": "ingest test", "n_ideas": 1})
    tid = brainstorm_res.json()["ideas"][0]["trajectory_id"]

    tc.post("/feedback", json={"trajectory_id": tid, "thumbs": "up", "rating": 5})
    assert len(calls) == 1
    assert calls[0]["thumbs"] == "up"


def test_feedback_missing_id_returns_404(client):
    tc, _ = client
    res = tc.post("/feedback", json={"trajectory_id": "nope", "thumbs": "down"})
    assert res.status_code == 404


def test_metrics_endpoint(client):
    tc, _ = client
    res = tc.get("/metrics")
    assert res.status_code == 200
    body = res.json()
    assert "total_requests" in body
    assert "mean_latency_ms" in body
