"""Integration tests for the RLHF feedback loop on /brainstorm.

Verifies that:
1. POST /brainstorm returns a non-empty trajectory_id for each idea.
2. Those trajectory_ids can be fed to POST /feedback and get a 200 OK.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from brainstormlm.api import server as server_module
from brainstormlm.api.server import build_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback_human.jsonl"
    log_path = tmp_path / "logs" / "server.jsonl"

    settings = server_module.get_settings()
    monkeypatch.setattr(settings, "feedback_human_path", feedback_path)
    monkeypatch.setattr(settings, "server_log_path", log_path)
    monkeypatch.setattr(settings, "rate_limit_capacity", 1000)

    app = build_app()
    return TestClient(app), feedback_path


def test_brainstorm_returns_trajectory_ids(client):
    tc, _ = client
    resp = tc.post("/brainstorm", json={"prompt": "how do we grow to 10M users?", "n_ideas": 3})
    assert resp.status_code == 200
    ideas = resp.json()["ideas"]
    assert len(ideas) >= 3
    for idea in ideas:
        tid = idea.get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0


def test_brainstorm_trajectories_are_feedbackable(client):
    tc, _ = client
    brainstorm_resp = tc.post(
        "/brainstorm",
        json={"prompt": "what is our best GTM strategy?", "n_ideas": 2},
    )
    assert brainstorm_resp.status_code == 200
    trajectory_id = brainstorm_resp.json()["ideas"][0]["trajectory_id"]
    assert trajectory_id

    feedback_resp = tc.post(
        "/feedback",
        json={"trajectory_id": trajectory_id, "thumbs": "up"},
    )
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["ok"] is True


def test_brainstorm_demo_mode_returns_trajectory_ids(client):
    tc, _ = client
    resp = tc.post("/brainstorm", json={"prompt": "how do we reduce churn?", "n_ideas": 4})
    assert resp.status_code == 200
    ideas = resp.json()["ideas"]
    assert len(ideas) == 4
    for idea in ideas:
        tid = idea.get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0
