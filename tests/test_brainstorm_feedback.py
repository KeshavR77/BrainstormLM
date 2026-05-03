"""Integration tests for the RLHF feedback loop on /brainstorm (plan-08).

Verifies that:
1. POST /brainstorm returns a non-empty trajectory_id for each idea.
2. Those trajectory_ids can be fed to POST /feedback and get a 200 OK.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from brainstormlm.api import server as server_module
from brainstormlm.api.server import build_app


class _StubGenerator:
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        return f"Stub idea for: {prompt}"

    def generate_with_meta(self, prompt: str, **kw):
        return self.generate(prompt, **kw), False


@pytest.fixture()
def stub_state(tmp_path, monkeypatch):
    """App with stubbed generator and temp paths; DEMO mode OFF so trajectories go through real path."""
    feedback_path = tmp_path / "feedback_human.jsonl"
    promoted_path = tmp_path / "promoted.json"
    ab_path = tmp_path / "ab_config.json"
    log_path = tmp_path / "logs" / "server.jsonl"

    settings = server_module.get_settings()
    monkeypatch.setattr(settings, "feedback_human_path", feedback_path)
    monkeypatch.setattr(settings, "promoted_adapter_path", promoted_path)
    monkeypatch.setattr(settings, "ab_config_path", ab_path)
    monkeypatch.setattr(settings, "server_log_path", log_path)
    monkeypatch.setattr(settings, "rate_limit_capacity", 1000)

    # Ensure demo mode is off so the real Trajectory path is exercised
    monkeypatch.setattr(server_module, "_DEMO_MODE", False)
    monkeypatch.setattr(server_module, "_ML_AVAILABLE", True)

    stub = _StubGenerator()

    def _fake_make_generator(
        adapter_path=None,
        *,
        steering_path=None,
        steering_strength=None,
        steering_layers=None,
        corpius_config_path=None,
    ):
        return stub

    app = build_app(make_generator=_fake_make_generator)
    return {
        "app": app,
        "feedback_path": feedback_path,
    }


def test_brainstorm_returns_trajectory_ids(stub_state):
    """Every idea in the /brainstorm response must carry a non-empty trajectory_id."""
    client = TestClient(stub_state["app"])
    resp = client.post("/brainstorm", json={"prompt": "how do we grow to 10M users?", "n_ideas": 3})
    assert resp.status_code == 200
    ideas = resp.json()["ideas"]
    assert len(ideas) == 3
    for idea in ideas:
        tid = idea.get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0, (
            f"Expected non-empty trajectory_id, got: {tid!r}"
        )


def test_brainstorm_trajectories_are_feedbackable(stub_state):
    """A trajectory_id from /brainstorm must be accepted by POST /feedback."""
    client = TestClient(stub_state["app"])

    # Generate ideas and grab the first trajectory_id
    brainstorm_resp = client.post(
        "/brainstorm",
        json={"prompt": "what is our best GTM strategy?", "n_ideas": 2},
    )
    assert brainstorm_resp.status_code == 200
    trajectory_id = brainstorm_resp.json()["ideas"][0]["trajectory_id"]
    assert trajectory_id

    # Submit thumbs-up feedback for that trajectory
    feedback_resp = client.post(
        "/feedback",
        json={"trajectory_id": trajectory_id, "thumbs": "up"},
    )
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["ok"] is True


def test_brainstorm_demo_mode_returns_trajectory_ids(tmp_path, monkeypatch):
    """Demo/smoke mode must also return trajectory_ids so the frontend RLHF loop works."""
    settings = server_module.get_settings()
    monkeypatch.setattr(settings, "feedback_human_path", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(settings, "promoted_adapter_path", tmp_path / "promoted.json")
    monkeypatch.setattr(settings, "ab_config_path", tmp_path / "ab.json")
    monkeypatch.setattr(settings, "server_log_path", tmp_path / "logs" / "server.jsonl")
    monkeypatch.setattr(settings, "rate_limit_capacity", 1000)
    monkeypatch.setattr(server_module, "_DEMO_MODE", True)

    app = build_app()
    client = TestClient(app)

    resp = client.post("/brainstorm", json={"prompt": "how do we reduce churn?", "n_ideas": 4})
    assert resp.status_code == 200
    ideas = resp.json()["ideas"]
    assert len(ideas) == 4
    for idea in ideas:
        tid = idea.get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0
