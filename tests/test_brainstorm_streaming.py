"""Integration tests for POST /brainstorm/stream SSE endpoint.

Verifies that:
1. The endpoint returns a valid SSE stream with correct event ordering.
2. raw_idea events carry trajectory_ids.
3. Final ideas from the done event are feedbackable via POST /feedback.
"""

from __future__ import annotations

import json

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
        return f"Stub idea for: {prompt} at temp {temperature}"

    def generate_with_meta(self, prompt: str, **kw):
        return self.generate(prompt, **kw), False


@pytest.fixture()
def stub_state(tmp_path, monkeypatch):
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
    return {"app": app, "feedback_path": feedback_path}


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if event_type and data:
            events.append({"event": event_type, "data": json.loads(data)})
    return events


def test_stream_returns_sse_with_correct_event_order(stub_state):
    """The stream must produce phase(diverse_pool), raw_idea(s), phase(synthesizing), final_idea(s), done."""
    client = TestClient(stub_state["app"])
    resp = client.post(
        "/brainstorm/stream",
        json={"prompt": "how do we grow?", "n_ideas": 3},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse_events(resp.text)
    event_types = [e["event"] for e in events]

    assert event_types[0] == "phase"
    assert events[0]["data"]["phase"] == "diverse_pool"

    raw_ideas = [e for e in events if e["event"] == "raw_idea"]
    assert len(raw_ideas) >= 3

    synth_idx = next(i for i, e in enumerate(events) if e["event"] == "phase" and e["data"].get("phase") == "synthesizing")
    first_raw_idx = event_types.index("raw_idea")
    assert synth_idx > first_raw_idx

    assert event_types[-1] == "done"


def test_stream_raw_ideas_have_trajectory_ids(stub_state):
    """Each raw_idea event must carry a trajectory_id."""
    client = TestClient(stub_state["app"])
    resp = client.post(
        "/brainstorm/stream",
        json={"prompt": "test trajectories", "n_ideas": 2},
    )
    events = _parse_sse_events(resp.text)
    raw_ideas = [e for e in events if e["event"] == "raw_idea"]

    for idea in raw_ideas:
        tid = idea["data"].get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0


def test_stream_final_ideas_are_feedbackable(stub_state):
    """trajectory_ids from final_idea events must be accepted by POST /feedback."""
    client = TestClient(stub_state["app"])
    resp = client.post(
        "/brainstorm/stream",
        json={"prompt": "feedback test", "n_ideas": 2},
    )
    events = _parse_sse_events(resp.text)
    final_ideas = [e for e in events if e["event"] == "final_idea"]
    assert len(final_ideas) >= 2

    tid = final_ideas[0]["data"]["trajectory_id"]
    feedback_resp = client.post(
        "/feedback",
        json={"trajectory_id": tid, "thumbs": "up"},
    )
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["ok"] is True


def test_stream_demo_mode(tmp_path, monkeypatch):
    """In demo mode, /brainstorm/stream should still return valid SSE events."""
    settings = server_module.get_settings()
    monkeypatch.setattr(settings, "feedback_human_path", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(settings, "promoted_adapter_path", tmp_path / "promoted.json")
    monkeypatch.setattr(settings, "ab_config_path", tmp_path / "ab.json")
    monkeypatch.setattr(settings, "server_log_path", tmp_path / "logs" / "server.jsonl")
    monkeypatch.setattr(settings, "rate_limit_capacity", 1000)
    monkeypatch.setattr(server_module, "_DEMO_MODE", True)

    app = build_app()
    client = TestClient(app)

    resp = client.post(
        "/brainstorm/stream",
        json={"prompt": "demo streaming test", "n_ideas": 4},
    )
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)

    event_types = [e["event"] for e in events]
    assert "phase" in event_types
    assert "raw_idea" in event_types
    assert "done" in event_types
