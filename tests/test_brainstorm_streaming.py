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


def _parse_sse_events(raw_text: str) -> list[dict]:
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


def test_stream_returns_sse_with_correct_event_order(client):
    tc, _ = client
    resp = tc.post(
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


def test_stream_raw_ideas_have_trajectory_ids(client):
    tc, _ = client
    resp = tc.post(
        "/brainstorm/stream",
        json={"prompt": "test trajectories", "n_ideas": 2},
    )
    events = _parse_sse_events(resp.text)
    raw_ideas = [e for e in events if e["event"] == "raw_idea"]

    for idea in raw_ideas:
        tid = idea["data"].get("trajectory_id")
        assert tid and isinstance(tid, str) and len(tid) > 0


def test_stream_final_ideas_are_feedbackable(client):
    tc, _ = client
    resp = tc.post(
        "/brainstorm/stream",
        json={"prompt": "feedback test", "n_ideas": 2},
    )
    events = _parse_sse_events(resp.text)
    final_ideas = [e for e in events if e["event"] == "final_idea"]
    assert len(final_ideas) >= 2

    tid = final_ideas[0]["data"]["trajectory_id"]
    feedback_resp = tc.post(
        "/feedback",
        json={"trajectory_id": tid, "thumbs": "up"},
    )
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["ok"] is True


def test_stream_demo_mode(client):
    tc, _ = client
    resp = tc.post(
        "/brainstorm/stream",
        json={"prompt": "demo streaming test", "n_ideas": 4},
    )
    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)

    event_types = [e["event"] for e in events]
    assert "phase" in event_types
    assert "raw_idea" in event_types
    assert "done" in event_types
