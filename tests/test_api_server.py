"""FastAPI server tests (plan-06).

All tests use ``TestClient`` against a stubbed generator and a temporary
``feedback_human_path`` so no real model loads and no shared state leaks.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from brainstormlm.api import server as server_module
from brainstormlm.api.server import build_app


class _StubGenerator:
    def __init__(self, response: str = "stubbed answer.") -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response

    def generate_with_meta(self, prompt: str, **kw):
        return self.generate(prompt, **kw), False


@pytest.fixture
def stub_state(tmp_path, monkeypatch):
    """Wire a clean app with stubbed generator factory and temp paths."""
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

    stub = _StubGenerator(response="hello world this is a stub response")

    def _fake_make_generator(adapter_path=None, *, steering_path=None, steering_strength=None, steering_layers=None):
        return stub

    app = build_app(make_generator=_fake_make_generator)
    return {
        "app": app,
        "stub": stub,
        "feedback_path": feedback_path,
        "promoted_path": promoted_path,
    }


def _sse_events(text: str) -> list[dict]:
    """Parse a Server-Sent Events response body into events."""
    events: list[dict] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
    if current:
        events.append(current)
    return events


def test_healthz(stub_state):
    client = TestClient(stub_state["app"])
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "promoted_adapter" in body


def test_ask_streams_meta_token_done(stub_state):
    client = TestClient(stub_state["app"])
    res = client.post("/ask", json={"prompt": "Design a chair", "session_id": "s1"})
    assert res.status_code == 200
    events = _sse_events(res.text)
    types = [e.get("event") for e in events]
    assert types[0] == "meta"
    assert "token" in types
    assert types[-1] == "done"

    meta = json.loads(events[0]["data"])
    done = json.loads(events[-1]["data"])
    assert meta["trajectory_id"] == done["trajectory_id"]
    assert meta["arm"] in {"a", "b"}


def test_ask_persists_trajectory(stub_state):
    client = TestClient(stub_state["app"])
    res = client.post("/ask", json={"prompt": "Brainstorm ice cream flavors", "session_id": "s1"})
    assert res.status_code == 200
    feedback_path = stub_state["feedback_path"]
    assert feedback_path.exists()
    rows = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["source"] == "human"
    assert rows[0]["human_rating"] is None
    assert rows[0]["prompt"] == "Brainstorm ice cream flavors"


def test_ask_rejects_oversized_prompt(stub_state):
    client = TestClient(stub_state["app"])
    res = client.post("/ask", json={"prompt": "x" * 5000, "session_id": "s1"})
    assert res.status_code == 400


def test_ask_rejects_empty_prompt(stub_state):
    client = TestClient(stub_state["app"])
    res = client.post("/ask", json={"prompt": "   ", "session_id": "s1"})
    assert res.status_code == 400


def test_feedback_attaches_rating(stub_state):
    client = TestClient(stub_state["app"])
    ask_res = client.post("/ask", json={"prompt": "name 3 lunch ideas", "session_id": "s1"})
    events = _sse_events(ask_res.text)
    meta = json.loads(events[0]["data"])
    trajectory_id = meta["trajectory_id"]

    res = client.post(
        "/feedback",
        json={"trajectory_id": trajectory_id, "thumbs": "up", "rating": 5, "critique": "nice"},
    )
    assert res.status_code == 200

    rows = [
        json.loads(line)
        for line in stub_state["feedback_path"].read_text().splitlines()
        if line.strip()
    ]
    rated = [r for r in rows if r["human_rating"] is not None]
    assert len(rated) == 1
    assert rated[0]["human_rating"]["thumbs"] == "up"
    assert rated[0]["human_rating"]["rating"] == 5


def test_feedback_missing_id_returns_404(stub_state):
    client = TestClient(stub_state["app"])
    res = client.post(
        "/feedback",
        json={"trajectory_id": "nope", "thumbs": "down"},
    )
    assert res.status_code == 404


def test_metrics_endpoint(stub_state):
    client = TestClient(stub_state["app"])
    client.post("/ask", json={"prompt": "p1", "session_id": "s1"})
    client.post("/ask", json={"prompt": "p2", "session_id": "s2"})
    res = client.get("/metrics")
    assert res.status_code == 200
    body = res.json()
    assert body["total_requests"] >= 2
