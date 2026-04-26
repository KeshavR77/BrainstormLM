"""Pure-parse tests for memory.lessons.extract_lessons (no network)."""

from __future__ import annotations

from brainstormlm.clients import openai as openai_client
from brainstormlm.clients.openai import Usage
from brainstormlm.feedback.trajectories import Trajectory
from brainstormlm.memory.lessons import (
    EXTRACTION_VERSION,
    Lesson,
    extract_lessons,
    make_lesson_id,
)


def _sample_trajectory(id_: str = "traj-1") -> Trajectory:
    return Trajectory(
        id=id_,
        timestamp="2026-04-25T00:00:00Z",
        domain="science.physics",
        prompt="why is the sky blue?",
        response="rayleigh scattering scatters short wavelengths more strongly",
        generator_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=4,
        logic=4,
        rationale="Concrete physical mechanism, well-grounded.",
        failure_modes=["could-be-deeper"],
    )


class _CallRecorder:
    """Stand-in for openai_client.complete_json that records call args."""

    def __init__(self, payload: dict, usage: Usage | None = None):
        self.payload = payload
        self.usage = usage or Usage(prompt_tokens=42, completion_tokens=17)
        self.calls: list[dict] = []

    def __call__(self, system, messages, model, schema, **kwargs):
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "model": model,
                "schema": schema,
                "kwargs": kwargs,
            }
        )
        return self.payload, self.usage


def test_extract_lessons_two_items_success_and_failure(monkeypatch):
    payload = {
        "lessons": [
            {
                "lesson": "Ground abstract questions in a concrete physical mechanism.",
                "kind": "success",
            },
            {
                "lesson": "Avoid restating the question without adding mechanism.",
                "kind": "failure",
            },
        ]
    }
    rec = _CallRecorder(payload)
    monkeypatch.setattr(openai_client, "complete_json", rec)

    t = _sample_trajectory("traj-1")
    out = extract_lessons(t)

    assert len(out) == 2
    assert all(isinstance(item, Lesson) for item in out)

    success, failure = out[0], out[1]

    assert success.kind == "success"
    assert success.lesson.startswith("Ground abstract")
    assert success.domain == t.domain
    assert success.source_trajectory_id == t.id
    assert success.extraction_version == EXTRACTION_VERSION
    assert success.extraction_version == 1
    assert success.extractor_model  # non-empty
    assert success.id == make_lesson_id(success.lesson, "success", t.id)

    assert failure.kind == "failure"
    assert failure.domain == t.domain
    assert failure.source_trajectory_id == t.id
    assert failure.id == make_lesson_id(failure.lesson, "failure", t.id)

    # IDs are deterministic and distinct.
    again = extract_lessons(t)
    assert [item.id for item in again] == [item.id for item in out]
    assert success.id != failure.id


def test_extract_lessons_one_item_boundary(monkeypatch):
    payload = {
        "lessons": [
            {"lesson": "One clear pattern is enough.", "kind": "success"},
        ]
    }
    monkeypatch.setattr(openai_client, "complete_json", _CallRecorder(payload))

    out = extract_lessons(_sample_trajectory("traj-low"))
    assert len(out) == 1
    assert out[0].kind == "success"


def test_extract_lessons_three_items_boundary(monkeypatch):
    payload = {
        "lessons": [
            {"lesson": "Pattern A.", "kind": "success"},
            {"lesson": "Pattern B.", "kind": "failure"},
            {"lesson": "Pattern C.", "kind": "success"},
        ]
    }
    monkeypatch.setattr(openai_client, "complete_json", _CallRecorder(payload))

    out = extract_lessons(_sample_trajectory("traj-3"))
    assert len(out) == 3
    assert [item.kind for item in out] == ["success", "failure", "success"]
    assert len({item.id for item in out}) == 3


def test_extract_lessons_passes_trajectory_context_in_user_message(monkeypatch):
    payload = {"lessons": [{"lesson": "x", "kind": "success"}]}
    rec = _CallRecorder(payload)
    monkeypatch.setattr(openai_client, "complete_json", rec)

    t = _sample_trajectory("traj-ctx")
    extract_lessons(t)

    assert len(rec.calls) == 1
    call = rec.calls[0]

    messages = call["messages"]
    assert isinstance(messages, list) and len(messages) >= 1
    user_blob = "\n".join(
        m["content"] for m in messages if m.get("role") == "user"
    )

    assert t.prompt in user_blob
    assert t.response in user_blob
    assert str(t.creativity) in user_blob
    assert str(t.logic) in user_blob
    assert t.rationale in user_blob
    for fm in t.failure_modes:
        assert fm in user_blob


def test_extract_lessons_schema_is_strict(monkeypatch):
    payload = {"lessons": [{"lesson": "x", "kind": "success"}]}
    rec = _CallRecorder(payload)
    monkeypatch.setattr(openai_client, "complete_json", rec)

    extract_lessons(_sample_trajectory("traj-schema"))
    schema = rec.calls[0]["schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["lessons"]

    lessons_schema = schema["properties"]["lessons"]
    assert lessons_schema["type"] == "array"
    assert lessons_schema["minItems"] == 1
    assert lessons_schema["maxItems"] == 3

    item = lessons_schema["items"]
    assert item["type"] == "object"
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"lesson", "kind"}
    assert item["properties"]["kind"]["enum"] == ["success", "failure"]


def test_extract_lessons_uses_configured_model(monkeypatch):
    payload = {"lessons": [{"lesson": "x", "kind": "success"}]}
    rec = _CallRecorder(payload)
    monkeypatch.setattr(openai_client, "complete_json", rec)

    out = extract_lessons(_sample_trajectory("traj-m"))
    assert rec.calls[0]["model"] == out[0].extractor_model


def test_extract_lessons_extracted_at_is_iso_utc(monkeypatch):
    payload = {"lessons": [{"lesson": "x", "kind": "success"}]}
    monkeypatch.setattr(openai_client, "complete_json", _CallRecorder(payload))

    out = extract_lessons(_sample_trajectory("traj-time"))
    ts = out[0].extracted_at
    assert ts.endswith("Z")
    assert "T" in ts
