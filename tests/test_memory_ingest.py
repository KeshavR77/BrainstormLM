"""Human-feedback → ReasoningBank ingestion (plan-06 follow-on)."""

from __future__ import annotations

from pathlib import Path

from brainstormlm.feedback.trajectories import HumanRating, Trajectory
from brainstormlm.memory import ingest as ingest_module
from brainstormlm.memory.ingest import ingest_human_feedback
from brainstormlm.memory.lessons import Lesson


def _make_rated_trajectory(thumbs: str, rating: int | None = 5, critique: str | None = "nice") -> Trajectory:
    return Trajectory(
        id="traj-abc",
        timestamp="2026-05-02T00:00:00Z",
        domain="user",
        prompt="three lunch ideas",
        response="1) ramen 2) tacos 3) salad",
        generator_model="stub",
        generator_temperature=1.0,
        prompter_model="(human)",
        evaluator_model="stub",
        rubric_version=1,
        creativity=1,
        logic=1,
        rationale="(awaiting human feedback)",
        failure_modes=[],
        memory_context_ids=[],
        adapter_path=None,
        steering_path=None,
        source="human",
        human_rating=HumanRating(
            thumbs=thumbs,  # type: ignore[arg-type]
            rating=rating,
            critique=critique,
            rated_at="2026-05-02T00:00:01Z",
        ),
    )


def _make_lesson(text: str, kind: str = "success") -> Lesson:
    return Lesson(
        id=f"lesson-{hash(text) & 0xffff:x}",
        lesson=text,
        kind=kind,  # type: ignore[arg-type]
        domain="user",
        source_trajectory_id="traj-abc",
        extractor_model="stub",
        extraction_version=1,
        extracted_at="2026-05-02T00:00:02Z",
    )


def test_thumbs_up_extracts_and_adds_lessons(tmp_path, monkeypatch):
    lessons_path = tmp_path / "lessons.jsonl"
    extracted: list[Lesson] = [_make_lesson("vary cuisine across suggestions")]
    seen_trajectory: dict = {}

    def fake_extract(t: Trajectory) -> list[Lesson]:
        seen_trajectory["t"] = t
        return extracted

    added: list[list[Lesson]] = []

    class FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == lessons_path

        def load(self) -> None:
            pass

        def add_many(self, lessons: list[Lesson]) -> int:
            added.append(lessons)
            return len(lessons)

    monkeypatch.setattr(ingest_module, "extract_lessons", fake_extract)
    monkeypatch.setattr(ingest_module, "LessonStore", FakeStore)

    rated = _make_rated_trajectory(thumbs="up", rating=5, critique="excellent")
    n = ingest_human_feedback(rated, lessons_path=lessons_path)

    assert n == 1
    assert added == [extracted]
    # Synthesized fields make the extractor see this as a high-quality success.
    seen = seen_trajectory["t"]
    assert seen.creativity == 5
    assert seen.logic == 5
    assert "excellent" in seen.rationale


def test_thumbs_down_skips_ingestion(tmp_path, monkeypatch):
    called = {"extract": 0}

    def fake_extract(t: Trajectory) -> list[Lesson]:
        called["extract"] += 1
        return []

    monkeypatch.setattr(ingest_module, "extract_lessons", fake_extract)

    rated = _make_rated_trajectory(thumbs="down", rating=2, critique="bad")
    n = ingest_human_feedback(rated, lessons_path=tmp_path / "lessons.jsonl")

    assert n == 0
    assert called["extract"] == 0


def test_unrated_trajectory_skips_ingestion(tmp_path, monkeypatch):
    called = {"extract": 0}

    def fake_extract(t: Trajectory) -> list[Lesson]:
        called["extract"] += 1
        return []

    monkeypatch.setattr(ingest_module, "extract_lessons", fake_extract)

    t = _make_rated_trajectory(thumbs="up")
    t = t.model_copy(update={"human_rating": None})
    n = ingest_human_feedback(t, lessons_path=tmp_path / "lessons.jsonl")

    assert n == 0
    assert called["extract"] == 0


def test_thumbs_up_without_rating_uses_default_scores(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_extract(t: Trajectory) -> list[Lesson]:
        seen["t"] = t
        return [_make_lesson("be specific")]

    class FakeStore:
        def __init__(self, path: Path) -> None: ...
        def load(self) -> None: ...
        def add_many(self, lessons: list[Lesson]) -> int:
            return len(lessons)

    monkeypatch.setattr(ingest_module, "extract_lessons", fake_extract)
    monkeypatch.setattr(ingest_module, "LessonStore", FakeStore)

    rated = _make_rated_trajectory(thumbs="up", rating=None, critique=None)
    n = ingest_human_feedback(rated, lessons_path=tmp_path / "lessons.jsonl")

    assert n == 1
    # No numeric rating → default to 4/5 range so extractor reads success.
    assert seen["t"].creativity >= 4
    assert seen["t"].logic >= 4
    assert "thumbs-up" in seen["t"].rationale.lower() or "thumbs up" in seen["t"].rationale.lower()


def test_extraction_failure_is_swallowed(tmp_path, monkeypatch):
    """Memory ingestion must not crash the caller — it runs in a background task."""

    def fake_extract(t: Trajectory) -> list[Lesson]:
        raise RuntimeError("openai down")

    monkeypatch.setattr(ingest_module, "extract_lessons", fake_extract)

    rated = _make_rated_trajectory(thumbs="up")
    # Should not raise.
    n = ingest_human_feedback(rated, lessons_path=tmp_path / "lessons.jsonl")
    assert n == 0
