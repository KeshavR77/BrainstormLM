"""Schema-only tests for memory.lessons. Extraction tests live separately."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brainstormlm.memory.lessons import EXTRACTION_VERSION, Lesson, make_lesson_id


def _sample(**overrides) -> Lesson:
    base = dict(
        id=make_lesson_id("be concrete", "success", "traj-1"),
        lesson="be concrete",
        kind="success",
        domain="science.physics",
        source_trajectory_id="traj-1",
        extractor_model="gpt-4o-mini",
        extraction_version=EXTRACTION_VERSION,
        extracted_at="2026-04-25T00:00:00Z",
    )
    base.update(overrides)
    return Lesson(**base)


def test_id_is_deterministic():
    a = make_lesson_id("be concrete", "success", "traj-1")
    b = make_lesson_id("be concrete", "success", "traj-1")
    assert a == b


def test_id_changes_with_kind():
    a = make_lesson_id("be concrete", "success", "traj-1")
    b = make_lesson_id("be concrete", "failure", "traj-1")
    assert a != b


def test_id_changes_with_source():
    a = make_lesson_id("be concrete", "success", "traj-1")
    b = make_lesson_id("be concrete", "success", "traj-2")
    assert a != b


def test_lesson_rejects_extra_fields():
    with pytest.raises(ValidationError):
        _sample(unexpected="boom")


def test_lesson_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        _sample(kind="meh")


def test_lesson_round_trips_through_dict():
    original = _sample()
    again = Lesson.model_validate(original.model_dump())
    assert again == original
