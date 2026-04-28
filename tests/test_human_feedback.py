"""Human-feedback JSONL store tests (plan-06)."""

from __future__ import annotations

import pytest

from brainstormlm.feedback.human import (
    append_human,
    attach_rating,
    read_all_human,
)
from brainstormlm.feedback.trajectories import HumanRating, Trajectory


def _trajectory(id_: str = "t1", source: str = "human") -> Trajectory:
    return Trajectory(
        id=id_,
        timestamp="2026-04-27T00:00:00Z",
        domain="science.physics",
        prompt="why is the sky blue?",
        response="rayleigh scattering",
        generator_model="Qwen/Qwen2.5-7B-Instruct",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=3,
        logic=4,
        rationale="ok",
        source=source,  # type: ignore[arg-type]
    )


def test_round_trip_three_human_trajectories(tmp_path):
    path = tmp_path / "feedback_human.jsonl"
    items = [_trajectory(f"t{i}") for i in range(3)]
    for t in items:
        append_human(t, path=path)
    loaded = list(read_all_human(path=path))
    assert loaded == items


def test_attach_rating_appends_new_row_without_mutating_original(tmp_path):
    path = tmp_path / "feedback_human.jsonl"
    original = _trajectory("orig")
    append_human(original, path=path)
    line_count_before = sum(1 for _ in path.read_text().splitlines())

    rating = HumanRating(thumbs="up", rating=5, rated_at="2026-04-27T00:01:00Z")
    rated = attach_rating("orig", rating, path=path)

    line_count_after = sum(1 for _ in path.read_text().splitlines())
    assert line_count_after == line_count_before + 1
    assert rated.id == "orig"
    assert rated.human_rating == rating
    assert rated.source == "human"

    rows = list(read_all_human(path=path))
    assert len(rows) == 2
    assert rows[0].human_rating is None  # original untouched
    assert rows[1].human_rating == rating


def test_attach_rating_missing_id_raises_keyerror(tmp_path):
    path = tmp_path / "feedback_human.jsonl"
    append_human(_trajectory("present"), path=path)
    with pytest.raises(KeyError):
        attach_rating(
            "missing",
            HumanRating(thumbs="down", rated_at="2026-04-27T00:00:00Z"),
            path=path,
        )


def test_read_all_human_on_missing_path_yields_empty(tmp_path):
    assert list(read_all_human(path=tmp_path / "nope.jsonl")) == []
