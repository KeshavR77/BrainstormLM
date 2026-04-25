import json

import pytest
from pydantic import ValidationError

from brainstormlm.feedback.trajectories import Trajectory, append, read_all


def _sample(id_: str = "t1") -> Trajectory:
    return Trajectory(
        id=id_,
        timestamp="2026-04-25T00:00:00Z",
        domain="science.physics",
        prompt="why is the sky blue?",
        response="rayleigh scattering",
        generator_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=3,
        logic=4,
        rationale="ok",
        failure_modes=["shallow"],
    )


def test_round_trip_three_trajectories(tmp_path):
    path = tmp_path / "out" / "trajectories.jsonl"
    items = [_sample(f"t{i}") for i in range(3)]
    for t in items:
        append(t, path)
    loaded = list(read_all(path))
    assert loaded == items


def test_appends_one_object_per_line(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    append(_sample("a"), path)
    append(_sample("b"), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is independently valid JSON


def test_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Trajectory(
            id="x",
            timestamp="t",
            domain="d",
            prompt="p",
            response="r",
            generator_model="g",
            generator_temperature=1.0,
            prompter_model="p",
            evaluator_model="e",
            rubric_version=1,
            creativity=3,
            logic=3,
            rationale="ok",
            unexpected_field="boom",
        )


def test_schema_clamps_score_range():
    with pytest.raises(ValidationError):
        Trajectory(
            id="x",
            timestamp="t",
            domain="d",
            prompt="p",
            response="r",
            generator_model="g",
            generator_temperature=1.0,
            prompter_model="p",
            evaluator_model="e",
            rubric_version=1,
            creativity=6,
            logic=3,
            rationale="ok",
        )


def test_read_all_on_missing_path_yields_empty(tmp_path):
    assert list(read_all(tmp_path / "does-not-exist.jsonl")) == []


def test_read_all_skips_blank_lines(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    append(_sample("a"), path)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
    append(_sample("b"), path)
    loaded = list(read_all(path))
    assert [t.id for t in loaded] == ["a", "b"]
