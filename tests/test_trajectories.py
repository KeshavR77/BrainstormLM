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


def test_source_defaults_to_synthetic_and_human_rating_round_trips(tmp_path):
    """plan-06: ``source`` defaults to ``synthetic``; ``human_rating`` round-trips."""
    from brainstormlm.feedback.trajectories import HumanRating

    legacy = _sample("legacy")
    assert legacy.source == "synthetic"
    assert legacy.human_rating is None

    rating = HumanRating(
        thumbs="up",
        rating=5,
        critique="loved this",
        rated_at="2026-04-27T00:00:00Z",
    )
    rated = Trajectory(
        id="rated",
        timestamp="2026-04-27T00:00:00Z",
        domain="art.visual",
        prompt="invent a chair",
        response="a chair shaped like a question mark",
        generator_model="Qwen/Qwen2.5-7B-Instruct",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=4,
        logic=3,
        rationale="ok",
        source="human",
        human_rating=rating,
    )
    path = tmp_path / "trajectories.jsonl"
    append(legacy, path)
    append(rated, path)
    loaded = list(read_all(path))
    assert loaded == [legacy, rated]
    assert loaded[1].source == "human"
    assert loaded[1].human_rating == rating


def test_human_rating_rejects_extra_fields():
    from pydantic import ValidationError as VE
    from brainstormlm.feedback.trajectories import HumanRating

    with pytest.raises(VE):
        HumanRating(
            thumbs="up",
            rated_at="2026-04-27T00:00:00Z",
            unexpected="boom",
        )


def test_human_rating_constrains_thumbs_and_rating():
    from pydantic import ValidationError as VE
    from brainstormlm.feedback.trajectories import HumanRating

    with pytest.raises(VE):
        HumanRating(thumbs="sideways", rated_at="t")
    with pytest.raises(VE):
        HumanRating(thumbs="up", rating=6, rated_at="t")


def test_steering_fields_default_to_none_and_round_trip(tmp_path):
    """Trajectories without steering provenance load fine; new ones round-trip."""
    legacy = _sample("legacy")
    assert legacy.steering_path is None
    assert legacy.steering_strength is None

    steered = Trajectory(
        id="steered-1",
        timestamp="2026-04-27T00:00:00Z",
        domain="art.visual",
        prompt="invent a chair",
        response="a chair shaped like a question mark",
        generator_model="Qwen/Qwen2.5-7B-Instruct",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=4,
        logic=3,
        rationale="ok",
        steering_path="data/steering/run-01/",
        steering_strength=1.0,
    )
    path = tmp_path / "trajectories.jsonl"
    append(legacy, path)
    append(steered, path)
    loaded = list(read_all(path))
    assert loaded == [legacy, steered]
    assert loaded[1].steering_path == "data/steering/run-01/"
    assert loaded[1].steering_strength == 1.0
