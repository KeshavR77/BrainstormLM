import pytest
from pydantic import ValidationError

from brainstormlm.feedback.evaluator import EvaluatorResult, parse


def test_parses_canned_payload():
    payload = {
        "creativity": 4,
        "logic": 3,
        "rationale": "Some unusual ideas, but a couple drifted off-topic.",
        "failure_modes": ["off-topic"],
    }
    result = parse(payload)
    assert result.creativity == 4
    assert result.logic == 3
    assert "drifted" in result.rationale
    assert result.failure_modes == ["off-topic"]


def test_parses_with_empty_failure_modes():
    result = parse(
        {
            "creativity": 5,
            "logic": 5,
            "rationale": "Excellent across the board.",
            "failure_modes": [],
        }
    )
    assert result.failure_modes == []


def test_rejects_extra_field():
    with pytest.raises(ValidationError):
        parse(
            {
                "creativity": 3,
                "logic": 3,
                "rationale": "ok",
                "failure_modes": [],
                "smuggled": "value",
            }
        )


def test_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        parse(
            {
                "creativity": 0,
                "logic": 3,
                "rationale": "ok",
                "failure_modes": [],
            }
        )


def test_evaluator_result_round_trips_through_dict():
    original = EvaluatorResult(
        creativity=2, logic=4, rationale="meh", failure_modes=["shallow"]
    )
    again = parse(original.model_dump())
    assert again == original
