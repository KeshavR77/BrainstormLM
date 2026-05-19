"""Tests for the EQ-Bench Creative Writing v3 subset harness."""

from __future__ import annotations

from brainstormlm.eval.eq_bench_creative import (
    CRITERIA_NEGATIVE,
    CRITERIA_POSITIVE,
    _to_snake,
    aggregate,
    load_prompts,
    parse_judge_payload,
    score_response,
)


def test_load_prompts_returns_ten_items() -> None:
    """The vendored subset has 10 prompts, each with id + category + prompt."""
    prompts = load_prompts()
    assert len(prompts) == 10
    for p in prompts:
        assert {"id", "category", "title", "prompt"}.issubset(p.keys())
        assert "<SEED>" not in p["prompt"]


def test_to_snake_handles_special_chars() -> None:
    assert _to_snake("Adherence to Instructions") == "adherence_to_instructions"
    assert _to_snake("Tell-Don't-Show") == "tell_dont_show"
    assert _to_snake("Consistent Voice/Tone of Writing") == "consistent_voice_tone_of_writing"


def test_parse_judge_payload_perfect_score() -> None:
    """All positives at 10 and all negatives at 0 → composite = 10."""
    payload = {_to_snake(c): 10 for c in CRITERIA_POSITIVE}
    payload.update({_to_snake(c): 0 for c in CRITERIA_NEGATIVE})
    result = parse_judge_payload(payload)
    assert abs(result["composite"] - 10.0) < 1e-9


def test_parse_judge_payload_worst_score() -> None:
    """All positives at 0 and all negatives at 10 → composite = 0."""
    payload = {_to_snake(c): 0 for c in CRITERIA_POSITIVE}
    payload.update({_to_snake(c): 10 for c in CRITERIA_NEGATIVE})
    result = parse_judge_payload(payload)
    assert abs(result["composite"] - 0.0) < 1e-9


def test_parse_judge_payload_clips_out_of_range() -> None:
    """Scores outside [0, 10] are clipped."""
    payload = {_to_snake(c): 15 for c in CRITERIA_POSITIVE}
    payload.update({_to_snake(c): -3 for c in CRITERIA_NEGATIVE})
    result = parse_judge_payload(payload)
    assert abs(result["composite"] - 10.0) < 1e-9


def test_score_response_calls_judge_with_prompt_and_response() -> None:
    """The judge callable receives a prompt containing both the brief and response."""
    captured = {}

    def stub_judge(s: str) -> dict:
        captured["arg"] = s
        return {_to_snake(c): 7 for c in CRITERIA_POSITIVE + CRITERIA_NEGATIVE}

    result = score_response("Write a sad story.", "Once upon a time...", judge=stub_judge)
    assert "Write a sad story" in captured["arg"]
    assert "Once upon a time" in captured["arg"]
    assert "composite" in result


def test_aggregate_means_across_scored() -> None:
    scored = [
        {"criteria": {"a": 5.0}, "composite": 5.0},
        {"criteria": {"a": 7.0}, "composite": 7.0},
        {"criteria": {"a": 9.0}, "composite": 9.0},
    ]
    agg = aggregate(scored)
    assert agg["n"] == 3
    assert abs(agg["mean_composite"] - 7.0) < 1e-9
    assert abs(agg["per_criterion_mean"]["a"] - 7.0) < 1e-9


def test_aggregate_empty() -> None:
    assert aggregate([])["n"] == 0
