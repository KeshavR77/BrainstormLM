"""Evaluator-vs-human agreement metric (plan-06)."""

from __future__ import annotations

from brainstormlm.feedback.trajectories import HumanRating, Trajectory
from scripts.agreement import compute_agreement


def _row(
    *,
    creativity: int,
    logic: int,
    thumbs: str | None,
    rating: int | None = None,
    domain: str = "tech.ai",
) -> Trajectory:
    rating_obj = (
        HumanRating(
            thumbs=thumbs,  # type: ignore[arg-type]
            rating=rating,
            rated_at="2026-04-27T00:00:00Z",
        )
        if thumbs is not None
        else None
    )
    return Trajectory(
        id=f"r-{creativity}-{logic}-{thumbs}",
        timestamp="2026-04-27T00:00:00Z",
        domain=domain,
        prompt="p",
        response="r",
        generator_model="g",
        generator_temperature=1.0,
        prompter_model="p",
        evaluator_model="e",
        rubric_version=1,
        creativity=creativity,
        logic=logic,
        rationale="ok",
        source="human",
        human_rating=rating_obj,
    )


def test_perfect_signed_agreement():
    rows = [
        _row(creativity=4, logic=4, thumbs="up"),    # total 8 ≥ 7, up → match
        _row(creativity=3, logic=4, thumbs="up"),    # total 7 ≥ 7, up → match
        _row(creativity=2, logic=2, thumbs="down"),  # total 4 < 7, down → match
    ]
    result = compute_agreement(rows)
    assert result["n"] == 3
    assert result["signed_agreement"] == 1.0


def test_zero_signed_agreement():
    rows = [
        _row(creativity=4, logic=4, thumbs="down"),  # high score, thumbs down
        _row(creativity=2, logic=2, thumbs="up"),    # low score, thumbs up
    ]
    result = compute_agreement(rows)
    assert result["n"] == 2
    assert result["signed_agreement"] == 0.0


def test_skips_rows_without_rating():
    rows = [
        _row(creativity=4, logic=4, thumbs=None),
        _row(creativity=2, logic=2, thumbs="down"),
    ]
    result = compute_agreement(rows)
    assert result["n"] == 1


def test_pearson_on_correlated_fixture():
    rows = [
        _row(creativity=5, logic=5, thumbs="up", rating=5),
        _row(creativity=4, logic=4, thumbs="up", rating=4),
        _row(creativity=2, logic=2, thumbs="down", rating=2),
        _row(creativity=1, logic=1, thumbs="down", rating=1),
    ]
    result = compute_agreement(rows)
    assert result["pearson"] is not None
    assert result["pearson"] > 0.95


def test_per_domain_breakdown():
    rows = [
        _row(creativity=4, logic=4, thumbs="up", domain="tech.ai"),
        _row(creativity=2, logic=2, thumbs="down", domain="tech.ai"),
        _row(creativity=4, logic=4, thumbs="down", domain="art.visual"),
    ]
    result = compute_agreement(rows)
    assert "tech.ai" in result["per_domain"]
    assert "art.visual" in result["per_domain"]
    assert result["per_domain"]["tech.ai"]["signed_agreement"] == 1.0
    assert result["per_domain"]["art.visual"]["signed_agreement"] == 0.0


def test_empty_input_returns_zero_n():
    result = compute_agreement([])
    assert result["n"] == 0
    assert result["signed_agreement"] is None
    assert result["pearson"] is None
