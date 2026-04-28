"""Pair-mining over human-rated trajectories (plan-06)."""

from __future__ import annotations

from brainstormlm.feedback.trajectories import HumanRating, Trajectory
from brainstormlm.training.preferences import mine_human_pairs


def _row(
    id_: str,
    prompt: str,
    response: str,
    *,
    domain: str = "science.physics",
    thumbs: str | None = None,
    rating: int | None = None,
    critique: str | None = None,
) -> Trajectory:
    human_rating = (
        HumanRating(
            thumbs=thumbs,  # type: ignore[arg-type]
            rating=rating,
            critique=critique,
            rated_at="2026-04-27T00:00:00Z",
        )
        if thumbs is not None
        else None
    )
    return Trajectory(
        id=id_,
        timestamp="2026-04-27T00:00:00Z",
        domain=domain,
        prompt=prompt,
        response=response,
        generator_model="Qwen/Qwen2.5-7B-Instruct",
        generator_temperature=1.0,
        prompter_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        creativity=3,
        logic=3,
        rationale="ok",
        source="human",
        human_rating=human_rating,
    )


def test_mine_human_pairs_picks_thumbs_up_vs_down_per_prompt():
    rows = [
        _row("u1", "P1", "loved", thumbs="up", rating=5),
        _row("d1", "P1", "hated", thumbs="down", rating=1),
    ]
    pairs = mine_human_pairs(rows, sibling_run_id="human-batch-1")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.prompt == "P1"
    assert pair.chosen == "loved"
    assert pair.rejected == "hated"
    assert pair.sibling_run_id == "human-batch-1"


def test_mine_human_pairs_skips_prompt_without_both_arms():
    rows = [
        _row("u1", "P1", "good", thumbs="up"),
        _row("u2", "P1", "also good", thumbs="up"),
    ]
    assert mine_human_pairs(rows, sibling_run_id="batch") == []


def test_mine_human_pairs_skips_unrated_rows():
    rows = [
        _row("nr1", "P1", "no rating", thumbs=None),
        _row("d1", "P1", "downvoted", thumbs="down"),
    ]
    assert mine_human_pairs(rows, sibling_run_id="batch") == []


def test_mine_human_pairs_rating_breaks_ties():
    rows = [
        _row("u1", "P1", "ok yay", thumbs="up", rating=3),
        _row("u2", "P1", "great yay", thumbs="up", rating=5),
        _row("d1", "P1", "meh", thumbs="down", rating=2),
        _row("d2", "P1", "awful", thumbs="down", rating=1),
    ]
    pairs = mine_human_pairs(rows, sibling_run_id="batch")
    assert len(pairs) == 1
    assert pairs[0].chosen == "great yay"
    assert pairs[0].rejected == "awful"


def test_mine_human_pairs_one_pair_per_prompt():
    rows = [
        _row("u1", "P1", "good", thumbs="up"),
        _row("d1", "P1", "bad", thumbs="down"),
        _row("u2", "P2", "good2", thumbs="up"),
        _row("d2", "P2", "bad2", thumbs="down"),
    ]
    pairs = mine_human_pairs(rows, sibling_run_id="batch")
    assert len(pairs) == 2
    prompts = {p.prompt for p in pairs}
    assert prompts == {"P1", "P2"}


def test_mine_human_pairs_skips_identical_responses():
    rows = [
        _row("u1", "P1", "same text", thumbs="up"),
        _row("d1", "P1", "same text", thumbs="down"),
    ]
    assert mine_human_pairs(rows, sibling_run_id="batch") == []
