"""Continuous-training orchestrator tests (plan-06).

The four phases — mining, sibling-sampling, training, eval — are stubbed.
We assert the orchestrator (a) calls them in order, (b) skips phases under
the appropriate flags, and (c) only writes the promoted-pointer when the
gate passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brainstormlm.feedback.trajectories import HumanRating, Trajectory
from brainstormlm.training.preferences import PreferencePair
from brainstormlm.training.promotion import read_promoted
from scripts.continuous_train import (
    ContinuousTrainConfig,
    Phases,
    run_pipeline,
)


def _human_row(prompt: str, response: str, thumbs: str) -> Trajectory:
    return Trajectory(
        id=f"{prompt}-{response}-{thumbs}",
        timestamp="2026-04-27T00:00:00Z",
        domain="science.physics",
        prompt=prompt,
        response=response,
        generator_model="g",
        generator_temperature=1.0,
        prompter_model="p",
        evaluator_model="e",
        rubric_version=1,
        creativity=4,
        logic=4,
        rationale="ok",
        source="human",
        human_rating=HumanRating(
            thumbs=thumbs,  # type: ignore[arg-type]
            rated_at="2026-04-27T00:00:00Z",
        ),
    )


def _summaries(total: float, p: float = 0.05) -> list[dict]:
    return [
        {"arm": "base", "n": 50, "mean_creativity": 3.5, "mean_logic": 3.5,
         "mean_total": 7.0, "mean_delta_vs_base": None, "p_value": None},
        {"arm": "trained+mem", "n": 50, "mean_creativity": total / 2,
         "mean_logic": total / 2, "mean_total": total,
         "mean_delta_vs_base": total - 7.0, "p_value": p},
    ]


@pytest.fixture
def stub_phases(tmp_path):
    calls: list[str] = []

    def stub_load_human() -> list[Trajectory]:
        calls.append("load_human")
        return [
            _human_row("P1", "loved", "up"),
            _human_row("P1", "hated", "down"),
        ]

    def stub_sibling_sampling() -> list[PreferencePair]:
        calls.append("sibling_sampling")
        return []  # synthetic side adds nothing in this fixture

    def stub_train(pairs: list[PreferencePair], *, run_id: str) -> Path:
        calls.append("train")
        out = tmp_path / "adapters" / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text("{}")
        return out

    def stub_eval(adapter_path: Path) -> list[dict]:
        calls.append("eval")
        return _summaries(total=8.0, p=0.05)

    phases = Phases(
        load_human_feedback=stub_load_human,
        sibling_sampling=stub_sibling_sampling,
        train=stub_train,
        evaluate=stub_eval,
    )
    return phases, calls


def test_full_pipeline_runs_all_phases_in_order(tmp_path, stub_phases):
    phases, calls = stub_phases
    promoted_path = tmp_path / "promoted.json"
    cfg = ContinuousTrainConfig(
        run_id="run-1",
        promoted_path=promoted_path,
        skip_mining=False,
        skip_training=False,
        dry_run=False,
    )
    result = run_pipeline(cfg, phases=phases)
    assert calls == ["load_human", "sibling_sampling", "train", "eval"]
    assert result["promoted"] is True
    assert read_promoted(promoted_path).run_id == "run-1"


def test_dry_run_calls_no_phases(tmp_path, stub_phases):
    phases, calls = stub_phases
    cfg = ContinuousTrainConfig(
        run_id="run-2",
        promoted_path=tmp_path / "promoted.json",
        dry_run=True,
    )
    result = run_pipeline(cfg, phases=phases)
    assert calls == []
    assert result["promoted"] is False
    assert result["reason"] == "dry-run"


def test_skip_mining_skips_both_miners(tmp_path, stub_phases):
    phases, calls = stub_phases
    cfg = ContinuousTrainConfig(
        run_id="run-3",
        promoted_path=tmp_path / "promoted.json",
        skip_mining=True,
    )
    run_pipeline(cfg, phases=phases)
    assert "load_human" not in calls
    assert "sibling_sampling" not in calls
    assert "train" in calls
    assert "eval" in calls


def test_skip_training_reuses_recent_adapter(tmp_path, stub_phases):
    phases, calls = stub_phases

    # Provide an adapter on disk that "exists" already.
    existing = tmp_path / "adapters" / "existing"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text("{}")

    cfg = ContinuousTrainConfig(
        run_id="run-4",
        promoted_path=tmp_path / "promoted.json",
        skip_training=True,
        reuse_adapter=existing,
    )
    run_pipeline(cfg, phases=phases)
    assert "train" not in calls
    assert "eval" in calls


def test_below_gate_does_not_promote(tmp_path):
    calls: list[str] = []

    def low_eval(adapter_path: Path) -> list[dict]:
        calls.append("eval")
        return _summaries(total=7.05)  # below the +0.10 floor

    promoted_path = tmp_path / "promoted.json"
    # Seed with an existing promotion so the gate has a baseline.
    from brainstormlm.training.promotion import Promoted as P, write_promoted_atomic
    write_promoted_atomic(
        P(adapter_path="adapters/old", held_out_total=7.0, run_id="old"),
        promoted_path,
    )

    phases = Phases(
        load_human_feedback=lambda: [],
        sibling_sampling=lambda: [],
        train=lambda pairs, run_id: tmp_path / "adapters" / run_id,
        evaluate=low_eval,
    )

    # Make the train phase create the adapter dir for promotion bookkeeping.
    def fake_train(pairs, run_id):
        out = tmp_path / "adapters" / run_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    phases = phases._replace(train=fake_train) if hasattr(phases, "_replace") else Phases(
        load_human_feedback=phases.load_human_feedback,
        sibling_sampling=phases.sibling_sampling,
        train=fake_train,
        evaluate=phases.evaluate,
    )

    cfg = ContinuousTrainConfig(
        run_id="run-5",
        promoted_path=promoted_path,
    )
    result = run_pipeline(cfg, phases=phases)
    assert result["promoted"] is False

    # Old promotion still in place.
    assert read_promoted(promoted_path).run_id == "old"
