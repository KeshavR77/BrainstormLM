"""Promotion-gate tests (plan-06)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from brainstormlm.training.promotion import (
    Promoted,
    promote_if_better,
    read_promoted,
    write_promoted_atomic,
)


def _summary(arm: str, mean_total: float, p_value: float | None) -> dict:
    return {
        "arm": arm,
        "n": 50,
        "mean_creativity": mean_total / 2,
        "mean_logic": mean_total / 2,
        "mean_total": mean_total,
        "mean_delta_vs_base": None,
        "p_value": p_value,
    }


def _summaries(trained_total: float, trained_p: float | None = 0.05) -> list[dict]:
    return [
        _summary("base", 7.0, None),
        _summary("base+mem", 7.2, 0.10),
        _summary("trained", trained_total - 0.1, 0.07),
        _summary("trained+mem", trained_total, trained_p),
    ]


def test_round_trip_promoted(tmp_path):
    p = Promoted(
        adapter_path="data/adapters/run-1",
        held_out_total=7.5,
        promoted_at="2026-04-27T00:00:00Z",
        run_id="run-1",
    )
    target = tmp_path / "promoted.json"
    write_promoted_atomic(p, target)
    loaded = read_promoted(target)
    assert loaded == p


def test_read_promoted_missing_file_returns_empty():
    p = read_promoted(Path("/tmp/definitely-does-not-exist-9999.json"))
    assert p == Promoted()
    assert p.adapter_path is None


def test_atomic_write_does_not_overwrite_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "promoted.json"
    original = Promoted(
        adapter_path="adapters/old",
        held_out_total=7.0,
        promoted_at="2026-04-27T00:00:00Z",
        run_id="old",
    )
    write_promoted_atomic(original, target)

    def _boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _boom)
    candidate = Promoted(
        adapter_path="adapters/new", held_out_total=8.5, run_id="new"
    )
    with pytest.raises(OSError):
        write_promoted_atomic(candidate, target)

    assert read_promoted(target) == original


def test_promote_if_better_first_promotion(tmp_path):
    promoted, reason = promote_if_better(
        candidate_run_id="run-1",
        candidate_adapter=tmp_path / "adapters/run-1",
        candidate_summary=_summaries(trained_total=7.5),
        current_promoted=Promoted(),
    )
    assert promoted is True
    assert "first promotion" in reason


def test_promote_if_better_below_threshold(tmp_path):
    current = Promoted(
        adapter_path="adapters/old", held_out_total=7.5, run_id="old"
    )
    promoted, reason = promote_if_better(
        candidate_run_id="run-2",
        candidate_adapter=tmp_path / "adapters/run-2",
        candidate_summary=_summaries(trained_total=7.55),  # +0.05 < 0.10
        current_promoted=current,
        min_improvement=0.10,
    )
    assert promoted is False
    assert "improvement" in reason


def test_promote_if_better_above_threshold(tmp_path):
    current = Promoted(adapter_path="adapters/old", held_out_total=7.5, run_id="old")
    promoted, reason = promote_if_better(
        candidate_run_id="run-3",
        candidate_adapter=tmp_path / "adapters/run-3",
        candidate_summary=_summaries(trained_total=7.7, trained_p=0.05),
        current_promoted=current,
    )
    assert promoted is True
    assert "passed gate" in reason


def test_promote_if_better_high_p_value_rejected(tmp_path):
    current = Promoted(adapter_path="adapters/old", held_out_total=7.5, run_id="old")
    promoted, reason = promote_if_better(
        candidate_run_id="run-4",
        candidate_adapter=tmp_path / "adapters/run-4",
        candidate_summary=_summaries(trained_total=7.9, trained_p=0.50),
        current_promoted=current,
        max_p_value=0.20,
    )
    assert promoted is False
    assert "p-value" in reason


def test_promote_if_better_falls_back_to_trained_arm(tmp_path):
    summaries = [
        _summary("base", 7.0, None),
        _summary("trained", 8.0, 0.05),
    ]
    promoted, reason = promote_if_better(
        candidate_run_id="run-5",
        candidate_adapter=tmp_path / "adapters/run-5",
        candidate_summary=summaries,
        current_promoted=Promoted(),
    )
    assert promoted is True
    assert reason


def test_write_promoted_creates_parent_dir(tmp_path):
    target = tmp_path / "deep" / "nested" / "promoted.json"
    p = Promoted(adapter_path="x", held_out_total=1.0, run_id="x")
    write_promoted_atomic(p, target)
    assert target.exists()
    assert json.loads(target.read_text())["adapter_path"] == "x"
