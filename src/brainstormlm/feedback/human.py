"""Append-only JSONL store for human-rated trajectories (plan-06).

Mirrors `feedback/trajectories.py`: schema is the existing `Trajectory`
(extended with `source` + `human_rating`). Old rows still validate.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

from brainstormlm.config import get_settings
from brainstormlm.feedback.trajectories import HumanRating, Trajectory


def _resolve_path(path: Path | None) -> Path:
    return path if path is not None else get_settings().feedback_human_path


def utc_now_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def new_trajectory_id() -> str:
    return str(uuid.uuid4())


def append_human(t: Trajectory, *, path: Path | None = None) -> None:
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(t.model_dump(), sort_keys=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_all_human(*, path: Path | None = None) -> Iterator[Trajectory]:
    target = _resolve_path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Trajectory.model_validate_json(line)


def find_by_id(trajectory_id: str, *, path: Path | None = None) -> Trajectory | None:
    for row in read_all_human(path=path):
        if row.id == trajectory_id:
            return row
    return None


def attach_rating(
    trajectory_id: str,
    rating: HumanRating,
    *,
    path: Path | None = None,
) -> Trajectory:
    """Find a row by id, append a new row that carries the rating, return it.

    The original row is NOT mutated — append-only discipline.
    """
    original = find_by_id(trajectory_id, path=path)
    if original is None:
        raise KeyError(f"No human-feedback row with id={trajectory_id!r}")
    rated = original.model_copy(update={"source": "human", "human_rating": rating})
    append_human(rated, path=path)
    return rated
