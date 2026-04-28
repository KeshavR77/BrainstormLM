"""Trajectory pydantic schema + append-only JSONL writer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: str
    timestamp: str
    domain: str
    prompt: str
    response: str
    generator_model: str
    generator_temperature: float
    prompter_model: str
    evaluator_model: str
    rubric_version: int
    creativity: int = Field(ge=1, le=5)
    logic: int = Field(ge=1, le=5)
    rationale: str
    failure_modes: list[str] = Field(default_factory=list)
    memory_context_ids: list[str] = Field(default_factory=list)
    adapter_path: str | None = None
    steering_path: str | None = None
    steering_strength: float | None = None


def append(t: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(t.model_dump(), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_all(path: Path) -> Iterator[Trajectory]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Trajectory.model_validate_json(line)
