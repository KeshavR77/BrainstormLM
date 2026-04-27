"""Schema validation + JSONL round-trip for PreferencePair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from brainstormlm.training.preferences import (
    PreferencePair,
    append,
    pair_id,
    read_all,
)


def _sample_kwargs(**overrides) -> dict:
    base = {
        "id": "deadbeef" * 8,
        "timestamp": "2026-04-25T00:00:00Z",
        "domain": "science.physics",
        "prompt": "Reframe entropy for a 10-year-old.",
        "chosen": "Imagine a tidy room slowly becoming messy.",
        "rejected": "Entropy is a measure of disorder.",
        "chosen_score": 8.0,
        "rejected_score": 4.0,
        "score_gap": 4.0,
        "chosen_temperature": 1.2,
        "rejected_temperature": 0.7,
        "generator_model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "evaluator_model": "gpt-4o-mini",
        "rubric_version": 1,
        "sibling_run_id": "01HZZ-RUN",
    }
    base.update(overrides)
    return base


def test_minimal_pair_validates() -> None:
    p = PreferencePair(**_sample_kwargs())
    assert p.schema_version == 1
    assert p.source == "sibling_sampling"
    assert p.score_gap == 4.0


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        PreferencePair(**_sample_kwargs(unexpected_field="nope"))


def test_pair_id_is_deterministic() -> None:
    a = pair_id("p", "c", "r")
    b = pair_id("p", "c", "r")
    assert a == b
    assert len(a) == 64
    assert pair_id("p", "c", "r") != pair_id("p", "r", "c")


def test_jsonl_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "preferences.jsonl"
    pairs = [
        PreferencePair(**_sample_kwargs(id=f"{i:064x}", prompt=f"prompt {i}")) for i in range(3)
    ]
    for p in pairs:
        append(p, out)

    loaded = list(read_all(out))
    assert len(loaded) == 3
    assert [p.prompt for p in loaded] == ["prompt 0", "prompt 1", "prompt 2"]
    assert all(p.schema_version == 1 for p in loaded)


def test_append_does_not_rewrite(tmp_path: Path) -> None:
    out = tmp_path / "preferences.jsonl"
    append(PreferencePair(**_sample_kwargs(id="a" * 64)), out)
    append(PreferencePair(**_sample_kwargs(id="b" * 64)), out)
    contents = out.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    assert json.loads(contents[0])["id"].startswith("a")
    assert json.loads(contents[1])["id"].startswith("b")


def test_read_all_missing_file_yields_nothing(tmp_path: Path) -> None:
    out = tmp_path / "does_not_exist.jsonl"
    assert list(read_all(out)) == []


def test_read_all_skips_blank_lines(tmp_path: Path) -> None:
    out = tmp_path / "preferences.jsonl"
    append(PreferencePair(**_sample_kwargs(id="c" * 64)), out)
    with out.open("a", encoding="utf-8") as f:
        f.write("\n\n")
    append(PreferencePair(**_sample_kwargs(id="d" * 64)), out)
    loaded = list(read_all(out))
    assert len(loaded) == 2
