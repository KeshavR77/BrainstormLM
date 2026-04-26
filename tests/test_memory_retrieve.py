"""Retrieve + format_memory_block + augmented_system unit tests (no network)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from brainstormlm.clients.openai import Usage
from brainstormlm.memory import store as store_module
from brainstormlm.memory.lessons import EXTRACTION_VERSION, Lesson, make_lesson_id
from brainstormlm.memory.retrieve import (
    augmented_system,
    format_memory_block,
    retrieve,
)
from brainstormlm.memory.store import LessonStore


def _deterministic_embed(text: str, dim: int = 8) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        norm = 1.0
    return vec / norm


@pytest.fixture
def fake_embed(monkeypatch):
    def fake(texts: list[str], *, model: str | None = None, use_cache: bool = True):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), Usage(0, 0, cached=True)
        matrix = np.stack([_deterministic_embed(t) for t in texts], axis=0)
        return matrix, Usage(prompt_tokens=0, completion_tokens=0, cached=False)

    monkeypatch.setattr(store_module, "embed", fake)


def _lesson(text: str, kind: str = "success", source: str | None = None) -> Lesson:
    source = source or f"traj-{text}"
    return Lesson(
        id=make_lesson_id(text, kind, source),
        lesson=text,
        kind=kind,  # type: ignore[arg-type]
        domain="science.physics",
        source_trajectory_id=source,
        extractor_model="gpt-4o-mini",
        extraction_version=EXTRACTION_VERSION,
        extracted_at="2026-04-25T00:00:00Z",
    )


def _populated_store(tmp_path, lessons: list[Lesson]) -> LessonStore:
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.load()
    store.add_many(lessons)
    return store


def test_retrieve_mixes_success_and_failure(tmp_path, fake_embed):
    lessons = [
        _lesson("good-1", kind="success"),
        _lesson("good-2", kind="success"),
        _lesson("good-3", kind="success"),
        _lesson("bad-1", kind="failure"),
        _lesson("bad-2", kind="failure"),
        _lesson("bad-3", kind="failure"),
    ]
    store = _populated_store(tmp_path, lessons)

    out = retrieve(store, "query", k_success=2, k_failure=1)
    assert len(out) == 3
    assert sum(1 for item in out if item.kind == "success") == 2
    assert sum(1 for item in out if item.kind == "failure") == 1
    # Success first, then failure.
    assert [item.kind for item in out] == ["success", "success", "failure"]


def test_retrieve_degrades_when_one_kind_empty(tmp_path, fake_embed):
    lessons = [
        _lesson("good-1", kind="success"),
        _lesson("good-2", kind="success"),
        _lesson("good-3", kind="success"),
    ]
    store = _populated_store(tmp_path, lessons)

    out = retrieve(store, "query", k_success=2, k_failure=2)
    assert len(out) == 2
    assert all(item.kind == "success" for item in out)


def test_retrieve_empty_store_returns_empty(tmp_path, fake_embed):
    store = LessonStore(tmp_path / "lessons.jsonl")
    store.load()
    assert retrieve(store, "query", k_success=3, k_failure=2) == []


def test_format_memory_block_empty_returns_empty_string():
    assert format_memory_block([]) == ""


def test_format_memory_block_includes_both_sections_in_order(tmp_path, fake_embed):
    lessons = [
        _lesson("be specific", kind="success"),
        _lesson("show your work", kind="success"),
        _lesson("avoid filler", kind="failure"),
    ]
    block = format_memory_block(lessons)

    assert "Relevant patterns from past brainstorms:" in block
    assert "Things that worked (apply when relevant):" in block
    assert "Things to avoid (anti-patterns from past failures):" in block
    assert "- be specific" in block
    assert "- show your work" in block
    assert "- avoid filler" in block

    # Success section comes before failure section.
    assert block.index("Things that worked") < block.index("Things to avoid")
    # Trailing whitespace stripped.
    assert block == block.rstrip()


def test_format_memory_block_omits_failure_section_when_no_failures():
    lessons = [
        _lesson("be specific", kind="success"),
        _lesson("show your work", kind="success"),
    ]
    block = format_memory_block(lessons)
    assert "Things that worked" in block
    assert "Things to avoid" not in block
    assert "anti-patterns" not in block


def test_format_memory_block_omits_success_section_when_no_successes():
    lessons = [_lesson("avoid filler", kind="failure")]
    block = format_memory_block(lessons)
    assert "Things to avoid" in block
    assert "Things that worked" not in block


def test_augmented_system_none_base_empty_lessons_returns_none():
    assert augmented_system(None, []) is None


def test_augmented_system_string_base_empty_lessons_returns_base_unchanged():
    assert augmented_system("base system", []) == "base system"


def test_augmented_system_prepends_block_to_base():
    lessons = [_lesson("be specific", kind="success")]
    result = augmented_system("you are a helper", lessons)
    assert result is not None
    block = format_memory_block(lessons)
    assert result.startswith(block)
    assert result.endswith("you are a helper")
    assert "\n\n" in result


def test_augmented_system_none_base_with_lessons_returns_block_only():
    lessons = [_lesson("be specific", kind="success")]
    result = augmented_system(None, lessons)
    assert result == format_memory_block(lessons)
