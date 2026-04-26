"""LessonStore unit tests (no network — embed() is monkeypatched)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from brainstormlm.clients.openai import Usage
from brainstormlm.memory import store as store_module
from brainstormlm.memory.lessons import EXTRACTION_VERSION, Lesson, make_lesson_id
from brainstormlm.memory.store import LessonStore


def _deterministic_embed(text: str, dim: int = 8) -> np.ndarray:
    """Hash-seeded unit vector. Stable across calls — required for cosine tests."""
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
    calls: list[list[str]] = []

    def fake(texts: list[str], *, model: str | None = None, use_cache: bool = True):
        calls.append(list(texts))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), Usage(0, 0, cached=True)
        matrix = np.stack([_deterministic_embed(t) for t in texts], axis=0)
        return matrix, Usage(prompt_tokens=0, completion_tokens=0, cached=False)

    monkeypatch.setattr(store_module, "embed", fake)
    return calls


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


def test_add_five_then_reload_restores_lessons_and_embeddings(tmp_path, fake_embed):
    path = tmp_path / "memory" / "lessons.jsonl"
    store = LessonStore(path)
    store.load()

    lessons = [_lesson(f"lesson-{i}") for i in range(5)]
    for lesson in lessons:
        assert store.add(lesson) is True

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line, lesson in zip(lines, lessons, strict=True):
        parsed = json.loads(line)
        assert parsed["id"] == lesson.id
        # Match trajectories.append's sort_keys=True formatting.
        assert line == json.dumps(lesson.model_dump(), sort_keys=True)

    reloaded = LessonStore(path)
    reloaded.load()
    assert [item.id for item in reloaded.lessons] == [item.id for item in lessons]
    assert reloaded.embeddings.shape == (5, 8)


def test_add_duplicate_id_is_noop(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    lesson = _lesson("only-one")

    assert store.add(lesson) is True
    embed_calls_after_first = len(fake_embed)
    line_count_after_first = len(path.read_text(encoding="utf-8").splitlines())

    assert store.add(lesson) is False
    assert len(fake_embed) == embed_calls_after_first  # no new embed call
    assert (
        len(path.read_text(encoding="utf-8").splitlines()) == line_count_after_first
    )
    assert len(store.lessons) == 1
    assert store.embeddings.shape == (1, 8)


def test_add_many_dedups_and_batches(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()

    seed_lessons = [_lesson("alpha"), _lesson("beta")]
    store.add_many(seed_lessons)
    embed_calls_before = len(fake_embed)
    lines_before = len(path.read_text(encoding="utf-8").splitlines())

    incoming = [
        _lesson("alpha"),  # dup
        _lesson("gamma"),  # new
        _lesson("delta"),  # new
        _lesson("beta"),  # dup
        _lesson("epsilon"),  # new
    ]
    added = store.add_many(incoming)
    assert added == 3
    # Exactly one batched embed() call for the 3 new lessons.
    assert len(fake_embed) == embed_calls_before + 1
    assert fake_embed[-1] == ["gamma", "delta", "epsilon"]

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == lines_before + 3
    assert store.embeddings.shape == (5, 8)
    assert [item.lesson for item in store.lessons] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]


def test_search_returns_top_k_by_cosine(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()

    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    store.add_many([_lesson(t) for t in texts])

    query = "alpha"
    q_vec = _deterministic_embed(query)
    scores = {t: float(np.dot(_deterministic_embed(t), q_vec)) for t in texts}
    expected_top2 = sorted(scores, key=scores.get, reverse=True)[:2]

    results = store.search(query, k=2)
    assert [lesson.lesson for lesson, _ in results] == expected_top2
    # Descending score order.
    assert results[0][1] >= results[1][1]


def test_search_filters_by_kind(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    store.add_many(
        [
            _lesson("good-1", kind="success"),
            _lesson("good-2", kind="success"),
            _lesson("bad-1", kind="failure"),
            _lesson("bad-2", kind="failure"),
            _lesson("bad-3", kind="failure"),
        ]
    )
    results = store.search("anything", k=2, kinds=("failure",))
    assert len(results) == 2
    assert all(lesson.kind == "failure" for lesson, _ in results)


def test_search_k_larger_than_store_returns_all(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    store.add_many([_lesson(t) for t in ("a", "b", "c")])

    results = store.search("query", k=10)
    assert len(results) == 3


def test_search_empty_store_returns_empty_list(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    assert store.search("anything", k=5) == []


def test_search_no_matching_kind_returns_empty(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    store.add_many(
        [_lesson("good-1", kind="success"), _lesson("good-2", kind="success")]
    )
    assert store.search("query", k=2, kinds=("failure",)) == []


def test_load_is_idempotent(tmp_path, fake_embed):
    path = tmp_path / "lessons.jsonl"
    store = LessonStore(path)
    store.load()
    store.add_many([_lesson("a"), _lesson("b")])
    store.load()
    assert len(store.lessons) == 2
    assert store.embeddings.shape == (2, 8)


def test_load_missing_file_yields_empty_state(tmp_path, fake_embed):
    store = LessonStore(tmp_path / "does-not-exist.jsonl")
    store.load()
    assert store.lessons == []
    assert store.embeddings.shape == (0, 0)
