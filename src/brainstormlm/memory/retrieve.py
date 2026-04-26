"""Retrieve top-k success+failure lessons and format them into a system block."""

from __future__ import annotations

from brainstormlm.memory.lessons import Lesson
from brainstormlm.memory.store import LessonStore

_HEADER = "Relevant patterns from past brainstorms:"
_SUCCESS_HEADER = "Things that worked (apply when relevant):"
_FAILURE_HEADER = "Things to avoid (anti-patterns from past failures):"


def retrieve(
    store: LessonStore,
    query: str,
    *,
    k_success: int = 3,
    k_failure: int = 2,
) -> list[Lesson]:
    """Top-k success lessons followed by top-k failure lessons. Scores dropped."""
    successes = (
        [lesson for lesson, _ in store.search(query, k_success, kinds=("success",))]
        if k_success > 0
        else []
    )
    failures = (
        [lesson for lesson, _ in store.search(query, k_failure, kinds=("failure",))]
        if k_failure > 0
        else []
    )
    return successes + failures


def format_memory_block(lessons: list[Lesson]) -> str:
    """Render lessons as a labeled prompt block. Empty input → empty string."""
    if not lessons:
        return ""

    successes = [lesson for lesson in lessons if lesson.kind == "success"]
    failures = [lesson for lesson in lessons if lesson.kind == "failure"]

    sections: list[str] = []
    if successes:
        success_lines = "\n".join(f"  - {lesson.lesson}" for lesson in successes)
        sections.append(f"{_SUCCESS_HEADER}\n{success_lines}")
    if failures:
        failure_lines = "\n".join(f"  - {lesson.lesson}" for lesson in failures)
        sections.append(f"{_FAILURE_HEADER}\n{failure_lines}")

    if not sections:
        return ""

    body = "\n\n".join(sections)
    return f"{_HEADER}\n\n{body}".rstrip()


def augmented_system(
    base_system: str | None,
    lessons: list[Lesson],
) -> str | None:
    """Prepend a memory block to `base_system`. Empty lessons → base unchanged."""
    if not lessons:
        return base_system

    block = format_memory_block(lessons)
    if not block:
        return base_system
    if base_system is None:
        return block
    return f"{block}\n\n{base_system}"
