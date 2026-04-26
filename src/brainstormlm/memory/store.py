"""Numpy-backed lesson store with append-only JSONL persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brainstormlm.clients.openai import embed
from brainstormlm.memory.lessons import Lesson


class LessonStore:
    """In-memory store of `Lesson`s plus their L2-normalized embeddings.

    Persistence is append-only JSONL — `add` and `add_many` never truncate.
    Reset by deleting the file externally before construction.
    """

    def __init__(self, lessons_path: Path) -> None:
        self.lessons_path = lessons_path
        self.lessons: list[Lesson] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._known_ids: set[str] = set()

    def load(self) -> None:
        """Re-read JSONL and re-embed (idempotent — discards prior in-memory state)."""
        self.lessons = []
        self._known_ids = set()
        if self.lessons_path.exists():
            with self.lessons_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lesson = Lesson.model_validate_json(line)
                    self.lessons.append(lesson)
                    self._known_ids.add(lesson.id)

        if not self.lessons:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            return

        matrix, _ = embed([lesson.lesson for lesson in self.lessons])
        self.embeddings = matrix

    def add(self, lesson: Lesson) -> bool:
        """Append a single lesson; returns False if its id is already known."""
        if lesson.id in self._known_ids:
            return False

        self._append_jsonl([lesson])
        matrix, _ = embed([lesson.lesson])
        self._stack_embeddings(matrix)
        self.lessons.append(lesson)
        self._known_ids.add(lesson.id)
        return True

    def add_many(self, lessons: list[Lesson]) -> int:
        """Bulk-append; one batched embed call for the new (non-duplicate) lessons."""
        new_lessons: list[Lesson] = []
        seen_in_batch: set[str] = set()
        for lesson in lessons:
            if lesson.id in self._known_ids or lesson.id in seen_in_batch:
                continue
            new_lessons.append(lesson)
            seen_in_batch.add(lesson.id)

        if not new_lessons:
            return 0

        self._append_jsonl(new_lessons)
        matrix, _ = embed([lesson.lesson for lesson in new_lessons])
        self._stack_embeddings(matrix)
        self.lessons.extend(new_lessons)
        for lesson in new_lessons:
            self._known_ids.add(lesson.id)
        return len(new_lessons)

    def search(
        self,
        query: str,
        k: int,
        *,
        kinds: tuple[str, ...] | None = None,
    ) -> list[tuple[Lesson, float]]:
        """Cosine top-k. Pre-filter by `kinds` so k applies to the filtered subset."""
        if not self.lessons or k <= 0:
            return []

        # Pre-filter so `k` always names the desired number of returned items
        # within the requested kinds (rather than top-k over all kinds then
        # filtered down, which would silently shrink the result set).
        if kinds is not None:
            allowed = set(kinds)
            filtered = [
                (idx, lesson)
                for idx, lesson in enumerate(self.lessons)
                if lesson.kind in allowed
            ]
        else:
            filtered = list(enumerate(self.lessons))

        if not filtered:
            return []

        query_matrix, _ = embed([query])
        query_vec = query_matrix[0]
        candidate_indices = np.array([idx for idx, _ in filtered])
        candidate_matrix = self.embeddings[candidate_indices]
        scores = candidate_matrix @ query_vec  # cosine — embeddings are L2-normalized

        k = min(k, len(filtered))
        # argpartition for top-k, then sort the slice for stable descending order.
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]

        return [(filtered[i][1], float(scores[i])) for i in top_local]

    def _append_jsonl(self, lessons: list[Lesson]) -> None:
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lessons_path.open("a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(lesson.model_dump(), sort_keys=True) + "\n")

    def _stack_embeddings(self, matrix: np.ndarray) -> None:
        if self.embeddings.size == 0:
            self.embeddings = matrix
        else:
            self.embeddings = np.vstack([self.embeddings, matrix])
