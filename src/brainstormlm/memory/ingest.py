"""Fold human-feedback trajectories into the ReasoningBank LessonStore.

Called from the FastAPI server's ``/feedback`` handler in a background task.
Only thumbs-up rows are ingested — down-votes lack the failure-mode signal
needed to extract a clean "avoid this" lesson.

Failures are swallowed: this runs out-of-band and must never crash the
caller (the user's HTTP response has already been returned).
"""

from __future__ import annotations

import logging
from pathlib import Path

from brainstormlm.config import get_settings
from brainstormlm.feedback.trajectories import Trajectory
from brainstormlm.memory.lessons import extract_lessons
from brainstormlm.memory.store import LessonStore

_log = logging.getLogger(__name__)


def _synthesize_evaluator_fields(t: Trajectory) -> Trajectory:
    """Map the human rating onto the placeholder evaluator fields.

    The lesson extractor reads ``creativity``, ``logic``, and ``rationale``
    to decide what counts as a success worth distilling. Human-flow
    trajectories ship with placeholders (1, 1, "(awaiting human feedback)"),
    so we replace them before extraction.
    """
    rating = t.human_rating
    assert rating is not None  # caller guards
    if rating.rating is not None:
        score = max(1, min(5, int(rating.rating)))
    else:
        score = 4 if rating.thumbs == "up" else 2

    if rating.critique:
        rationale = f"Human thumbs-{rating.thumbs}: {rating.critique}"
    else:
        rationale = f"Human thumbs-{rating.thumbs} (no critique provided)."

    return t.model_copy(update={"creativity": score, "logic": score, "rationale": rationale})


def ingest_human_feedback(
    trajectory: Trajectory,
    *,
    lessons_path: Path | None = None,
) -> int:
    """Extract + add lessons from a thumbs-up trajectory. Returns lesson count added.

    Returns 0 (and logs) on any failure so the background task is safe to
    fire-and-forget. Skips silently for thumbs-down or unrated rows.
    """
    if trajectory.human_rating is None or trajectory.human_rating.thumbs != "up":
        return 0

    path = lessons_path if lessons_path is not None else get_settings().lessons_path
    synthesized = _synthesize_evaluator_fields(trajectory)

    try:
        lessons = extract_lessons(synthesized)
    except Exception as e:
        _log.warning("lesson extraction failed for trajectory_id=%s: %s", trajectory.id, e)
        return 0

    if not lessons:
        return 0

    try:
        store = LessonStore(path)
        store.load()
        added = store.add_many(lessons)
    except Exception as e:
        _log.warning("lesson store add failed for trajectory_id=%s: %s", trajectory.id, e)
        return 0

    return added
