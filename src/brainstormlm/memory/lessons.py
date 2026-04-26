"""Lesson schema + LLM-backed extraction."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from brainstormlm.clients import openai as openai_client
from brainstormlm.config import get_settings
from brainstormlm.feedback.trajectories import Trajectory

EXTRACTION_VERSION = 1


def make_lesson_id(lesson: str, kind: str, source_trajectory_id: str) -> str:
    """Deterministic content hash; re-extraction is idempotent on this id."""
    payload = json.dumps([lesson, kind, source_trajectory_id], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: str
    lesson: str
    kind: Literal["success", "failure"]
    domain: str
    source_trajectory_id: str
    extractor_model: str
    extraction_version: int
    extracted_at: str


_EXTRACTOR_SYSTEM = """\
You are a ReasoningBank lesson extractor. Given a single brainstorm trajectory \
(prompt, response, evaluator scores, rationale, failure modes), distill 1-3 \
GENERALIZABLE reasoning patterns that would help a model do better on FUTURE \
prompts in similar domains. You are not summarizing this specific trajectory.

Each lesson must be:
- Transferable: phrased as a reusable heuristic, not bound to this prompt's surface details.
- Concrete: 1-3 sentences, actionable, specific enough to change behavior.
- Labeled by kind:
  - "success" = a reasoning move that worked here and is worth repeating.
  - "failure" = an anti-pattern observed here that should be avoided next time.

Prefer fewer high-quality lessons over padding to three. If only one clear \
pattern is present, return one. Mix kinds when both apply.\
"""

_EXTRACTOR_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lessons"],
    "properties": {
        "lessons": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["lesson", "kind"],
                "properties": {
                    "lesson": {"type": "string"},
                    "kind": {"type": "string", "enum": ["success", "failure"]},
                },
            },
        }
    },
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _format_user_message(t: Trajectory) -> str:
    failure_modes = ", ".join(t.failure_modes) if t.failure_modes else "(none)"
    return (
        f"Prompt:\n{t.prompt}\n\n"
        f"Response:\n{t.response}\n\n"
        f"Evaluator scores: creativity={t.creativity}, logic={t.logic}\n"
        f"Rationale: {t.rationale}\n"
        f"Failure modes: {failure_modes}"
    )


def extract_lessons(t: Trajectory) -> list[Lesson]:
    """LLM-distill 1-3 generalizable reasoning lessons from a scored trajectory."""
    settings = get_settings()
    model = settings.lesson_extractor_model

    user_message = _format_user_message(t)
    payload, _usage = openai_client.complete_json(
        system=_EXTRACTOR_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
        model=model,
        schema=_EXTRACTOR_SCHEMA,
        schema_name="reasoning_bank_lessons",
    )

    extracted_at = _utc_now_iso()
    lessons: list[Lesson] = []
    for item in payload["lessons"]:
        lesson_text = item["lesson"]
        kind = item["kind"]
        lessons.append(
            Lesson(
                id=make_lesson_id(lesson_text, kind, t.id),
                lesson=lesson_text,
                kind=kind,
                domain=t.domain,
                source_trajectory_id=t.id,
                extractor_model=model,
                extraction_version=EXTRACTION_VERSION,
                extracted_at=extracted_at,
            )
        )
    return lessons
