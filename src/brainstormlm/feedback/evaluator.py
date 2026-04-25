"""OpenAI JSON-schema-backed evaluator scoring (prompt, response) pairs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from brainstormlm.clients import openai as openai_client
from brainstormlm.clients.openai import Usage
from brainstormlm.config import get_settings

RUBRIC_VERSION = 1

SYSTEM_PROMPT = """\
You are an expert evaluator of brainstormed responses to open-ended prompts.

Given a (prompt, response) pair, score the response on two dimensions, each
on a 1-5 scale:

  creativity — originality, divergent thinking, surprising-but-relevant ideas.
    1 = generic, predictable.
    3 = a few interesting angles.
    5 = strikingly original, multiple unexpected directions.

  logic — coherence, internal consistency, the response actually addresses
          the prompt and reasons soundly.
    1 = incoherent, off-topic, or self-contradictory.
    3 = mostly on-topic, minor lapses.
    5 = tight, well-reasoned, fully on-topic.

Also produce:
  rationale — 1-3 sentences justifying the scores.
  failure_modes — short tags for weaknesses (e.g. "shallow", "off-topic",
                  "list-without-reasoning"). Empty list if none apply.

Return strict JSON matching the provided schema.
"""

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "creativity": {"type": "integer", "minimum": 1, "maximum": 5},
        "logic": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["creativity", "logic", "rationale", "failure_modes"],
}


class EvaluatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creativity: int = Field(ge=1, le=5)
    logic: int = Field(ge=1, le=5)
    rationale: str
    failure_modes: list[str] = Field(default_factory=list)


def parse(payload: dict) -> EvaluatorResult:
    return EvaluatorResult.model_validate(payload)


def score(
    prompt: str,
    response: str,
    *,
    model: str | None = None,
    use_cache: bool = True,
) -> tuple[EvaluatorResult, Usage]:
    model = model or get_settings().evaluator_model
    user = (
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "Score the response and return JSON."
    )
    payload, usage = openai_client.complete_json(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        model=model,
        schema=_SCHEMA,
        schema_name="evaluator_result",
        max_tokens=400,
        temperature=0.0,
        use_cache=use_cache,
    )
    return parse(payload), usage
