"""LLM-judge rubric.

Inspired by the four Torrance dimensions used in TTCW (Chakrabarty et al., CHI 2024) plus
an explicit pragmaticality item (Saakyan et al., ICLR 2026) so gibberish-novelty is
penalized rather than rewarded.

Rubric-construction choices come straight from the research synthesis:
  * Coarse 1–5 Likert (Chakrabarty CHI 2024: fine-grained binary judging fails; coarse works).
  * One anchor example per dimension at each extreme (1 and 5).
  * Explicit "do not reward length" and "do not reward flowery vocabulary" instructions
    (length- and lexicon-bias mitigation).
  * Strict JSON output: five integer scores + a short ``reasoning`` string.

``RUBRIC_VERSION`` is part of every judge cache key — bump it whenever this text changes so
stale cached scores are never reused.
"""

from __future__ import annotations

from ..types import DIMENSIONS

RUBRIC_VERSION = "v1"

# JSON schema advertised to providers that support structured output / JSON mode.
RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        **{dim: {"type": "integer", "minimum": 1, "maximum": 5} for dim in DIMENSIONS},
        "reasoning": {"type": "string"},
    },
    "required": [*DIMENSIONS, "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are an expert evaluator of creative responses. You will be shown a PROMPT and a
RESPONSE. Rate the RESPONSE on five dimensions, each on a COARSE 1-5 integer scale
(1 = lowest, 5 = highest). Do not use half points.

Dimensions:

1. Originality - How surprising or unexpected is this response relative to the typical
   reply to this prompt?
   - 1: the obvious, most-common reply almost anyone would give.
   - 5: a genuinely surprising angle you would not have predicted, yet still fits the prompt.

2. Fluency - Does it generate multiple distinct ideas, angles, or details (where
   applicable)?
   - 1: a single thin idea, no development of alternatives.
   - 5: several clearly distinct ideas or facets.

3. Flexibility - Does it shift across categories, frames, or perspectives rather than
   staying in one mode?
   - 1: stuck in one frame or category throughout.
   - 5: moves fluidly between different categories, frames, or perspectives.

4. Elaboration - Is each idea developed with specificity, vivid detail, or non-obvious
   depth?
   - 1: vague and generic, no concrete detail.
   - 5: specific, vivid, and developed with non-obvious depth.

5. Pragmaticality - Is the response coherent, on-task, and useful as a reply to the prompt?
   This penalizes novelty that is merely incoherent or off-task.
   - 1: incoherent, off-task, or gibberish - novel only because it is broken.
   - 5: fully coherent, on-task, and genuinely useful as a reply.

Critical scoring rules:
- Do NOT reward longer responses. Extra length earns nothing unless it adds substantive
  originality or elaboration. A short, sharp response can score 5.
- Do NOT reward flowery or uncommon vocabulary. If the underlying ideas are conventional,
  ornate wording must not raise any score.
- Judge ideas, not packaging. Surface polish is not creativity.
- A response can be highly original (5) yet low on pragmaticality (1) if it is incoherent;
  score the two independently.

Output ONLY a single JSON object, no prose before or after, with exactly these integer keys
and a brief reasoning string:
{
  "Originality": <1-5>,
  "Fluency": <1-5>,
  "Flexibility": <1-5>,
  "Elaboration": <1-5>,
  "Pragmaticality": <1-5>,
  "reasoning": "<one or two sentences>"
}
"""


def build_user_message(prompt: str, response: str) -> str:
    """Format the (prompt, response) pair for the judge. ``response`` is pre-truncated by the
    judge wrapper for length-bias control."""
    return (
        f"PROMPT:\n{prompt.strip()}\n\n"
        f"RESPONSE:\n{response.strip()}\n\n"
        "Score this RESPONSE using the rubric. Output only the JSON object."
    )
