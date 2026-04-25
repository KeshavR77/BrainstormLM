"""OpenAI-backed prompter that synthesizes brainstorming prompts per domain."""

from __future__ import annotations

from brainstormlm.clients import openai as openai_client
from brainstormlm.clients.openai import Usage
from brainstormlm.config import get_settings

DOMAINS: list[str] = [
    "science.physics",
    "science.biology",
    "science.chemistry",
    "science.astronomy",
    "art.visual",
    "art.music",
    "art.writing_fiction",
    "art.writing_poetry",
    "art.film",
    "business.startup",
    "business.marketing",
    "business.ops",
    "product.consumer",
    "product.b2b",
    "product.hardware",
    "personal.productivity",
    "personal.relationships",
    "personal.hobbies",
    "tech.software",
    "tech.ai",
    "tech.hardware",
    "education.teaching",
    "education.learning",
    "everyday.food",
    "everyday.travel",
    "everyday.home",
]

SYSTEM_PROMPT = """\
You generate brainstorming prompts for a creative-reasoning model.

Constraints for every prompt you produce:
  - One prompt only. Output the prompt string and nothing else (no quotes, no
    preamble, no numbering).
  - Specific enough that the model can answer concretely.
  - Open-ended enough that there are many reasonable answers worth exploring.
  - Avoid yes/no questions. Avoid factual lookups.
  - Aim for a single sentence or two; never more than 60 words.
"""


def generate_prompt(
    domain: str,
    *,
    salt: int = 0,
    model: str | None = None,
    use_cache: bool = True,
) -> tuple[str, Usage]:
    model = model or get_settings().prompter_model
    user = (
        f"Domain: {domain}\n"
        f"Variation: {salt}\n"
        "Produce one brainstorming prompt for this domain. Output only the prompt."
    )
    text, usage = openai_client.complete(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        model=model,
        max_tokens=120,
        temperature=1.2,
        use_cache=use_cache,
    )
    return text.strip().strip('"').strip(), usage
