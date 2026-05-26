"""Prompt sets for the creative vs neutral contrast.

We render each shared *topic* under two system framings — a creativity-invoking one and a
neutral one — so the only thing that differs between the positive and negative pools is
the creativity instruction (topic is held fixed). This matched-contrast design mirrors the
conformity paper, and the framings reuse the repo's existing CREATIVE/GENERIC wording so
behaviour is comparable to the current Conceptial pipeline.

The default set has 48 topics; with several stochastic samples per (topic, framing) this
comfortably clears the ``min_per_class`` (=100) verified-subset target after filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CREATIVE_SYSTEM = (
    "You are a wildly inventive brainstorming partner. Embrace surprising ideas, "
    "unconventional analogies, and creative leaps. Be original. Quality over consensus."
)
NEUTRAL_SYSTEM = (
    "You are a careful, conventional assistant. Give the most standard, expected, "
    "straightforward answer. Avoid speculation or unusual ideas."
)

TOPICS: tuple[str, ...] = (
    "Name unusual uses for a paperclip.",
    "Pitch a new consumer coffee brand.",
    "Design a logo direction for a sustainable-packaging startup.",
    "Suggest names for a new public park.",
    "Invent a board game about city planning.",
    "Describe a museum exhibit about sound.",
    "Propose a slogan for a bicycle co-op.",
    "Imagine a festival for the winter solstice.",
    "Reinvent the umbrella.",
    "Design a playground for an elderly community.",
    "Name three uses for an empty shipping container.",
    "Pitch a documentary about tap water.",
    "Concept a campaign for a public library.",
    "Invent a snack for long train journeys.",
    "Describe a garden that works at night.",
    "Propose a new sport played in zero gravity.",
    "Reimagine the morning alarm clock.",
    "Design packaging for fresh herbs.",
    "Suggest a theme for a small-town diner.",
    "Invent a toy that teaches patience.",
    "Concept an app that makes commuting fun.",
    "Pitch a perfume inspired by rain.",
    "Name a constellation that does not exist yet.",
    "Design a chair for reading.",
    "Propose a mural for a hospital hallway.",
    "Invent a holiday for neighbours.",
    "Describe a restaurant with no menu.",
    "Reinvent the to-do list.",
    "Pitch a podcast about everyday objects.",
    "Design a fountain for a desert town.",
    "Suggest uses for leftover bread.",
    "Concept a museum gift-shop product.",
    "Invent a musical instrument for children.",
    "Describe a treehouse for adults.",
    "Propose a name for a comet.",
    "Design a bus stop people enjoy waiting at.",
    "Pitch a tea blend for late-night studying.",
    "Reimagine the welcome mat.",
    "Invent a game played with shadows.",
    "Concept a brand of recycled notebooks.",
    "Describe a lighthouse repurposed for tourists.",
    "Propose a flavour of ice cream for autumn.",
    "Design a coat for cyclists.",
    "Suggest a ritual for finishing a big project.",
    "Invent a vending machine for a school.",
    "Pitch a short film set entirely in an elevator.",
    "Reinvent the greeting card.",
    "Concept a community seed library.",
)


@dataclass
class PromptSet:
    creative_system: str
    neutral_system: str
    topics: tuple[str, ...]

    def items(self) -> list[tuple[str, str, str]]:
        """Yield ``(framing, system, topic)`` for both framings over every topic.

        ``framing`` is ``"creative"`` or ``"neutral"`` — the label that decides which
        verified pool a passing response joins.
        """
        out: list[tuple[str, str, str]] = []
        for topic in self.topics:
            out.append(("creative", self.creative_system, topic))
            out.append(("neutral", self.neutral_system, topic))
        return out


def default_prompt_set() -> PromptSet:
    return PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS)


def load_prompt_set(path: str | Path) -> PromptSet:
    """Load a prompt set from JSON: ``{creative_system, neutral_system, topics: [...]}``."""
    raw = json.loads(Path(path).read_text())
    return PromptSet(
        creative_system=raw["creative_system"],
        neutral_system=raw["neutral_system"],
        topics=tuple(raw["topics"]),
    )
