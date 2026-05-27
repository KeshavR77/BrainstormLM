"""Prompt sets for the creative vs neutral contrast.

We render each shared *topic* under two system framings — a creativity-invoking one and a
neutral one — so the only thing that differs between the positive and negative pools is
the creativity instruction (topic is held fixed). This matched-contrast design mirrors the
conformity paper, and the framings reuse the repo's existing CREATIVE/GENERIC wording so
behaviour is comparable to the current Conceptial pipeline.

The default set has 150 topics. With one sample per (topic, framing) the distinct-negative
count equals the topic count, so size the topic set to the verified-subset ``min_per_class``
target. (Greedy neutral makes extra samples duplicates; a low-temperature sampled neutral
gives distinct draws but one-per-topic is the cleanest way to avoid near-duplicates.)
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
    "Invent a breakfast for astronauts.",
    "Pitch a walking tour of an ordinary street.",
    "Design a clock for someone who cannot see.",
    "Reimagine the office water cooler.",
    "Name a new colour and describe it.",
    "Concept a subscription box for curiosity.",
    "Invent a bedtime story for robots.",
    "Describe a café that opens only at dawn.",
    "Propose a public artwork powered by wind.",
    "Design a backpack for city explorers.",
    "Suggest uses for a retired subway car.",
    "Pitch a board game about the weather.",
    "Reinvent the house key.",
    "Invent a sport for rainy days.",
    "Concept a library that lends tools.",
    "Describe a hotel built inside a cliff.",
    "Propose a soda flavour for summer nights.",
    "Design a uniform for future park rangers.",
    "Suggest a name for a newly discovered bird.",
    "Invent a machine that organises memories.",
    "Pitch a museum dedicated to silence.",
    "Reimagine the neighbourhood mailbox.",
    "Invent a game for a long car ride.",
    "Design a fruit bowl that reduces waste.",
    "Pitch a radio show for night-shift workers.",
    "Concept a park bench that tells stories.",
    "Reimagine the garden hose.",
    "Invent a sandwich named after an emotion.",
    "Pitch a TV show set in a laundromat.",
    "Design a wheelchair for off-road adventures.",
    "Name a cocktail inspired by a thunderstorm.",
    "Concept a museum you visit in total darkness.",
    "Reimagine the office stapler.",
    "Suggest a sport that combines swimming and chess.",
    "Invent a board game about running a bakery.",
    "Describe a city park designed for dogs.",
    "Propose a public sculpture that changes with the seasons.",
    "Design a lunchbox that keeps food warm for eight hours.",
    "Pitch a magazine for people who collect maps.",
    "Name a public holiday celebrating failure.",
    "Concept an app that helps strangers cook together.",
    "Invent a musical instrument played with the feet.",
    "Describe a bookstore that sells one book at a time.",
    "Reinvent the bicycle bell.",
    "Propose a theme for a rooftop garden.",
    "Design a tent that sets itself up.",
    "Suggest uses for a thousand mismatched socks.",
    "Pitch a podcast hosted by two rival chefs.",
    "Name a perfume that smells like a library.",
    "Concept a vending machine for poetry.",
    "Invent a game played entirely in the dark.",
    "Describe a hotel room designed for stargazing.",
    "Propose a flavour of tea for a rainy afternoon.",
    "Design a backpack for a travelling musician.",
    "Suggest a ritual for moving into a new home.",
    "Reimagine the traffic light.",
    "Pitch a documentary about abandoned shopping malls.",
    "Invent a snack that changes colour as you eat it.",
    "Name a startup that rents out houseplants.",
    "Concept a festival celebrating the ocean at night.",
    "Design a playground inspired by the deep sea.",
    "Describe a restaurant that serves only breakfast at midnight.",
    "Propose a mural for an underground subway station.",
    "Reinvent the shopping cart.",
    "Name a comet that appears once a century.",
    "Invent a toy that teaches kids about money.",
    "Pitch a board game about exploring caves.",
    "Design a coat that doubles as a sleeping bag.",
    "Concept a library that lends musical instruments.",
    "Name a brand of ethically-made sneakers.",
    "Describe a treehouse village for remote workers.",
    "Propose a public artwork made entirely of light.",
    "Invent a kitchen gadget that reduces food waste.",
    "Reimagine the park bench for winter.",
    "Suggest a theme for a children's science museum.",
    "Pitch a short film told entirely through sounds.",
    "Design a water bottle that encourages you to drink more.",
    "Name a tea house aimed at insomniacs.",
    "Concept a board game about building a city underwater.",
    "Invent a holiday tradition for people who live alone.",
    "Describe a garden that attracts fireflies.",
    "Propose a slogan for a campaign to plant more trees.",
    "Reinvent the alarm for heavy sleepers.",
    "Suggest uses for an old phone booth.",
    "Pitch a magazine about everyday acts of kindness.",
    "Design a desk for someone who works standing up.",
    "Name a constellation that tells a love story.",
    "Concept a museum dedicated to forgotten inventions.",
    "Invent a board game about running a small farm.",
    "Describe a café designed for writing a novel.",
    "Propose a public fountain that plays music.",
    "Reimagine the umbrella for two people.",
    "Name a new flavour of chewing gum.",
    "Pitch a podcast about the history of board games.",
    "Design a chair that encourages good posture.",
    "Concept a festival of homemade inventions.",
    "Invent a snack inspired by a foreign city.",
    "Describe a hotel built around a giant tree.",
    "Propose a logo for a community bicycle-repair shop.",
    "Reinvent the school backpack.",
    "Suggest a ritual for finishing a long journey.",
    "Name a brand of notebooks for travellers.",
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
