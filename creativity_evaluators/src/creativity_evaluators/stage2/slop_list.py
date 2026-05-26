"""Static slop list — over-represented LLM phrases ("GPT-isms").

Used by ``stage1.lexical`` to compute a slop-overlap rate (higher == more redundant ==
less creative), in the spirit of EQ-Bench Creative Writing v3's slop penalty.

This is a high-confidence seed (~70 phrases). TODO: replace/extend with the full public
EQ-Bench v3 slop list (~500 entries) when available — drop it into a data file and load it
here, keeping the seed below as a fallback. Phrases are matched case-insensitively with
flexible internal whitespace and word boundaries (see ``stage1.lexical.slop_rate``).
"""

from __future__ import annotations

SLOP_PHRASES: list[str] = [
    # discourse hedges / filler
    "it's worth noting that",
    "worth noting",
    "it is important to note",
    "needless to say",
    "at the end of the day",
    "when it comes to",
    "in conclusion",
    "in today's fast-paced world",
    "in the world of",
    "in the realm of",
    "in the heart of",
    # over-used verbs
    "delve into",
    "dive into",
    "embark on a journey",
    "embark on",
    "navigate the complexities",
    "navigate the",
    "shed light on",
    "pave the way",
    "harness the power",
    "unlock the potential",
    "unleash",
    "leverage",
    "foster",
    "elevate",
    "showcase",
    "underscore",
    "underscores",
    "resonate",
    "boasts",
    "stand as a testament",
    "play a crucial role",
    "weaving",
    # over-used nouns / adjectives
    "tapestry",
    "rich tapestry",
    "a testament to",
    "testament to",
    "beacon",
    "treasure trove",
    "landscape",
    "ever-evolving",
    "ever-changing",
    "game-changer",
    "cutting-edge",
    "state-of-the-art",
    "seamless",
    "robust",
    "myriad",
    "a myriad of",
    "plethora",
    "nuanced",
    "multifaceted",
    "pivotal",
    "intricate",
    "vibrant",
    "bustling",
    "whimsical",
    "enigmatic",
    "captivating",
    "breathtaking",
    "meticulous",
    "meticulously",
    "symphony of",
    "kaleidoscope",
    "labyrinth",
    "paradigm shift",
    "holistic",
    "synergy",
    "realm",
    "moreover",
    "furthermore",
    "aforementioned",
]
