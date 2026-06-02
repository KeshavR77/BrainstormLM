"""Salad detector — flags degenerate "word-salad" generations.

Four independent failure modes (any one → salad), all computed from the raw
string with no embeddings:

  - token_loop:   a token or short n-gram repeats in a tight consecutive loop
                  ("the the the the", "to be or to be or to be").
  - comma_bleed:  comma-per-token ratio > 0.30 — runaway comma-separated lists.
  - micro_output: fewer than 20 whitespace tokens — too short to be a real idea.
  - non_ascii:    more than 5% of characters are non-ASCII — encoding/gibberish.

These are the in-CI canaries from the Diversity-Ideation-Sprint plan: cheap,
deterministic, and meant to catch the salad failure modes that small-model
repulsion/steering can induce. They gate output; they do NOT score quality.
"""

from __future__ import annotations

import re
from typing import Sequence

COMMA_RATIO_THRESHOLD = 0.30
MICRO_TOKEN_MIN = 20
NON_ASCII_THRESHOLD = 0.05
TOKEN_LOOP_MIN_RUN = 4  # this many identical consecutive tokens → loop
PHRASE_LOOP_MIN_REPEATS = 3  # this many consecutive equal n-gram blocks → loop

_WORD_RE = re.compile(r"\S+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _has_token_loop(tokens: Sequence[str]) -> bool:
    """True if a single token runs ≥ TOKEN_LOOP_MIN_RUN times in a row, or a
    2- or 3-gram block repeats ≥ PHRASE_LOOP_MIN_REPEATS times back-to-back."""
    lowered = [t.lower() for t in tokens]

    # (i) consecutive identical single-token run
    run = 1
    for i in range(1, len(lowered)):
        if lowered[i] == lowered[i - 1]:
            run += 1
            if run >= TOKEN_LOOP_MIN_RUN:
                return True
        else:
            run = 1

    # (ii) consecutive equal non-overlapping n-gram blocks
    for n in (2, 3):
        if len(lowered) < n * PHRASE_LOOP_MIN_REPEATS:
            continue
        for start in range(len(lowered) - n + 1):
            block = lowered[start : start + n]
            reps = 1
            j = start + n
            while j + n <= len(lowered) and lowered[j : j + n] == block:
                reps += 1
                j += n
            if reps >= PHRASE_LOOP_MIN_REPEATS:
                return True
    return False


def salad_detector(text: str) -> dict[str, bool]:
    """Return per-mode flags plus ``is_salad`` (the OR of all four modes)."""
    tokens = _tokens(text)
    n_tokens = len(tokens)

    micro_output = n_tokens < MICRO_TOKEN_MIN
    comma_bleed = (text.count(",") / max(n_tokens, 1)) > COMMA_RATIO_THRESHOLD
    non_ascii = bool(text) and (
        sum(1 for c in text if ord(c) > 127) / len(text)
    ) > NON_ASCII_THRESHOLD
    token_loop = _has_token_loop(tokens)

    is_salad = micro_output or comma_bleed or non_ascii or token_loop
    return {
        "token_loop": token_loop,
        "comma_bleed": comma_bleed,
        "micro_output": micro_output,
        "non_ascii": non_ascii,
        "is_salad": is_salad,
    }


def salad_rate(texts: Sequence[str]) -> float:
    """Fraction of ``texts`` flagged as salad. Empty input → 0.0."""
    if not texts:
        return 0.0
    flagged = sum(1 for t in texts if salad_detector(t)["is_salad"])
    return flagged / len(texts)
