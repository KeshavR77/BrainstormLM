"""Verbalized Sampling (N1) — arXiv 2510.01171.

One LLM call returns N candidate ideas with self-estimated probabilities.
We parse robustly then sample from the tail (down-weighting the modal idea).

Usage::

    items = verbalized_sample("brainstorm business models for a coffee shop", llm=my_llm, n=5)
    chosen = tail_sample(items)          # returns one VSItem

Inject ``llm`` for tests (no API calls needed):

    def fake_llm(prompt: str, **kwargs) -> str:
        return "1. Idea A (0.6)\\n2. Idea B (0.3)\\n3. Idea C (0.1)"

    items = verbalized_sample("...", llm=fake_llm, n=3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VSItem:
    """One candidate idea with its self-estimated probability."""

    text: str
    prob: float  # normalised to [0, 1]; sums to 1.0 across a parsed response


# ---------------------------------------------------------------------------
# LLM protocol (for type-checking; tests inject a plain callable)
# ---------------------------------------------------------------------------


class LLMCallable(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_VS_SYSTEM = (
    "You are a creative brainstorming assistant. "
    "For the given brief, generate exactly {n} distinct candidate ideas. "
    "For EACH idea, assign a probability (0–1) reflecting how likely a typical model "
    "would generate that idea — common/expected ideas get higher probabilities, "
    "unusual/creative ones get lower. Probabilities must sum to 1.0.\n\n"
    "Format STRICTLY as a numbered list, one idea per line:\n"
    "1. <idea text> (<probability as decimal or %>)\n"
    "2. …\n"
    "Return ONLY the numbered list, nothing else."
)


def _build_prompt(brief: str, n: int) -> str:
    system = _VS_SYSTEM.format(n=n)
    return f"{system}\n\nBrief: {brief}"


# ---------------------------------------------------------------------------
# Robust parser
# ---------------------------------------------------------------------------

# Patterns to extract a probability value from various formats:
#   (0.40) | (40%) | — 40% | prob ≈ 40% | : 40% | [0.40] | probability: 40%
_PROB_PATTERNS = [
    re.compile(r"\((\d+(?:\.\d+)?)\s*%\s*\)"),         # (40%)
    re.compile(r"\[(\d+(?:\.\d+)?)\s*%?\s*\]"),        # [0.40] or [40%]
    re.compile(r"(?:prob(?:ability)?[:\s≈~]+)(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE),  # prob: 40%
    re.compile(r"[-—]\s*(\d+(?:\.\d+)?)\s*%"),         # — 40%
    re.compile(r":\s*(\d+(?:\.\d+)?)\s*%"),             # : 40%
    re.compile(r"\((\d+(?:\.\d+)?)\)"),                 # (0.40)
    re.compile(r"(\d+(?:\.\d+)?)\s*%"),                 # trailing 40%
    re.compile(r"(?:^|\s)0\.(\d+)"),                    # bare 0.XX
]

# Strip a leading numbered list marker: "1. ", "1) ", etc.
_LIST_PREFIX = re.compile(r"^\s*\d+[\.\)]\s*")
# Strip markdown bold **…**
_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def _extract_prob(raw: str) -> float | None:
    """Try each pattern in order; return the first numeric probability found."""
    for pat in _PROB_PATTERNS:
        m = pat.search(raw)
        if m:
            val = float(m.group(1))
            # Values clearly > 1 are percentages
            return val / 100.0 if val > 1.0 else val
    return None


def _clean_text(raw: str) -> str:
    """Remove the probability annotation and list prefix from the idea text."""
    text = _LIST_PREFIX.sub("", raw)
    text = _BOLD.sub(r"\1", text)
    # Remove the probability annotation — try each pattern
    for pat in _PROB_PATTERNS:
        text = pat.sub("", text)
    # Clean up trailing punctuation / whitespace left by the removal
    text = re.sub(r"[\s\-—:]+$", "", text.strip())
    return text.strip()


def parse_vs_response(response: str) -> list[VSItem]:
    """Parse a verbalized-sampling LLM response into a list of VSItems.

    Handles: decimal probs (0.40), percentages (40%), 'prob ≈ 25%',
    [0.20], '— 10%', '**bold text**', unnormalized totals.

    Returns at least one item (falls back to equal-weight if parsing fails).
    """
    items: list[VSItem] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        # Must look like a numbered list item (starts with "N." or "N)")
        if not re.match(r"^\s*\d+[\.\)]", line):
            continue
        prob = _extract_prob(line)
        if prob is None:
            continue
        text = _clean_text(line)
        if not text:
            continue
        items.append(VSItem(text=text, prob=prob))

    if not items:
        # Last-resort: equal-weight parse of any numbered lines
        for line in response.splitlines():
            if re.match(r"^\s*\d+[\.\)]", line.strip()):
                text = _LIST_PREFIX.sub("", line.strip()).strip()
                if text:
                    items.append(VSItem(text=text, prob=1.0))

    if not items:
        return items

    # Renormalise so probabilities sum to 1.0
    total = sum(i.prob for i in items)
    if total > 0:
        items = [VSItem(text=i.text, prob=i.prob / total) for i in items]
    return items


# ---------------------------------------------------------------------------
# Tail sampling
# ---------------------------------------------------------------------------


def tail_sample(
    items: list[VSItem],
    *,
    rng: np.random.Generator | None = None,
    modal_floor: float = 0.0,
) -> VSItem:
    """Sample one VSItem, down-weighting the modal (highest-probability) candidate.

    The modal item's effective weight is set to ``modal_floor`` (default 0.0),
    which eliminates it entirely. Remaining weights are renormalised before
    sampling.

    Args:
        items: parsed VS output (probabilities sum to 1).
        rng: optional numpy random generator (for reproducibility in tests).
        modal_floor: weight floor for the modal idea. 0.0 = never sample it.

    Returns:
        One VSItem drawn from the modified distribution.
    """
    if not items:
        raise ValueError("items must be non-empty")
    if rng is None:
        rng = np.random.default_rng()

    max_prob = max(i.prob for i in items)
    weights = np.array([
        modal_floor if i.prob == max_prob else i.prob
        for i in items
    ], dtype=np.float64)

    total = weights.sum()
    if total == 0.0:
        # All weights zeroed (e.g. all items tied for modal); fall back to uniform
        weights = np.ones(len(items), dtype=np.float64)
        total = weights.sum()

    weights /= total
    idx = int(rng.choice(len(items), p=weights))
    return items[idx]


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def verbalized_sample(
    brief: str,
    *,
    llm: LLMCallable,
    n: int = 5,
    **llm_kwargs,
) -> list[VSItem]:
    """Issue one LLM call requesting N ideas with self-estimated probabilities.

    Args:
        brief: the user's brainstorming prompt.
        llm: callable ``(prompt: str, **kwargs) -> str``.
        n: number of candidates to request.
        **llm_kwargs: forwarded to ``llm``.

    Returns:
        Parsed list of VSItems (possibly fewer than n if the LLM omits some).
    """
    prompt = _build_prompt(brief, n)
    response = llm(prompt, **llm_kwargs)
    return parse_vs_response(response)


__all__ = [
    "VSItem",
    "parse_vs_response",
    "tail_sample",
    "verbalized_sample",
]
