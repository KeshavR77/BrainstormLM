"""Planner → K orthogonal angles (N2).

Given a brief, ask an LLM for K distinct 'angles/approaches', then deduplicate
them by embedding proximity so the diverger only expands genuinely different
starting points.

Reference: DPWriter (arXiv 2601.09609), G2 (arXiv 2511.00432),
           sprint note Family 3 / shortlist N2.

Usage::

    from brainstormlm.generation.planner_angles import plan_angles
    from brainstormlm.clients.openai import embed

    angles = plan_angles(
        brief="Design a logo for a fintech startup",
        k=6,
        llm=my_llm,
        embedder=embed,
    )
    # → ["typographic wordmark", "abstract geometric", "heritage badge", ...]

Inject ``llm`` and ``embedder`` for tests (no API calls):

    angles = plan_angles(
        brief="...",
        k=5,
        llm=fake_llm,
        embedder=fake_embedder,
    )
"""

from __future__ import annotations

import re
from typing import Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class LLMCallable(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a creative strategist. For the brief below, generate exactly {k} "
    "DISTINCT, ORTHOGONAL creative angles or approaches (not full ideas). "
    "Each angle should represent a fundamentally different way of thinking about "
    "the problem — different conceptual starting points, not variations of the same idea.\n\n"
    "Format as a numbered list, one angle per line:\n"
    "1. <angle>\n"
    "2. <angle>\n"
    "…\n\n"
    "Return ONLY the numbered list. Be concise — each angle in ≤15 words."
)


def _build_prompt(brief: str, k: int) -> str:
    return f"{_PLANNER_SYSTEM.format(k=k)}\n\nBrief: {brief}"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_LIST_PREFIX = re.compile(r"^\s*\d+[\.\)]\s*")


def parse_angles(response: str) -> list[str]:
    """Parse a numbered-list LLM response into a list of angle strings.

    Strips numbering, markdown bold, and leading/trailing whitespace.
    Skips empty lines and non-list lines.
    """
    angles: list[str] = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        if not re.match(r"^\s*\d+[\.\)]", line):
            continue
        text = _LIST_PREFIX.sub("", line)
        # Strip markdown bold **…**
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = text.strip().strip(".,;:")
        if text:
            angles.append(text)
    return angles


# ---------------------------------------------------------------------------
# Deduplication by embedding proximity
# ---------------------------------------------------------------------------


def dedupe_angles(
    angles: list[str],
    *,
    embedder: EmbedCallable,
    threshold: float = 0.95,
) -> list[str]:
    """Remove near-duplicate angles using cosine similarity.

    Greedy first-match: iterate through angles in order; keep an angle only if
    its cosine similarity to every already-kept angle is below ``threshold``.
    First-encountered wins — order is preserved for surviving angles.

    Args:
        angles: list of angle strings (from parse_angles or raw).
        embedder: callable ``(texts: list[str]) -> np.ndarray`` of shape (N, D),
                  L2-normalised. Use a fake in tests; ``openai.embed`` in prod.
        threshold: cosine-similarity cutoff above which an angle is considered
                   a near-duplicate. Default 0.95 matches sprint spec.

    Returns:
        Deduplicated list (subset of ``angles``, original order preserved).
    """
    if not angles:
        return []

    embeddings = embedder(angles)  # (N, D) L2-normalised

    kept_indices: list[int] = []
    for i in range(len(angles)):
        if not kept_indices:
            kept_indices.append(i)
            continue
        kept_embs = embeddings[kept_indices]  # (K, D)
        sims = kept_embs @ embeddings[i]     # (K,) cosine sims
        if sims.max() < threshold:
            kept_indices.append(i)

    return [angles[i] for i in kept_indices]


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def plan_angles(
    brief: str,
    *,
    k: int = 6,
    llm: LLMCallable,
    embedder: EmbedCallable,
    dedupe_threshold: float = 0.95,
    **llm_kwargs,
) -> list[str]:
    """Ask an LLM for K orthogonal angles and deduplicate by embedding proximity.

    Args:
        brief: the user's brainstorming prompt.
        k: number of angles to request from the LLM.
        llm: callable ``(prompt: str, **kwargs) -> str``.
        embedder: callable ``(texts: list[str]) -> np.ndarray`` (L2-normalised).
        dedupe_threshold: cosine-sim cutoff for deduplication (default 0.95).
        **llm_kwargs: forwarded to ``llm``.

    Returns:
        Deduplicated list of angle strings (≤ k items).
    """
    prompt = _build_prompt(brief, k)
    response = llm(prompt, **llm_kwargs)
    angles = parse_angles(response)
    if not angles:
        return angles
    return dedupe_angles(angles, embedder=embedder, threshold=dedupe_threshold)


__all__ = [
    "parse_angles",
    "dedupe_angles",
    "plan_angles",
]
