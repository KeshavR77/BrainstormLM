"""Conceptial demo streaming source — project-scoped, anti-mode aware.

This module powers the live demo without requiring a model load. It pulls
high-quality pre-written ideas from ``demo_packs.py``, filters them against
the IdeaMemoryStore for the active project (so regenerations never repeat),
and streams them out at realistic SSE pacing.

The design intent is that the streaming endpoint looks indistinguishable from
a real-time Conceptial 14B run: 3-7 second per-idea latency, ideas arriving
one at a time, the exclusion list growing visibly, the memory chip ticking up.
Beneath that, everything is deterministic and cheap.

The IdeaMemoryStore is the single source of truth for what's been shown.
Persistent across sessions. Survives page refreshes. Resets only on explicit
``delete_project`` call.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from brainstormlm.conceptial.demo_packs import (
    CONCEPTIAL_PACKS,
    DemoIdea,
    all_demo_ideas,
)
from brainstormlm.conceptial.idea_store import (
    SOURCE_GENERATED,
    IdeaMemoryStore,
)


# Keyword → pack mapping. Order matters (first match wins).
_PACK_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "brand-identity",
        (
            "identity",
            "logo",
            "logomark",
            "wordmark",
            "brand",
            "typography",
            "type system",
            "visual system",
            "palette",
        ),
    ),
    (
        "campaign-craft",
        (
            "campaign",
            "launch",
            "drop",
            "press",
            "editorial",
            "concept directions",
            "marketing",
            "promotional",
        ),
    ),
    (
        "object-design",
        (
            "packaging",
            "package",
            "carton",
            "bottle",
            "object",
            "physical",
            "merch",
            "product design",
            "container",
        ),
    ),
]


def _pack_for_prompt(prompt: str) -> str | None:
    """Pick the most likely pack for a brief based on keyword matching.

    Returns ``None`` if no keyword matches — falls back to full-catalogue
    rotation in that case.
    """
    lowered = prompt.lower()
    for pack_name, kws in _PACK_KEYWORDS:
        if any(kw in lowered for kw in kws):
            return pack_name
    return None


@dataclass(frozen=True)
class StreamingIdea:
    """One idea as it flows out of the streaming source."""

    idea_id: str
    summary: str
    content: str
    pack: str | None
    similarity_score: float  # 0.0 = brand new direction, 1.0 = redundant with memory
    excluded_summaries: tuple[str, ...]  # snapshot of the exclusion list at time of emit


def _hash_seed(prompt: str, project_id: str) -> int:
    """Deterministic seed so the same (prompt, project) pair starts in the same pack."""
    h = hashlib.sha256(f"{project_id}::{prompt}".encode()).hexdigest()
    return int(h[:8], 16)


def _pacing_seconds(n_total: int, *, base: float = 2.5, jitter: float = 0.8) -> list[float]:
    """Pacing schedule for SSE emits — realistic 'model is generating' cadence.

    Returns N pauses (one before each emit), with a faster first one (~1 sec)
    because that's when the user's attention is highest and they want feedback fast.
    """
    out = [1.0]  # first idea arrives quickly
    for k in range(1, n_total):
        # add modest jitter so the cadence feels alive, not metronomic
        jitter_signed = ((k * 0.61803) % 1.0 - 0.5) * jitter
        out.append(max(0.6, base + jitter_signed))
    return out


class ConceptialStreamingSource:
    """Project-scoped streaming source for the demo flow.

    Usage::

        src = ConceptialStreamingSource(store=memory, project_id="brand-1")
        for idea in src.stream(prompt="campaign concepts for...", n=5):
            yield_to_sse(idea)

    The store is mutated as ideas are emitted — each emit persists the idea
    so the next call's exclusion list includes it.
    """

    def __init__(
        self,
        *,
        store: IdeaMemoryStore,
        project_id: str,
        hidden_dim: int = 64,
        pacing: bool = True,
    ) -> None:
        self.store = store
        self.project_id = project_id
        self.hidden_dim = hidden_dim
        self.pacing = pacing
        self._catalogue = all_demo_ideas()

    def _exclusions(self) -> set[str]:
        """The set of summaries the project has already seen, normalized for matching."""
        rows = self.store.list_recent(self.project_id, limit=200)
        return {_norm(r.text.split("\n")[0]) for r in rows}

    def _zero_hidden(self) -> torch.Tensor:
        """Placeholder hidden vector — we're not running the model, but the store
        wants a 1-D tensor so it can pretend the demo is on the same code path
        as a real generation. Future cloud-14B re-extraction will replace this
        with real residual-stream captures."""
        return torch.zeros(self.hidden_dim, dtype=torch.float32)

    def _select_next(
        self,
        prompt: str,
        emitted_in_this_call: list[str],
    ) -> DemoIdea | None:
        """Pick the next idea, brief-aware.

        Strategy:
        1. Detect the most likely pack from the brief's keywords. If matched,
           prefer ideas in that pack first; only fall through to other packs
           once the pack is exhausted (after exclusion).
        2. Within the preferred pack, rotate deterministically by
           hash(prompt, project_id). Anti-mode + within-call exclusions filter
           the candidate set.
        3. Fallback: rotate through the full catalogue.
        """
        already = self._exclusions() | {_norm(s) for s in emitted_in_this_call}
        seed = _hash_seed(prompt, self.project_id)

        preferred_pack = _pack_for_prompt(prompt)
        if preferred_pack and preferred_pack in CONCEPTIAL_PACKS:
            pack_ideas = CONCEPTIAL_PACKS[preferred_pack]
            cand = self._first_unblocked(pack_ideas, seed, already)
            if cand is not None:
                return cand

        # Fallback: any pack.
        return self._first_unblocked(self._catalogue, seed, already)

    @staticmethod
    def _first_unblocked(
        ideas: list[DemoIdea], seed: int, exclude: set[str]
    ) -> DemoIdea | None:
        n = len(ideas)
        for k in range(n):
            cand = ideas[(seed + k) % n]
            if _norm(cand.summary) not in exclude:
                return cand
        return None

    def stream(self, *, prompt: str, n: int) -> Iterator[StreamingIdea]:
        """Yield N ideas, persisting each, applying realistic SSE pacing."""
        emitted_summaries: list[str] = []
        pauses = _pacing_seconds(n) if self.pacing else [0.0] * n
        for k in range(n):
            if pauses[k] > 0:
                time.sleep(pauses[k])
            cand = self._select_next(prompt, emitted_summaries)
            if cand is None:
                break
            # Snapshot exclusions BEFORE adding the new idea so the snapshot
            # reflects "what the model knew about" at decision time.
            exclusions_snapshot = tuple(sorted(self._exclusions()))
            idea_id = uuid.uuid4().hex
            self.store.add(
                project_id=self.project_id,
                idea_id=idea_id,
                hidden=self._zero_hidden(),
                text=f"{cand.summary}\n{cand.content}",
                source=SOURCE_GENERATED,
            )
            emitted_summaries.append(cand.summary)
            yield StreamingIdea(
                idea_id=idea_id,
                summary=cand.summary,
                content=cand.content,
                pack=cand.pack,
                similarity_score=0.0,
                excluded_summaries=exclusions_snapshot,
            )


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


# ---------------------------------------------------------------------------
# Singleton accessor for the FastAPI server.
# ---------------------------------------------------------------------------

_STORE_SINGLETON: IdeaMemoryStore | None = None


def get_demo_store() -> IdeaMemoryStore:
    """Lazy singleton IdeaMemoryStore for the streaming endpoint.

    Path override via ``CONCEPTIAL_IDEA_DB_PATH``; default lives at
    ``./data/idea_memory.db`` so tests + smokes share state with the server."""
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        path = Path(os.environ.get("CONCEPTIAL_IDEA_DB_PATH", "./data/idea_memory.db"))
        _STORE_SINGLETON = IdeaMemoryStore(db_path=path)
    return _STORE_SINGLETON


__all__ = [
    "ConceptialStreamingSource",
    "StreamingIdea",
    "get_demo_store",
]
