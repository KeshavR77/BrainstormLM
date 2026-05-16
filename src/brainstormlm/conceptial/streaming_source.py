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

from brainstormlm.conceptial.demo_packs import DemoIdea, all_demo_ideas
from brainstormlm.conceptial.idea_store import (
    SOURCE_GENERATED,
    IdeaMemoryStore,
)


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
        """Pick the next idea: first non-excluded one in a deterministic rotation.

        Rotation seed: hash(prompt, project_id). Within one call, we also skip
        anything we've emitted in this call (so back-to-back ideas don't repeat
        even if the store write hasn't flushed yet).
        """
        seed = _hash_seed(prompt, self.project_id)
        n_cat = len(self._catalogue)
        already = self._exclusions() | {_norm(s) for s in emitted_in_this_call}
        # Start at seed offset; walk forward until we hit one not in exclusion list.
        for k in range(n_cat):
            idx = (seed + k) % n_cat
            cand = self._catalogue[idx]
            if _norm(cand.summary) not in already:
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
