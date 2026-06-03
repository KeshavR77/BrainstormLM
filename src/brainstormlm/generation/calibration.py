"""Tail-sampling temperature/probability calibration sweep (sprint items 1.2/1.5/1.8).

Verbalized Sampling exposes two knobs that trade diversity against coherence:

  - **temperature** of the generating model, and
  - **modal_floor** of ``tail_sample`` (how much weight the modal candidate keeps).

This module sweeps a grid over both, measuring the resulting set's **diversity**
(mean pairwise distinctness over the sampled multiset) and a **coherence proxy**,
then recommends the operating point with the most diversity that still clears a
coherence floor. Pure/deterministic given a seed; the LLM + embedder are injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from brainstormlm.generation.diversity_critic import mean_distinctness
from brainstormlm.generation.verbalized_sampling import tail_sample, verbalized_sample


class LLMCallable(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray:
        ...


def is_coherent(text: str, *, min_chars: int = 3, max_non_ascii: float = 0.05) -> bool:
    """Cheap coherence proxy — guards the micro-output and non-ASCII salad modes.

    Returns False for empty/too-short text or text whose non-ASCII fraction exceeds
    ``max_non_ascii``. Not a quality judge — just a floor that keeps the sweep from
    recommending an incoherent tail.
    """
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    non_ascii = sum(ord(ch) > 127 for ch in t)
    if non_ascii / len(t) > max_non_ascii:
        return False
    return True


@dataclass
class CellMetrics:
    modal_floor: float
    temperature: float
    diversity: float       # mean pairwise distinctness of the sampled multiset
    distinct_count: int     # number of unique ideas sampled
    coherence: float        # fraction of sampled ideas passing is_coherent
    n_ideas: int


@dataclass
class SweepResult:
    cells: list[CellMetrics]
    recommended: CellMetrics | None

    def to_rows(self) -> list[dict]:
        """Flatten to dict rows for CSV/JSON export."""
        return [
            {
                "modal_floor": c.modal_floor,
                "temperature": c.temperature,
                "diversity": c.diversity,
                "distinct_count": c.distinct_count,
                "coherence": c.coherence,
                "n_ideas": c.n_ideas,
                "recommended": c is self.recommended,
            }
            for c in self.cells
        ]


def sweep_tail_calibration(
    brief: str,
    *,
    llm: LLMCallable,
    embedder: EmbedCallable,
    floors: list[float],
    temperatures: list[float],
    m: int = 50,
    vs_n: int = 6,
    seed: int = 0,
    coherence_floor: float = 0.5,
    min_chars: int = 3,
) -> SweepResult:
    """Sweep ``modal_floor`` x ``temperature``; return per-cell metrics + a recommendation.

    For each temperature we issue ONE verbalized-sampling call (temperature flows through
    to the LLM), then for each floor draw ``m`` tail samples from that distribution and
    measure the resulting multiset. Per-cell RNG is reseeded from ``seed`` so the whole
    sweep is reproducible.

    The recommendation is the coherent cell (``coherence >= coherence_floor``) with the
    highest diversity, tie-broken by distinct count then by the lower floor.
    """
    cells: list[CellMetrics] = []

    for temp in temperatures:
        items = verbalized_sample(brief, llm=llm, n=vs_n, temperature=temp)
        for floor in floors:
            rng = np.random.default_rng(seed)
            if items:
                picks = [tail_sample(items, rng=rng, modal_floor=floor).text for _ in range(m)]
            else:
                picks = []

            diversity = mean_distinctness(picks, embedder=embedder) if len(picks) >= 2 else 0.0
            distinct_count = len(set(picks))
            coherence = (
                sum(is_coherent(p, min_chars=min_chars) for p in picks) / len(picks)
                if picks
                else 0.0
            )

            cells.append(
                CellMetrics(
                    modal_floor=float(floor),
                    temperature=float(temp),
                    diversity=float(diversity),
                    distinct_count=int(distinct_count),
                    coherence=float(coherence),
                    n_ideas=len(picks),
                )
            )

    coherent = [c for c in cells if c.coherence >= coherence_floor]
    recommended = (
        max(coherent, key=lambda c: (c.diversity, c.distinct_count, -c.modal_floor))
        if coherent
        else None
    )

    return SweepResult(cells=cells, recommended=recommended)


__all__ = [
    "is_coherent",
    "CellMetrics",
    "SweepResult",
    "sweep_tail_calibration",
]
