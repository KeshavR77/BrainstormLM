"""Shared data records for the derivation phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResponseExample:
    """One generated response plus its cached per-layer activations and creativity score."""

    framing: str  # "creative" | "neutral" — which framing produced it
    system: str
    prompt: str
    response: str
    activations: dict[str, np.ndarray] = field(default_factory=dict)  # position -> (L, H)
    score: float | None = None  # final composite (quality-gated novelty in v3); provenance
    # Selection signals set by the evaluator filter (separate from ``score`` so the contrast
    # poles can be picked by novelty while ``score`` carries the gated composite):
    novelty: float | None = None  # batch-relative Stage-1 novelty (model-free; defined for all)
    quality_ok: bool | None = None  # raw judge pragmaticality cleared the quality floor


@dataclass
class VerifiedSubset:
    """Examples that passed the evaluator filter, split into positive / negative pools."""

    positives: list[ResponseExample]  # creative-framed AND scored creative
    negatives: list[ResponseExample]  # neutral-framed AND scored not-creative
    n_dropped: int  # examples that fell in the ambiguous middle / wrong-side

    @property
    def n_positive(self) -> int:
        return len(self.positives)

    @property
    def n_negative(self) -> int:
        return len(self.negatives)

    def stacks(self, position: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positive_stack, negative_stack)`` of shape ``(n, n_layers, hidden_dim)``."""
        if not self.positives or not self.negatives:
            raise ValueError("verified subset has an empty pool; cannot build activation stacks")
        pos = np.stack([ex.activations[position] for ex in self.positives])
        neg = np.stack([ex.activations[position] for ex in self.negatives])
        return pos, neg
