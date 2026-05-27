"""Experiment 1 — per-layer matched extract-and-inject alpha sweep.

For every layer ``L`` and every ``alpha`` (both signs), inject ``alpha * d_L`` at layer
``L`` and score the resulting creativity. This localises *where* the creativity direction
has the largest behavioural effect and *which sign/strength* maximises it — exactly the
conformity paper's Experiment 1, retargeted to creativity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.backend import ModelBackend
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS, CellScore, score_briefs
from creativity_steering.patching.steering import inject_single_layer
from creativity_steering.scoring import CreativityScorer


@dataclass
class SweepResult:
    baseline: CellScore
    cells: list[dict]  # each: layer, alpha, composite, mean_single, collection, delta
    layers: list[int]
    alphas: list[float]

    def best_cell(self) -> dict:
        return max(self.cells, key=lambda c: c["composite"])

    def composite_matrix(self) -> np.ndarray:
        """``(n_layers, n_alphas)`` composite-score grid (rows=layers, cols=alphas)."""
        m = np.full((len(self.layers), len(self.alphas)), np.nan, dtype=np.float32)
        li = {L: i for i, L in enumerate(self.layers)}
        ai = {a: j for j, a in enumerate(self.alphas)}
        for c in self.cells:
            m[li[c["layer"]], ai[c["alpha"]]] = c["composite"]
        return m

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "layers": self.layers,
            "alphas": self.alphas,
            "best": self.best_cell(),
            "cells": self.cells,
        }


def per_layer_alpha_sweep(
    backend: ModelBackend,
    vector_set: SteeringVectorSet,
    scorer: CreativityScorer,
    *,
    alphas: Sequence[float],
    layers: Sequence[int] | None = None,
    rescale: bool = True,
    briefs: Sequence[str] = DEFAULT_BRIEFS,
    n_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
) -> SweepResult:
    layers = list(range(vector_set.n_layers)) if layers is None else [int(L) for L in layers]
    alphas = [float(a) for a in alphas]

    def _score(injections, sd):
        return score_briefs(
            backend,
            scorer,
            injections=injections,
            briefs=briefs,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=sd,
        )

    baseline = _score([], seed)
    cells: list[dict] = []
    for L in layers:
        for a in alphas:
            cell = _score(inject_single_layer(vector_set, L, a, rescale=rescale), seed + L)
            row = {"layer": L, "alpha": a, **cell.to_dict()}
            row["delta"] = cell.composite - baseline.composite
            cells.append(row)
    return SweepResult(baseline=baseline, cells=cells, layers=layers, alphas=alphas)
