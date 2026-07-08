"""Experiment 1 — per-layer matched extract-and-inject alpha sweep.

For every layer ``L`` and every ``alpha`` (both signs), inject ``alpha * d_L`` at layer
``L`` and score the resulting creativity. This localises *where* the creativity direction
has the largest behavioural effect and *which sign/strength* maximises it — exactly the
conformity paper's Experiment 1, retargeted to creativity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "layers": self.layers,
            "alphas": self.alphas,
            "best": self.best_cell(),
            "cells": self.cells,
        }


def select_promising_layers(sweep_results: dict, n: int) -> list[int]:
    """Rank layers by their best sweep cell and return the top ``n``.

    Operates on the serialized sweep dict from either ``per_layer_alpha_sweep``
    (``SweepResult``) or ``two_stage_cell_sweep`` (``TwoStageSweepResult``) — i.e. the
    ``sweep_results.json`` Experiment 1 writes. Each layer is scored by its best cell over
    alphas; layers are then ranked, ties broken by ascending layer index.

    For a two-stage sweep the cells are on two scales: judged cells (the top-k by Stage-1)
    carry the full two-stage ``composite``; un-judged cells keep their Stage-1 score. We rank
    **judged-first** — every layer with a judged cell ranks above every un-judged layer,
    ordered by its best judged two-stage composite — then fill remaining slots from un-judged
    layers by their best Stage-1 score (the only signal present on one scale for every cell).
    A per-item sweep has no judging, so layers simply rank by best ``composite``.
    """
    cells = sweep_results.get("cells", [])
    two_stage = sweep_results.get("evaluator") == "two_stage"

    judged_best: dict[int, float] = {}  # layer -> best two-stage composite (judged cells)
    stage1_best: dict[int, float] = {}  # layer -> best consistent score (every cell)
    for c in cells:
        if "layer" not in c:
            continue
        layer = int(c["layer"])
        # Stage-1 is present on every two-stage cell; the per-item cell only has ``composite``,
        # which is itself the consistent score there.
        base = c.get("stage1", c.get("composite"))
        if base is None:
            continue
        base = float(base)
        if base > stage1_best.get(layer, float("-inf")):
            stage1_best[layer] = base
        if two_stage and c.get("judged"):
            comp = float(c.get("composite", base))
            if comp > judged_best.get(layer, float("-inf")):
                judged_best[layer] = comp

    if judged_best:
        judged = sorted(judged_best, key=lambda layer: (-judged_best[layer], layer))
        rest = sorted(
            (layer for layer in stage1_best if layer not in judged_best),
            key=lambda layer: (-stage1_best[layer], layer),
        )
        ranked = judged + rest
    else:
        ranked = sorted(stage1_best, key=lambda layer: (-stage1_best[layer], layer))
    return ranked[:n]


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
