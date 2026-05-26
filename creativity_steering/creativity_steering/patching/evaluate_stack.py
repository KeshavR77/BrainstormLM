"""Score steered generations with the two-stage ``creativity_evaluators`` stack.

This is the patching-phase counterpart to derivation's ``evaluator_filter``: instead of the
per-item ``CreativityScorer`` (``patching/evaluate.py``), score every steered generation with
the full stack — Stage 1 model-free signals (DSI, lexical, perplexity, optional reward) plus
an optional Stage-2 LLM-as-judge ensemble — so the layer/alpha winner is chosen by the same
rich signal used to build the contrast set.

Two things make this different from the per-item path:

* **Batch-relative scoring.** The evaluators' composite is z-scored *within the batch* it is
  given, so cells cannot be scored independently and compared. We generate *every* cell's
  outputs first and score them in one batch, so the per-cell means are on one scale.
* **Judge gating (cost control).** Stage 1 runs over all generations (cheap, local). The
  expensive Stage-2 judge ensemble runs only on the generations from the **top-k cells** by
  Stage-1 score (plus the baseline, so deltas stay meaningful), and the winner is chosen by
  the full two-stage composite among those. Cells outside the top-k keep their Stage-1 score.

Pass injected ``scorers`` (``creativity_evaluators.Scorers``) to run offline in tests; in a
real run leave it ``None`` and the stack is built from the ``EvalConfig`` (loading models +
judges). The sibling-package import stays lazy via the same path bridge as the filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from creativity_steering.derivation.evaluator_filter import (
    _ensure_creativity_evaluators_importable,
)
from creativity_steering.models.backend import Injection, ModelBackend
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS


@dataclass
class StackCell:
    """One steering configuration to score: a label, its injections, and free-form meta
    (e.g. ``{"layer": L, "alpha": a}``) that is echoed into the result for the CLI."""

    label: str
    injections: list[Injection]
    meta: dict = field(default_factory=dict)


@dataclass
class StackCellScore:
    label: str
    meta: dict
    composite: float  # mean two-stage score if judged, else mean Stage-1 score
    stage1: float  # mean Stage-1-only score (always present)
    judged: bool
    n_samples: int
    delta: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            **self.meta,
            "composite": self.composite,
            "stage1": self.stage1,
            "judged": self.judged,
            "n_samples": self.n_samples,
            "delta": self.delta,
        }


@dataclass
class TwoStageSweepResult:
    baseline: StackCellScore
    cells: list[StackCellScore]
    judged_labels: list[str]
    enable_judge: bool

    def best_cell(self) -> dict:
        """Best non-baseline cell. When judging is on, restrict to judged cells (only those
        were fully evaluated on one scale); otherwise rank all cells by Stage-1 composite."""
        candidates = self.cells
        if self.enable_judge and any(c.judged for c in self.cells):
            candidates = [c for c in self.cells if c.judged]
        if not candidates:
            candidates = self.cells
        return max(candidates, key=lambda c: c.composite).to_dict()

    def to_dict(self) -> dict:
        return {
            "evaluator": "two_stage",
            "enable_judge": self.enable_judge,
            "judged_labels": self.judged_labels,
            "baseline": self.baseline.to_dict(),
            "best": self.best_cell() if self.cells else None,
            "cells": [c.to_dict() for c in self.cells],
        }


def _generate_cell_records(
    backend: ModelBackend,
    cells: Sequence[StackCell],
    *,
    briefs: Sequence[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    """Generate every cell's steered outputs. Returns ``(records, cell_index_per_record)``
    where ``records`` are ``{prompt, response, category}`` dicts ready for ``evaluate_batch``.
    The briefs carry no creative framing, so every record is tagged ``NEUTRAL``; the steering
    vector — not the prompt — is what any creativity signal must come from."""
    from creativity_evaluators import PromptCategory

    records: list[dict] = []
    cell_idx: list[int] = []
    for ci, cell in enumerate(cells):
        for bi, brief in enumerate(briefs):
            for s in range(n_samples):
                text = backend.generate_steered(
                    brief,
                    injections=cell.injections,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed * 10_000 + ci * 1_000 + bi * 100 + s,
                )
                records.append(
                    {"prompt": brief, "response": text, "category": PromptCategory.NEUTRAL}
                )
                cell_idx.append(ci)
    return records, cell_idx


def _mean_by_cell(scores: list[float], cell_idx: list[int], n_cells: int) -> list[float]:
    sums = [0.0] * n_cells
    counts = [0] * n_cells
    for s, ci in zip(scores, cell_idx):
        sums[ci] += float(s)
        counts[ci] += 1
    return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(n_cells)]


def two_stage_cell_sweep(
    backend: ModelBackend,
    cells: Sequence[StackCell],
    *,
    eval_config=None,  # creativity_evaluators.EvalConfig; built from kwargs if None
    scorers=None,  # creativity_evaluators.Scorers; built from eval_config if None (real run)
    briefs: Sequence[str] = DEFAULT_BRIEFS,
    n_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
    enable_judge: bool = True,
    judge_top_k: int = 3,
    generator_family: str | None = "qwen",
    stage1_floor_percentile: float = 20.0,
    use_reward_fallback: bool = False,
    stage1_device: str | None = None,
    baseline_injections: Sequence[Injection] = (),
) -> TwoStageSweepResult:
    """Generate all cells, score with the two-stage stack, gate judges to the top-k cells.

    The first scored "cell" is always the unsteered baseline (``baseline_injections``,
    empty by default). Returns per-cell composites and a ``best_cell`` chosen by the full
    two-stage score among judged cells.
    """
    _ensure_creativity_evaluators_importable()
    from creativity_evaluators import EvalConfig
    from creativity_evaluators.pipeline import build_scorers, evaluate_batch

    # baseline first, then the candidate cells.
    full_cells: list[StackCell] = [
        StackCell(label="baseline", injections=list(baseline_injections), meta={"baseline": True})
    ] + list(cells)
    n_cells = len(full_cells)

    # The judging config carries Stage-2 on (so build_scorers makes the ensemble); the
    # Stage-1 pass reuses the same scorers but with Stage 2 switched off.
    if eval_config is None:
        eval_config = EvalConfig(
            enable_stage2=enable_judge,
            generator_family=generator_family,
            stage1_percentile_floor=stage1_floor_percentile,
            use_reward_fallback=use_reward_fallback,
            stage1_device=stage1_device,
        )
    judge_cfg = eval_config
    stage1_cfg = eval_config.model_copy(update={"enable_stage2": False})

    if scorers is None:
        scorers = build_scorers(judge_cfg)

    records, cell_idx = _generate_cell_records(
        backend, full_cells, briefs=briefs, n_samples=n_samples,
        max_new_tokens=max_new_tokens, temperature=temperature, seed=seed,
    )

    # --- Pass A: Stage 1 over every generation (cheap) ---
    # Rank/gate by the Stage-1 *composite component* (novelty-based: DSI/PPL/lexical), not the
    # final creativity_score. The final score is a novelty×quality geometric mean, so with no
    # quality pole (judges off in Pass A, reward model off) it collapses to 0 for every cell —
    # which would make the top-k judge gate arbitrary. stage1_composite stays discriminative.
    s1_results = evaluate_batch(records, stage1_cfg, scorers=scorers)
    s1_scores = [
        float(r.components.get("stage1_composite", r.creativity_score)) for r in s1_results
    ]
    stage1_by_cell = _mean_by_cell(s1_scores, cell_idx, n_cells)

    # --- Decide which cells get judged: baseline + top-k non-baseline by Stage-1 ---
    can_judge = enable_judge and getattr(scorers, "ensemble", None) is not None
    judged_idx: set[int] = set()
    if can_judge:
        ranked = sorted(range(1, n_cells), key=lambda i: stage1_by_cell[i], reverse=True)
        top = ranked[: max(0, judge_top_k)]
        judged_idx = {0, *top}  # always include the baseline

    # --- Pass B: Stage 2 on the judged cells' generations only ---
    composite_by_cell = list(stage1_by_cell)
    judged_flags = [False] * n_cells
    if judged_idx:
        keep = [j for j, ci in enumerate(cell_idx) if ci in judged_idx]
        judged_records = [records[j] for j in keep]
        judged_cellidx = [cell_idx[j] for j in keep]
        s2_results = evaluate_batch(judged_records, judge_cfg, scorers=scorers)
        s2_scores = [float(r.creativity_score) for r in s2_results]
        s2_by_cell = _mean_by_cell(s2_scores, judged_cellidx, n_cells)
        for ci in judged_idx:
            composite_by_cell[ci] = s2_by_cell[ci]
            judged_flags[ci] = True

    n_per_cell = [cell_idx.count(ci) for ci in range(n_cells)]
    baseline_composite = composite_by_cell[0]

    def _score(ci: int) -> StackCellScore:
        return StackCellScore(
            label=full_cells[ci].label,
            meta={k: v for k, v in full_cells[ci].meta.items() if k != "baseline"},
            composite=composite_by_cell[ci],
            stage1=stage1_by_cell[ci],
            judged=judged_flags[ci],
            n_samples=n_per_cell[ci],
            delta=composite_by_cell[ci] - baseline_composite,
        )

    baseline_score = _score(0)
    cell_scores = [_score(ci) for ci in range(1, n_cells)]
    judged_labels = [full_cells[ci].label for ci in sorted(judged_idx)]

    return TwoStageSweepResult(
        baseline=baseline_score,
        cells=cell_scores,
        judged_labels=judged_labels,
        enable_judge=enable_judge,
    )


def build_alpha_cells(vector_set, layers, alphas, *, rescale: bool = True) -> list[StackCell]:
    """Enumerate per-layer × per-alpha cells (Experiment 1 grid) for the two-stage sweep."""
    from creativity_steering.patching.steering import inject_single_layer

    cells: list[StackCell] = []
    for layer in layers:
        for alpha in alphas:
            cells.append(
                StackCell(
                    label=f"L{layer}_a{alpha}",
                    injections=inject_single_layer(vector_set, layer, alpha, rescale=rescale),
                    meta={"layer": int(layer), "alpha": float(alpha)},
                )
            )
    return cells


def build_all_layers_cells(
    vector_set, source_layers, alphas, *, rescale: bool = True
) -> list[StackCell]:
    """Experiment 2a cells: one chosen layer's vector injected at *every* layer.

    Tests how large an effect a single direction produces when reinforced across the whole
    forward pass (needs a small alpha). One cell per (source_layer, alpha)."""
    from creativity_steering.patching.steering import inject_all_layers

    cells: list[StackCell] = []
    for src in source_layers:
        for alpha in alphas:
            cells.append(
                StackCell(
                    label=f"allL_src{int(src)}_a{float(alpha)}",
                    injections=inject_all_layers(vector_set, int(src), float(alpha), rescale=rescale),
                    meta={
                        "experiment": "all_layers",
                        "source_layer": int(src),
                        "alpha": float(alpha),
                    },
                )
            )
    return cells


def build_multi_vector_cells(
    vector_set, layer_groups, alphas, *, rescale: bool = True
) -> list[StackCell]:
    """Experiment 2b cells: several layers' own vectors injected at their own layers at once.

    Tests whether stacking complementary directions beats any single layer. One cell per
    (layer group, alpha)."""
    from creativity_steering.patching.steering import inject_multi

    cells: list[StackCell] = []
    for group in layer_groups:
        g = [int(layer) for layer in group]
        for alpha in alphas:
            cells.append(
                StackCell(
                    label=f"multi_{'-'.join(str(layer) for layer in g)}_a{float(alpha)}",
                    injections=inject_multi(vector_set, g, float(alpha), rescale=rescale),
                    meta={"experiment": "multi_vector", "layers": g, "alpha": float(alpha)},
                )
            )
    return cells
