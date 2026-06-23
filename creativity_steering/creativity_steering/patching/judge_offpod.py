"""Off-pod LLM-judging of persisted sweep generations.

A Phase-1 sweep runs on-pod with judging OFF and writes every generation to
``sweep_generations.jsonl`` (``{cell, prompt, response, [layer, alpha]}``). This module
re-loads those generations and runs the full ``creativity_evaluators`` stack (Stage-1 novelty
recomputed deterministically from the saved text + the 3-judge LLM ensemble) on EVERY cell in
**one batch**, then aggregates per cell into the same shape an on-pod judged sweep produces.

Why one batch: the composite z-scores every signal against the batch (see
``creativity_evaluators.composite``), so all cells must be scored together for the per-cell
composites and deltas to be on one comparable scale. No GPU and no regeneration are needed —
only the (CPU/MPS) Stage-1 models and the judge APIs.
"""

from __future__ import annotations

import json
import re

import numpy as np

_LABEL_RE = re.compile(r"^L(\d+)_a(-?[0-9.]+)$")


def load_generation_records(paths: list[str]) -> list[dict]:
    """Load + concatenate generation jsonl files, keeping the baseline from only the FIRST
    file that contains it (each chunk's sweep re-emits the baseline; duplicating it would
    over-weight it in the batch z-scoring). All steered-cell rows are kept."""
    records: list[dict] = []
    baseline_taken = False
    for p in paths:
        with open(p) as f:
            file_recs = [json.loads(line) for line in f if line.strip()]
        file_has_baseline = any(r.get("cell") == "baseline" for r in file_recs)
        for r in file_recs:
            if r.get("cell") == "baseline" and baseline_taken:
                continue
            records.append(r)
        if file_has_baseline:
            baseline_taken = True
    return records


def _parse_label(label: str, rec: dict) -> tuple[int | None, float | None]:
    if rec.get("layer") is not None and rec.get("alpha") is not None:
        return int(rec["layer"]), float(rec["alpha"])
    m = _LABEL_RE.match(label)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def judge_generations(
    records: list[dict],
    *,
    scorers=None,
    eval_config=None,
    generator_family: str = "qwen",
    quality_floor: float = 2.0,
    stage1_percentile_floor: float = 0.0,
    n_workers: int = 4,
) -> dict:
    """Judge all cells in ``records`` in one batch and aggregate per cell.

    ``stage1_percentile_floor=0`` judges every response (no Stage-2 gating); ``quality_floor``
    matches the sweep's quality-gated-novelty composite. Returns a dict shaped like an on-pod
    judged sweep result (``baseline`` + ``cells`` with composite/stage1/pragmaticality/delta).
    """
    from creativity_evaluators import EvalConfig
    from creativity_evaluators.pipeline import evaluate_batch
    from creativity_evaluators.types import PromptCategory

    if eval_config is None:
        eval_config = EvalConfig(
            enable_stage2=True,
            generator_family=generator_family,
            stage1_percentile_floor=stage1_percentile_floor,
            run_reward=False,
            composite_quality_floor=quality_floor,
        )

    batch = [
        {"prompt": r["prompt"], "response": r["response"], "category": PromptCategory.NEUTRAL}
        for r in records
    ]
    results = evaluate_batch(batch, eval_config, n_workers=n_workers, scorers=scorers)

    # group per cell, preserving first-seen order
    cells: dict[str, dict] = {}
    order: list[str] = []
    for rec, ev in zip(records, results):
        lbl = rec["cell"]
        if lbl not in cells:
            cells[lbl] = {"composite": [], "stage1": [], "prag": [], "rec": rec}
            order.append(lbl)
        c = cells[lbl]
        c["composite"].append(float(ev.creativity_score))
        c["stage1"].append(float(ev.components.get("stage1_composite", float("nan"))))
        if ev.ensemble is not None and ev.ensemble.per_dimension_mean:
            p = ev.ensemble.per_dimension_mean.get("Pragmaticality")
            if p is not None:
                c["prag"].append(float(p))

    def _agg(c):
        comp = float(np.mean(c["composite"]))
        s1 = float(np.nanmean(c["stage1"])) if c["stage1"] else float("nan")
        prag = float(np.mean(c["prag"])) if c["prag"] else None
        return comp, s1, prag, len(c["composite"])

    baseline_comp = _agg(cells["baseline"])[0] if "baseline" in cells else 0.0

    baseline_dict = None
    out_cells: list[dict] = []
    judged: list[str] = []
    for lbl in order:
        comp, s1, prag, n = _agg(cells[lbl])
        is_judged = bool(cells[lbl]["prag"])
        if is_judged:
            judged.append(lbl)
        entry = {
            "label": lbl,
            "composite": comp,
            "stage1": s1,
            "pragmaticality": prag,
            "judged": is_judged,
            "n_samples": n,
            "delta": comp - baseline_comp,
        }
        if lbl == "baseline":
            entry["delta"] = 0.0
            baseline_dict = entry
        else:
            layer, alpha = _parse_label(lbl, cells[lbl]["rec"])
            entry["layer"] = layer
            entry["alpha"] = alpha
            out_cells.append(entry)

    out_cells.sort(key=lambda c: c["composite"], reverse=True)
    return {
        "evaluator": "two_stage_offpod",
        "enable_judge": True,
        "judged_labels": judged,
        "baseline": baseline_dict,
        "cells": out_cells,
    }
