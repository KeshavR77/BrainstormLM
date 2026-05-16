"""Percent uplift helpers for Corpius benchmark JSON summaries (Vendi + near-duplicate rate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _arm_summary(blob: dict[str, Any], arm: str) -> dict[str, Any]:
    by = blob.get("summary_by_arm") or {}
    if arm not in by:
        raise KeyError(f"arm {arm!r} not in summary_by_arm; have {sorted(by.keys())}")
    return by[arm]


def mean_vendi_from_vendi_json(path: Path | str, arm: str = "qwen_v17") -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(_arm_summary(data, arm)["mean_vendi"])


def mean_near_duplicate_from_clusters_json(path: Path | str, arm: str = "qwen_v17") -> float:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(_arm_summary(data, arm)["mean_near_duplicate_rate"])


def percent_uplifts(
    baseline_vendi: float,
    baseline_near_dup: float,
    candidate_vendi: float,
    candidate_near_dup: float,
    *,
    vendi_weight: float = 1.0,
    near_dup_weight: float = 1.0,
) -> dict[str, float]:
    """Return relative % change in Vendi (higher better) and % reduction in NearDup (higher better).

    NearDup is framed as *improvement* when the candidate rate is lower than baseline:
    ``100 * (baseline - candidate) / baseline``.
    """
    if baseline_vendi <= 0:
        raise ValueError("baseline_vendi must be > 0")
    if baseline_near_dup <= 0:
        raise ValueError("baseline_near_dup must be > 0")

    vendi_rel_pct = 100.0 * (float(candidate_vendi) - float(baseline_vendi)) / float(baseline_vendi)
    neardup_reduction_pct = 100.0 * (float(baseline_near_dup) - float(candidate_near_dup)) / float(
        baseline_near_dup
    )
    combined = float(vendi_weight) * vendi_rel_pct + float(near_dup_weight) * neardup_reduction_pct
    return {
        "vendi_relative_pct": vendi_rel_pct,
        "neardup_reduction_pct": neardup_reduction_pct,
        "combined_uplift_pct": combined,
    }
