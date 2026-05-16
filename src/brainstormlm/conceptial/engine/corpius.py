"""Corpius 1 runtime definition.

Corpius 1 is the frozen v17 activation-steered Qwen2.5-3B model used for
current citation and retest workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from brainstormlm.conceptial.engine.pipeline import SteeringEngine, load_steering_checkpoint
from brainstormlm.conceptial.engine.steering_presets import apply_v10_parametric_preset
from brainstormlm.conceptial.engine.vector_purify import purify_vector


# After the Day-2 consolidation, the module lives at:
#   BrainstormLM/src/brainstormlm/conceptial/engine/corpius.py
# so parents[4] resolves to the BrainstormLM repo root.
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CORPIUS_CONFIG = ROOT / "config" / "conceptial.json"


@dataclass(frozen=True)
class CorpiusSpec:
    id: str
    display_name: str
    base_model: str
    benchmark_arm: str
    local_variants: str
    local_arms: str
    steering_method: str
    creative_vector: Path
    technical_vector: Path
    creative_start_alpha: float
    exp_zero_by_step: int
    technical_start_alpha: float
    sigmoid_mid_step: float
    sigmoid_steepness: float
    steering_math: str

    @property
    def schedule_kwargs(self) -> dict[str, float | int]:
        return {
            "creative_start_alpha": self.creative_start_alpha,
            "exp_zero_by_step": self.exp_zero_by_step,
            "technical_start_alpha": self.technical_start_alpha,
            "sigmoid_mid_step": self.sigmoid_mid_step,
            "sigmoid_steepness": self.sigmoid_steepness,
        }


def _repo_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def load_corpius_spec(path: Path = DEFAULT_CORPIUS_CONFIG) -> CorpiusSpec:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    vectors = raw["steering_vectors"]
    schedule = raw["schedule"]
    return CorpiusSpec(
        id=str(raw["id"]),
        display_name=str(raw["display_name"]),
        base_model=str(raw["base_model"]),
        benchmark_arm=str(raw["benchmark_arm"]),
        local_variants=str(raw["local_variants"]),
        local_arms=str(raw["local_arms"]),
        steering_method=str(raw["steering_method"]),
        creative_vector=_repo_path(str(vectors["creative"])),
        technical_vector=_repo_path(str(vectors["technical"])),
        creative_start_alpha=float(schedule["creative_start_alpha"]),
        exp_zero_by_step=int(schedule["exp_zero_by_step"]),
        technical_start_alpha=float(schedule["technical_start_alpha"]),
        sigmoid_mid_step=float(schedule["sigmoid_mid_step"]),
        sigmoid_steepness=float(schedule["sigmoid_steepness"]),
        steering_math=str(schedule["steering_math"]),
    )


def build_iterative_creative_vector(
    creative_vector_path: Path,
    prior_first_token_hiddens: list,
) -> torch.Tensor:
    """Orthogonalize the creative steering direction against the mean previous first-token residual.

    ``prior_first_token_hiddens`` are 1D CPU float tensors (hidden dim) captured at the creative
    layer on the first generated token of prior samples.
    """
    c_path = Path(creative_vector_path)
    bundle = load_steering_checkpoint(c_path, map_location="cpu")
    v = bundle["vector"]
    if not prior_first_token_hiddens:
        return v
    mean_h = torch.stack([h.float().reshape(-1) for h in prior_first_token_hiddens], dim=0).mean(0)
    return purify_vector(v, [mean_h])


def apply_corpius_1(engine: SteeringEngine, spec: CorpiusSpec | None = None) -> None:
    """Apply the frozen Corpius 1 v17 steering schedule to an already-loaded engine."""
    spec = spec or load_corpius_spec()
    apply_v10_parametric_preset(
        engine,
        spec.creative_vector.parent,
        "v17",
        creative_vector_path=spec.creative_vector,
        technical_vector_path=spec.technical_vector,
        steering_math=spec.steering_math,
        **spec.schedule_kwargs,
    )
