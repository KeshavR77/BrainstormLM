"""Corpius Cloud runtime: Qwen2.5-14B-Instruct with dual-layer steering (L20 + L36).

Isolated from ``corpius.py`` / frozen Corpius 1 config. Uses :mod:`shrooms_llm.pipeline`
and :mod:`shrooms_llm.steering_presets` only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from brainstormlm.conceptial.engine.config import SteeringConfig, SteeringStrategy
from brainstormlm.conceptial.engine.pipeline import SteeringEngine
from brainstormlm.conceptial.engine.steering_presets import apply_v8_idea_forward_phase_shift

CLOUD_MODEL_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"
CLOUD_CREATIVE_LAYER = 20
CLOUD_TECHNICAL_LAYER = 36


def _cloud_steering_config(
    model_id: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    attn_implementation: str | None = "flash_attention_2",
) -> SteeringConfig:
    return SteeringConfig(
        model_id=model_id,
        strategy=SteeringStrategy.CONTROL,
        layer_injections=[],
        layer_alphas={},
        decay_tokens=0,
        torch_dtype="bfloat16",
        preferred_device="auto",
        device_map=device_map,
        attn_implementation=attn_implementation,  # type: ignore[arg-type]
        inject_last_token_only=False,
    )


def build_cloud_engine(
    steering_dir: Path,
    creative_vector: Path,
    technical_vector: Path,
    *,
    model_id: str = CLOUD_MODEL_DEFAULT,
    creative_layer: int = CLOUD_CREATIVE_LAYER,
    technical_layer: int = CLOUD_TECHNICAL_LAYER,
    creative_start_alpha: float = 2.0,
    exp_zero_by_step: int = 132,
    technical_start_alpha: float = 0.68,
    sigmoid_mid_step: float = 62.0,
    sigmoid_steepness: float = 0.1,
    steering_math: str = "spherical",
    device_map: str | dict[str, Any] | None = "auto",
    force_sdpa: bool = False,
) -> SteeringEngine:
    """
    Load 14B in bfloat16 on GPU(s) with flash_attention_2 when available.

    If ``force_sdpa`` is True or flash_attention_2 is unavailable, falls back to SDPA
    (still CUDA-accelerated on Ampere+).
    """
    attn: str | None = "sdpa" if force_sdpa else "flash_attention_2"
    if not force_sdpa and os.environ.get("CORPIUS_CLOUD_FORCE_SDPA", "").lower() in ("1", "true", "yes"):
        attn = "sdpa"
    cfg = _cloud_steering_config(model_id, device_map=device_map, attn_implementation=attn)
    engine = SteeringEngine(config=cfg)
    try:
        apply_v8_idea_forward_phase_shift(
            engine,
            Path(steering_dir),
            creative_vector_path=Path(creative_vector),
            technical_vector_path=Path(technical_vector),
            creative_layer=int(creative_layer),
            technical_layer=int(technical_layer),
            creative_start_alpha=float(creative_start_alpha),
            exp_zero_by_step=int(exp_zero_by_step),
            technical_start_alpha=float(technical_start_alpha),
            sigmoid_mid_step=float(sigmoid_mid_step),
            sigmoid_steepness=float(sigmoid_steepness),
            steering_math=str(steering_math),
        )
    except Exception as exc:  # pragma: no cover - env-specific (missing flash-attn)
        if attn == "flash_attention_2":
            cfg = _cloud_steering_config(model_id, device_map=device_map, attn_implementation="sdpa")
            engine = SteeringEngine(config=cfg)
            apply_v8_idea_forward_phase_shift(
                engine,
                Path(steering_dir),
                creative_vector_path=Path(creative_vector),
                technical_vector_path=Path(technical_vector),
                creative_layer=int(creative_layer),
                technical_layer=int(technical_layer),
                creative_start_alpha=float(creative_start_alpha),
                exp_zero_by_step=int(exp_zero_by_step),
                technical_start_alpha=float(technical_start_alpha),
                sigmoid_mid_step=float(sigmoid_mid_step),
                sigmoid_steepness=float(sigmoid_steepness),
                steering_math=str(steering_math),
            )
            print(
                "cloud_corpius: flash_attention_2 failed; retried with sdpa. Original error:",
                exc,
                flush=True,
            )
        else:
            raise
    return engine


class CloudCorpius:
    """Thin wrapper around :class:`~shrooms_llm.pipeline.SteeringEngine` for cloud dual-vector runs."""

    def __init__(
        self,
        steering_dir: Path,
        creative_vector: Path,
        technical_vector: Path,
        *,
        model_id: str = CLOUD_MODEL_DEFAULT,
        creative_layer: int = CLOUD_CREATIVE_LAYER,
        technical_layer: int = CLOUD_TECHNICAL_LAYER,
        creative_start_alpha: float = 2.0,
        exp_zero_by_step: int = 132,
        technical_start_alpha: float = 0.68,
        sigmoid_mid_step: float = 62.0,
        sigmoid_steepness: float = 0.1,
        steering_math: str = "spherical",
        device_map: str | dict[str, Any] | None = "auto",
        force_sdpa: bool = False,
    ) -> None:
        self.steering_dir = Path(steering_dir)
        self.creative_vector = Path(creative_vector)
        self.technical_vector = Path(technical_vector)
        self.model_id = model_id
        self.creative_layer = creative_layer
        self.technical_layer = technical_layer
        self.creative_start_alpha = creative_start_alpha
        self.exp_zero_by_step = exp_zero_by_step
        self.technical_start_alpha = technical_start_alpha
        self.sigmoid_mid_step = sigmoid_mid_step
        self.sigmoid_steepness = sigmoid_steepness
        self.steering_math = steering_math
        self.device_map = device_map
        self.force_sdpa = force_sdpa
        self._engine: SteeringEngine | None = None

    @property
    def engine(self) -> SteeringEngine:
        if self._engine is None:
            self._engine = build_cloud_engine(
                self.steering_dir,
                self.creative_vector,
                self.technical_vector,
                model_id=self.model_id,
                creative_layer=self.creative_layer,
                technical_layer=self.technical_layer,
                creative_start_alpha=self.creative_start_alpha,
                exp_zero_by_step=self.exp_zero_by_step,
                technical_start_alpha=self.technical_start_alpha,
                sigmoid_mid_step=self.sigmoid_mid_step,
                sigmoid_steepness=self.sigmoid_steepness,
                steering_math=self.steering_math,
                device_map=self.device_map,
                force_sdpa=self.force_sdpa,
            )
        return self._engine

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.72,
        do_sample: bool = True,
        seed: int | None = None,
    ) -> str:
        if seed is not None:
            s = int(seed)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        return self.engine.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            do_steering=True,
        )
