from __future__ import annotations

from typing import Any

import torch

from brainstormlm.conceptial.engine.config import SteeringConfig
from brainstormlm.conceptial.engine.pipeline import (
    SteeringEngine,
    load_steering_vectors_from_layer_injections,
    resolve_hf_token,
)

"""Primary HF model id for Corpius V2 / overnight Qwen3B development."""
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_engine(
    config: SteeringConfig | None = None,
    *,
    token: str | None = None,
    model_id: str | None = None,
) -> SteeringEngine:
    """
    Construct a :class:`~shrooms_llm.pipeline.SteeringEngine` and load weights/tokenizer.

    ``model_id`` overrides ``config.model_id`` when provided.
    """
    cfg = config or SteeringConfig()
    if model_id is not None:
        cfg.model_id = model_id
    eng = SteeringEngine(cfg, token=token if token is not None else resolve_hf_token())
    eng.load()
    return eng


def load_steering_vectors_for_config(
    config: SteeringConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[int, torch.Tensor]:
    """Load per-layer steering tensors from ``layer_injections`` (CPU tensors)."""
    return load_steering_vectors_from_layer_injections(config, map_location=map_location)


def generate_with_steering(
    engine: SteeringEngine,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    do_steering: bool = True,
    do_sample: bool = True,
    temperature: float = 0.7,
    use_ghost_rejection: bool = False,
    **gen_kwargs: Any,
) -> str:
    """
    Decode **full** prompt+completion string (benchmark scripts often prefer :meth:`SteeringEngine.complete`).
    """
    engine.load()
    return engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_steering=do_steering,
        use_ghost_rejection=use_ghost_rejection,
        do_sample=do_sample,
        temperature=temperature,
        **gen_kwargs,
    )
