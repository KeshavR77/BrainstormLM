"""Activation steering (RepE) utilities for Hugging Face causal LMs."""

from brainstormlm.conceptial.engine.config import Normalization, Pooling, SteeringConfig, SteeringStrategy, validate_layer_injections
from brainstormlm.conceptial.engine.hooks import register_steering_hook, steering_context
from brainstormlm.conceptial.engine.pipeline import (
    SteeringEngine,
    apply_layer_injections,
    apply_steering_checkpoint,
    load_steering_checkpoint,
    load_steering_vectors_from_layer_injections,
    resolve_hf_token,
)

__all__ = [
    "Normalization",
    "Pooling",
    "SteeringConfig",
    "SteeringEngine",
    "SteeringStrategy",
    "apply_layer_injections",
    "apply_steering_checkpoint",
    "load_steering_checkpoint",
    "load_steering_vectors_from_layer_injections",
    "register_steering_hook",
    "resolve_hf_token",
    "steering_context",
    "validate_layer_injections",
]
