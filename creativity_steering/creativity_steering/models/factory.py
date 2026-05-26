"""Build a backend from an ExperimentConfig."""

from __future__ import annotations

from creativity_steering.config import ExperimentConfig
from creativity_steering.models.backend import ModelBackend


def build_backend(config: ExperimentConfig) -> ModelBackend:
    if config.backend == "mock":
        from creativity_steering.models.mock import MockBackend

        return MockBackend(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            creative_layer=config.mock_creative_layer,
            seed=config.seed,
        )
    if config.backend == "hf":
        from creativity_steering.models.hf import HFBackend

        return HFBackend(config.model_id, device=config.device, dtype=config.dtype)
    raise ValueError(f"unknown backend {config.backend!r}; use 'mock' or 'hf'")
