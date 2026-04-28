"""Single switch point between API-served, locally-served, and steered generators."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from brainstormlm.config import get_settings
from brainstormlm.model.base import TogetherGenerator
from brainstormlm.model.hf import HFGenerator
from brainstormlm.model.interface import Generator
from brainstormlm.model.steered import SteeredGenerator


def make_generator(
    adapter_path: Path | None = None,
    *,
    steering_path: Path | None = None,
    steering_strength: float | None = None,
    steering_layers: Sequence[int] | None = None,
) -> Generator:
    """Pick the right generator for the given adapter / steering combo."""
    if steering_path is not None:
        settings = get_settings()
        return SteeredGenerator(
            steering_path=steering_path,
            steering_strength=(
                steering_strength
                if steering_strength is not None
                else settings.default_steering_strength
            ),
            steering_layers=steering_layers,
            adapter_path=adapter_path,
        )
    if adapter_path is not None:
        return HFGenerator(adapter_path=adapter_path)
    return TogetherGenerator()
