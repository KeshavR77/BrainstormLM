"""Single switch point between API-served and locally-served generators."""

from __future__ import annotations

from pathlib import Path

from brainstormlm.model.base import TogetherGenerator
from brainstormlm.model.hf import HFGenerator
from brainstormlm.model.interface import Generator


def make_generator(adapter_path: Path | None = None) -> Generator:
    """Return an HFGenerator if adapter_path is set, otherwise TogetherGenerator."""
    if adapter_path is not None:
        return HFGenerator(adapter_path=adapter_path)
    return TogetherGenerator()
