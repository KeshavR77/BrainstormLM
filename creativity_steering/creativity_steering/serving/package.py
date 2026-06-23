"""A *placeable* steering package: the {vector, layer, alpha} spec you ship for testing.

A package is a small JSON file. Swapping the shipped config (a different layer/alpha, or a
different vector directory) is just pointing the harness at a different package — no code
change. The package knows how to turn itself into the single-layer ``Injection`` the backend
applies during generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.backend import Injection
from creativity_steering.patching.steering import inject_single_layer


@dataclass
class SteeringPackage:
    name: str
    vectors_path: str            # dir containing vectors.npz + manifest.json
    layer: int
    alpha: float
    rescale: bool = True
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    device: str | None = None
    dtype: str = "bf16"
    system: str | None = None    # system prompt at generation; None = clean (steering-only A/B)
    max_new_tokens: int = 256
    temperature: float = 0.9
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringPackage":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    @classmethod
    def load(cls, path: str | Path) -> "SteeringPackage":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def injections(self, vs: SteeringVectorSet) -> list[Injection]:
        """The single-layer injection this package applies (unit direction × alpha)."""
        return inject_single_layer(vs, self.layer, self.alpha, rescale=self.rescale)

    def public(self) -> dict:
        """Serialisable provenance for the /package + /health endpoints (no secrets)."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "vectors_path": self.vectors_path,
            "layer": self.layer,
            "alpha": self.alpha,
            "rescale": self.rescale,
            "system": self.system,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
