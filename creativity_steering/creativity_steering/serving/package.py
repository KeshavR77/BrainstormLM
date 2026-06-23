"""A *placeable* steering package: the {vector, layer, alpha} spec(s) you ship for testing.

A package is a small JSON file. It can carry a single (layer, alpha) — or several named
**variants**, each a list of (layer, alpha) injections — so the API user chooses which steering
to apply per request (e.g. just ``L17`` or the ``bracket`` L14+L16+L17). Swapping the shipped
config is just pointing the harness at a different package — no code change.

Back-compat: a package with top-level ``layer``/``alpha`` (and no ``variants``) is normalised to
a single-variant set, so the simple single-layer form keeps working unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.backend import Injection


@dataclass
class SteeringPackage:
    name: str
    vectors_path: str            # dir containing vectors.npz + manifest.json
    layer: int | None = None     # single-layer shorthand (mutually optional with `variants`)
    alpha: float | None = None
    variants: dict[str, list[dict]] = field(default_factory=dict)  # name -> [{layer, alpha}, ...]
    default_variant: str | None = None
    rescale: bool = True
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    device: str | None = None
    dtype: str = "bf16"
    system: str | None = None    # system prompt at generation; None = clean (steering-only A/B)
    max_new_tokens: int = 256
    temperature: float = 0.9
    seed: int = 0

    def __post_init__(self):
        # Normalise the single-layer shorthand into the variant map.
        if not self.variants:
            if self.layer is None or self.alpha is None:
                raise ValueError("package needs either `variants` or top-level `layer`+`alpha`")
            vname = f"L{self.layer}"
            self.variants = {vname: [{"layer": self.layer, "alpha": self.alpha}]}
            self.default_variant = self.default_variant or vname
        else:
            self.default_variant = self.default_variant or next(iter(self.variants))
            if self.default_variant not in self.variants:
                raise ValueError(f"default_variant {self.default_variant!r} not in variants")

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringPackage":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    @classmethod
    def load(cls, path: str | Path) -> "SteeringPackage":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def variant_names(self) -> list[str]:
        return list(self.variants)

    def all_layers(self) -> set[int]:
        """Every layer referenced by any variant (for range validation)."""
        return {int(i["layer"]) for spec in self.variants.values() for i in spec}

    def injections(self, vs: SteeringVectorSet, variant: str | None = None) -> list[Injection]:
        """Build the injection list for ``variant`` (default variant if None). Each entry uses
        its own layer's unit DiffMeans direction × alpha — the matched extract-and-inject."""
        name = variant or self.default_variant
        if name not in self.variants:
            raise KeyError(f"unknown variant {name!r}; have {self.variant_names()}")
        unit = vs.unit_vectors()
        return [
            Injection(layer=int(i["layer"]), vector=unit[int(i["layer"])],
                      alpha=float(i["alpha"]), rescale=self.rescale)
            for i in self.variants[name]
        ]

    def public(self) -> dict:
        """Serialisable provenance for the /package + /health endpoints (no secrets)."""
        default = self.variants[self.default_variant]
        return {
            "name": self.name,
            "model_id": self.model_id,
            "vectors_path": self.vectors_path,
            "default_variant": self.default_variant,
            "variants": self.variants,
            # convenience: the default variant's first layer/alpha (single-layer ships)
            "layer": default[0]["layer"],
            "alpha": default[0]["alpha"],
            "rescale": self.rescale,
            "system": self.system,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }
