"""Persist and reload per-layer steering vectors.

Format (one directory per capture position):
  ``vectors.npz``   — array ``vectors`` of shape (n_layers, hidden_dim), raw (un-normalised).
  ``manifest.json`` — model id, capture position, counts, per-layer norms, framings, provenance.

This mirrors the existing repo's ``data/steering_14b`` manifest style so the artifacts are
familiar, but stores all layers at once (the project derives a vector at *every* layer).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from creativity_steering.derivation.diffmeans import DiffMeansResult, unit_normalize


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class SteeringVectorSet:
    """Per-layer DiffMeans vectors for one capture position, plus provenance."""

    vectors: np.ndarray  # (n_layers, hidden_dim), raw
    model_id: str
    capture_position: str
    n_positive: int
    n_negative: int
    creative_system: str = ""
    neutral_system: str = ""
    derived_at: str = field(default_factory=_utc_now)

    @property
    def n_layers(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def layer_norms(self) -> np.ndarray:
        return np.linalg.norm(self.vectors, axis=-1)

    def unit_vectors(self) -> np.ndarray:
        return unit_normalize(self.vectors)

    @classmethod
    def from_diffmeans(
        cls,
        result: DiffMeansResult,
        *,
        model_id: str,
        capture_position: str,
        creative_system: str = "",
        neutral_system: str = "",
    ) -> "SteeringVectorSet":
        return cls(
            vectors=result.vectors,
            model_id=model_id,
            capture_position=capture_position,
            n_positive=result.n_positive,
            n_negative=result.n_negative,
            creative_system=creative_system,
            neutral_system=neutral_system,
        )

    def manifest(self) -> dict:
        return {
            "model_id": self.model_id,
            "capture_position": self.capture_position,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "normalization": "raw_diffmeans",
            "layer_norms": [float(x) for x in self.layer_norms],
            "framings": {"creative": self.creative_system, "neutral": self.neutral_system},
            "derived_at": self.derived_at,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "vectors.npz", vectors=self.vectors.astype(np.float32))
        (p / "manifest.json").write_text(json.dumps(self.manifest(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVectorSet":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        vectors = np.load(p / "vectors.npz")["vectors"]
        framings = manifest.get("framings", {})
        return cls(
            vectors=vectors,
            model_id=manifest["model_id"],
            capture_position=manifest["capture_position"],
            n_positive=int(manifest["n_positive"]),
            n_negative=int(manifest["n_negative"]),
            creative_system=framings.get("creative", ""),
            neutral_system=framings.get("neutral", ""),
            derived_at=manifest.get("derived_at", ""),
        )
