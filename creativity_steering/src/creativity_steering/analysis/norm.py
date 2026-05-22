"""Norm analysis — which layers have the largest DiffMeans direction.

A large ``||d_L||`` means the creative/neutral framings move the residual stream a lot at
layer ``L``: a coarse but useful signal for where the creativity contrast is strongest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormAnalysis:
    layer_norms: np.ndarray  # (n_layers,)
    ranked_layers: list[int]  # layer indices, largest norm first

    @property
    def top_layer(self) -> int:
        return self.ranked_layers[0]

    def to_dict(self) -> dict:
        return {
            "layer_norms": [float(x) for x in self.layer_norms],
            "ranked_layers": list(self.ranked_layers),
            "top_layer": self.top_layer,
        }


def analyze_norms(vectors: np.ndarray) -> NormAnalysis:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1)
    ranked = [int(i) for i in np.argsort(-norms)]
    return NormAnalysis(layer_norms=norms, ranked_layers=ranked)
