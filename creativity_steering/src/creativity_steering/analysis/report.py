"""Fuse norm + cosine + oracle into a hypothesis of which layers to patch.

The spec: "From this data, we should hypothesise a good guess of what layer vectors are
best for activation patching." We z-score each signal, add a bonus for membership in the
high-cosine band, and rank layers by the composite. The top layers are the candidates the
patching phase sweeps first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from creativity_steering.analysis.cosine import CosineAnalysis, analyze_cosine
from creativity_steering.analysis.norm import NormAnalysis, analyze_norms
from creativity_steering.analysis.oracle import (
    CREATIVITY_QUESTION,
    ActivationOracle,
    oracle_layer_scan,
)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sd = x.std()
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


@dataclass
class CandidateReport:
    norm: NormAnalysis
    cosine: CosineAnalysis
    oracle_scores: np.ndarray | None
    composite: np.ndarray
    candidate_layers: list[int]

    def to_dict(self) -> dict:
        return {
            "candidate_layers": list(self.candidate_layers),
            "composite": [float(x) for x in self.composite],
            "norm": self.norm.to_dict(),
            "cosine": self.cosine.to_dict(),
            "oracle_scores": None
            if self.oracle_scores is None
            else [float(x) for x in self.oracle_scores],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Vector analysis — candidate layers",
            "",
            f"**Top candidates for patching:** {self.candidate_layers}",
            f"**High-cosine band:** layers {self.cosine.band[0]}–{self.cosine.band[1]}",
            f"**Largest-norm layer:** {self.norm.top_layer}",
            "",
            "| layer | norm | neighbor_cos | oracle | composite |",
            "|------:|-----:|-------------:|-------:|----------:|",
        ]
        oracle = self.oracle_scores
        for L in range(len(self.composite)):
            o = "—" if oracle is None else f"{oracle[L]:.2f}"
            lines.append(
                f"| {L} | {self.norm.layer_norms[L]:.3f} | "
                f"{self.cosine.neighbor_cosine[L]:.3f} | {o} | {self.composite[L]:.3f} |"
            )
        return "\n".join(lines)


def build_report(
    vectors: np.ndarray,
    *,
    oracle: ActivationOracle | None = None,
    question: str = CREATIVITY_QUESTION,
    k_neighbor: int = 5,
    n_candidates: int = 5,
) -> CandidateReport:
    vectors = np.asarray(vectors, dtype=np.float32)
    norm = analyze_norms(vectors)
    cosine = analyze_cosine(vectors, k=k_neighbor)

    composite = _zscore(norm.layer_norms) + _zscore(cosine.neighbor_cosine)
    oracle_scores = None
    if oracle is not None:
        oracle_scores = oracle_layer_scan(oracle, vectors, question=question)
        composite = composite + _zscore(oracle_scores)

    # Bonus for sitting inside the mutually-aligned band.
    band_lo, band_hi = cosine.band
    band_bonus = np.zeros(len(composite), dtype=np.float32)
    band_bonus[band_lo : band_hi + 1] = 0.5
    composite = composite + band_bonus

    candidate_layers = [int(i) for i in np.argsort(-composite)[:n_candidates]]
    return CandidateReport(
        norm=norm,
        cosine=cosine,
        oracle_scores=oracle_scores,
        composite=composite,
        candidate_layers=candidate_layers,
    )
