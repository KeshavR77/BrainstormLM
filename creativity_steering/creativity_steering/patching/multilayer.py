"""Experiment 2 — push creativity past the single-layer ceiling.

Two designs, both starting from the promising layers the Experiment 1 sweep surfaced
(top behavioural-creativity layers; see ``sweep.select_promising_layers``):

  * ``all_layers_single_vector`` — inject one chosen vector at *every* layer with a small
    alpha (the conformity paper's Experiment 2). Tests how large an effect a single
    direction can produce when reinforced across the whole forward pass.
  * ``multi_vector_combo`` — inject several layer-specific vectors at their own layers in
    one run. Tests whether stacking complementary directions beats any single one.

Both return a ``CellScore`` and a delta vs the unsteered baseline so they are directly
comparable to Experiment 1 cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.backend import ModelBackend
from creativity_steering.patching.evaluate import DEFAULT_BRIEFS, CellScore, score_briefs
from creativity_steering.patching.steering import inject_all_layers, inject_multi
from creativity_steering.scoring import CreativityScorer


@dataclass
class MultiLayerResult:
    label: str
    score: CellScore
    baseline: CellScore
    detail: dict

    @property
    def delta(self) -> float:
        return self.score.composite - self.baseline.composite

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score.to_dict(),
            "baseline": self.baseline.to_dict(),
            "delta": self.delta,
            "detail": self.detail,
        }


def _score(backend, scorer, injections, briefs, n_samples, max_new_tokens, temperature, seed):
    return score_briefs(
        backend,
        scorer,
        injections=injections,
        briefs=briefs,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


def all_layers_single_vector(
    backend: ModelBackend,
    vector_set: SteeringVectorSet,
    scorer: CreativityScorer,
    *,
    source_layer: int,
    alpha: float = 0.1,
    rescale: bool = True,
    briefs: Sequence[str] = DEFAULT_BRIEFS,
    n_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
) -> MultiLayerResult:
    base = _score(backend, scorer, [], briefs, n_samples, max_new_tokens, temperature, seed)
    inj = inject_all_layers(vector_set, source_layer, alpha, rescale=rescale)
    sc = _score(backend, scorer, inj, briefs, n_samples, max_new_tokens, temperature, seed + 1)
    return MultiLayerResult(
        label=f"all_layers(src=L{source_layer}, alpha={alpha})",
        score=sc,
        baseline=base,
        detail={"source_layer": source_layer, "alpha": alpha, "n_inject_layers": vector_set.n_layers},
    )


def multi_vector_combo(
    backend: ModelBackend,
    vector_set: SteeringVectorSet,
    scorer: CreativityScorer,
    *,
    layers: Sequence[int],
    alpha: float | Sequence[float],
    rescale: bool = True,
    briefs: Sequence[str] = DEFAULT_BRIEFS,
    n_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
) -> MultiLayerResult:
    base = _score(backend, scorer, [], briefs, n_samples, max_new_tokens, temperature, seed)
    inj = inject_multi(vector_set, layers, alpha, rescale=rescale)
    sc = _score(backend, scorer, inj, briefs, n_samples, max_new_tokens, temperature, seed + 1)
    return MultiLayerResult(
        label=f"multi_vector(layers={list(layers)}, alpha={alpha})",
        score=sc,
        baseline=base,
        detail={"layers": list(layers), "alpha": alpha},
    )
