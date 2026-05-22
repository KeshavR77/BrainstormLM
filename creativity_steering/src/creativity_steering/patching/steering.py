"""Build the ``Injection`` lists for the three patching designs.

All injections use the *unit* DiffMeans direction; magnitude is set by ``alpha`` (and, when
``rescale=True``, matched to each token's activation norm inside the backend hook). Using
the unit direction keeps ``alpha`` comparable across layers and capture positions.
"""

from __future__ import annotations

from collections.abc import Sequence

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.backend import Injection


def inject_single_layer(
    vector_set: SteeringVectorSet, layer: int, alpha: float, *, rescale: bool = True
) -> list[Injection]:
    """Exp 1: the layer-``L`` vector injected at layer ``L`` (matched extract-and-inject)."""
    unit = vector_set.unit_vectors()
    return [Injection(layer=layer, vector=unit[layer], alpha=alpha, rescale=rescale)]


def inject_all_layers(
    vector_set: SteeringVectorSet, source_layer: int, alpha: float, *, rescale: bool = True
) -> list[Injection]:
    """Exp 2: one chosen vector (``source_layer``'s direction) injected at *every* layer."""
    unit = vector_set.unit_vectors()
    v = unit[source_layer]
    return [
        Injection(layer=L, vector=v, alpha=alpha, rescale=rescale)
        for L in range(vector_set.n_layers)
    ]


def inject_multi(
    vector_set: SteeringVectorSet,
    layers: Sequence[int],
    alpha: float | Sequence[float],
    *,
    rescale: bool = True,
) -> list[Injection]:
    """Combo: several layer-specific vectors injected at their *own* layers in one run."""
    unit = vector_set.unit_vectors()
    layers = list(layers)
    if isinstance(alpha, (int, float)):
        alphas = [float(alpha)] * len(layers)
    else:
        alphas = [float(a) for a in alpha]
        if len(alphas) != len(layers):
            raise ValueError("alpha sequence length must match layers length")
    return [
        Injection(layer=L, vector=unit[L], alpha=a, rescale=rescale)
        for L, a in zip(layers, alphas)
    ]
