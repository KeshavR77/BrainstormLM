"""Build the ``Injection`` lists for the three patching designs.

All injections use the *unit* DiffMeans direction; magnitude is set by ``alpha`` (and, when
``rescale=True``, matched to each token's activation norm inside the backend hook). Using
the unit direction keeps ``alpha`` comparable across layers and capture positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.derivation.archetype_bank import ArchetypeBank
from creativity_steering.derivation.diffmeans import unit_normalize
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


@dataclass
class OrthoArchetypeDirs:
    """Per-sample steering directions from an orthogonalised archetype bank (X1×X2).

    Attributes
    ----------
    dirs:
        ``(batch_size, hidden_dim)`` float32 — the direction assigned to each batch sample
        (round-robin over the K modes), ready for ``HFBackend.generate_steered_batch``.
    retention_cos:
        ``(k,)`` cosine of each orthonormalised mode with its source archetype, *before* any
        fallback. A low value means that mode was heavily distorted by Gram–Schmidt.
    fell_back:
        Indices (original archetype order) whose retention dropped below the threshold and were
        therefore left as the *raw* unit archetype instead of the orthogonalised direction.
    order:
        The most-distinct-first ordering used for orthogonalisation.
    """

    dirs: np.ndarray
    retention_cos: np.ndarray
    fell_back: list[int]
    order: list[int]


def build_orthogonal_archetype_dirs(
    bank: ArchetypeBank,
    batch_size: int,
    *,
    retention_threshold: float = 0.5,
) -> OrthoArchetypeDirs:
    """Fuse the archetype bank (X2) with the orthonormal frame (X1): *orthogonalise the bank*.

    Decorrelates the K archetype DiffMeans directions so a batch fans across distinct **and**
    mutually-orthogonal modes, with two guardrails against the fact that "creativity" archetypes
    are often highly correlated (Gram–Schmidt then distorts the later vectors most):

    1. **Order most-distinct-first** (``distinctness_order``) before orthogonalising, so the salient
       modes keep their meaning and the distortion lands on the redundant ones.
    2. **Cosine-retention fallback** — any orthogonalised mode whose cosine with its source
       archetype drops below ``retention_threshold`` is replaced by the *raw* unit archetype (no
       orthogonalisation) and recorded in ``fell_back``. Too-collinear archetypes thus degrade
       gracefully to plain round-robin assignment instead of drifting into noise.

    Steering vectors are non-identifiable (arXiv:2602.06801): this geometry is necessary, not
    sufficient — validate the spread it produces on a real model.
    """
    from creativity_steering.patching.orthogonal import distinctness_order, orthonormalize_rows

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    unit = unit_normalize(np.asarray(bank.vectors, dtype=np.float32))  # (k, d), unit rows
    k = unit.shape[0]
    order = distinctness_order(unit)
    ortho_in_order = orthonormalize_rows(unit[order])  # Gram–Schmidt in most-distinct-first order
    ortho = np.empty_like(ortho_in_order)
    ortho[order] = ortho_in_order  # scatter back to original archetype slots

    # Both ortho and unit rows are unit-norm, so the row-wise dot product is the cosine.
    retention_cos = np.einsum("ij,ij->i", ortho, unit).astype(np.float32)
    final = ortho.copy()
    fell_back: list[int] = []
    for i in range(k):
        if retention_cos[i] < retention_threshold:
            final[i] = unit[i]  # too collinear to orthogonalise cleanly -> keep the raw archetype
            fell_back.append(i)

    idx = [i % k for i in range(batch_size)]
    dirs = final[idx].astype(np.float32)
    return OrthoArchetypeDirs(
        dirs=dirs,
        retention_cos=retention_cos,
        fell_back=fell_back,
        order=[int(x) for x in order],
    )
