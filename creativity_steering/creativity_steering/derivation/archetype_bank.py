"""X2: Archetype-indexed DiffMeans bank.

Holds K creativity sub-vectors — one per "mode of creativity" (e.g. structural,
metaphorical, contrarian) — derived from the existing DiffMeans pipeline.  Each
sub-vector is indexed by archetype index k and assigned to batch sample k so that
a batch fans across modes.

Design notes:
  - Each archetype vector is the DiffMeans direction at a chosen layer.  Using
    different layers as archetypes is the simplest reuse of the derivation
    pipeline: different layers capture different levels of abstraction, so
    layer-indexed directions naturally differ in character.
  - The bank does NOT guarantee semantic distinctness (steering vectors are
    non-identifiable, arXiv:2602.06801).  Validate semantic spread with a real
    model run.
  - ``assign_to_batch`` wraps round-robin when batch_size > k, so a bank of K
    modes can serve any batch size without error.

References: STARS (arXiv:2601.22010), archetype-DiffMeans idea (design doc X2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence as SequenceType

import numpy as np

from creativity_steering.derivation.diffmeans import diffmeans_per_layer, unit_normalize


@dataclass
class ArchetypeBank:
    """K creativity sub-vectors, one per archetype mode, indexed 0 .. k-1.

    Parameters
    ----------
    vectors:
        2-D float32 array of shape ``(k, hidden_dim)``.  Raw (un-normalised)
        DiffMeans directions; unit-normalisation is available via ``unit_vectors()``.
    archetype_names:
        Optional list of human-readable labels, length k.  ``None`` if unnamed.
    """

    vectors: np.ndarray  # (k, hidden_dim)
    archetype_names: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (k, hidden_dim); got shape {self.vectors.shape}"
            )
        k = self.vectors.shape[0]
        if self.archetype_names is not None and len(self.archetype_names) != k:
            raise ValueError(
                f"archetype_names has {len(self.archetype_names)} entries but k={k}"
            )

    # ------------------------------------------------------------------ properties

    @property
    def k(self) -> int:
        """Number of archetype modes."""
        return int(self.vectors.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.vectors.shape[1])

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_diffmeans_result(
        cls,
        positive: np.ndarray,
        negative: np.ndarray,
        *,
        archetype_layers: SequenceType[int] | None = None,
        archetype_names: list[str] | None = None,
    ) -> "ArchetypeBank":
        """Derive the bank from positive/negative activation stacks.

        Parameters
        ----------
        positive, negative:
            ``(n_examples, n_layers, hidden_dim)`` stacks passed directly to
            ``diffmeans_per_layer``.
        archetype_layers:
            Which layer indices to use as archetype sub-vectors.  Defaults to
            all layers (one archetype per layer).
        archetype_names:
            Optional human-readable labels; must match len(archetype_layers).
        """
        dm_matrix = diffmeans_per_layer(positive, negative)  # (n_layers, hidden_dim)
        n_layers = dm_matrix.shape[0]
        if archetype_layers is None:
            archetype_layers = list(range(n_layers))
        else:
            archetype_layers = list(archetype_layers)
            for L in archetype_layers:
                if not (0 <= L < n_layers):
                    raise ValueError(
                        f"archetype_layers contains layer {L} which is out of range "
                        f"[0, {n_layers})"
                    )
        vecs = dm_matrix[archetype_layers]  # (k, hidden_dim)
        return cls(vectors=vecs, archetype_names=archetype_names)

    @classmethod
    def from_stacks(
        cls,
        positive: np.ndarray,
        negative: np.ndarray,
        *,
        archetype_layers: SequenceType[int] | None = None,
        archetype_names: list[str] | None = None,
    ) -> "ArchetypeBank":
        """Convenience alias for ``from_diffmeans_result``."""
        return cls.from_diffmeans_result(
            positive,
            negative,
            archetype_layers=archetype_layers,
            archetype_names=archetype_names,
        )

    # ------------------------------------------------------------------ access

    def get(self, k: int) -> np.ndarray:
        """Return a copy of archetype sub-vector ``k`` (raw, un-normalised)."""
        if not (0 <= k < self.k):
            raise IndexError(
                f"archetype index {k} out of range [0, {self.k})"
            )
        return self.vectors[k].copy()

    def unit_vectors(self) -> np.ndarray:
        """Return all archetype directions unit-normalised, shape ``(k, hidden_dim)``."""
        return unit_normalize(self.vectors)

    # ------------------------------------------------------------------ batch assignment

    def assign_to_batch(self, batch_size: int) -> np.ndarray:
        """Return a ``(batch_size, hidden_dim)`` array assigning sample i → archetype i % k.

        Round-robin wrapping when batch_size > k.  Raises ``ValueError`` for
        batch_size <= 0.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        indices = [i % self.k for i in range(batch_size)]
        return self.vectors[indices].copy()
