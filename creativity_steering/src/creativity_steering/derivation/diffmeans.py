"""DiffMeans (Contrastive Activation Addition) — pure numpy, one direction per layer.

Given stacks of per-example, per-layer residual activations for the *positive*
(verified-creative) and *negative* (verified-not-creative) subsets, the DiffMeans
direction at layer ``L`` is

    d_L = mean_i(h^L_positive_i) - mean_j(h^L_negative_j).

This is the canonical CAA / persona-vector / conformity-paper construction. We keep the
*raw* (un-normalised) vectors because their per-layer L2 norm is itself a signal the
analysis phase uses (norm analysis); unit-normalisation is a separate, explicit step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _check_stack(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3-D (n_examples, n_layers, hidden_dim); got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty; need at least one example to take a mean")
    return arr


def diffmeans_per_layer(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Return the per-layer DiffMeans matrix of shape ``(n_layers, hidden_dim)``.

    ``positive`` / ``negative`` are ``(n_examples, n_layers, hidden_dim)`` stacks.
    """
    pos = _check_stack(positive, "positive")
    neg = _check_stack(negative, "negative")
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError(
            f"positive {pos.shape[1:]} and negative {neg.shape[1:]} layer/dim shapes differ"
        )
    return pos.mean(axis=0) - neg.mean(axis=0)


def unit_normalize(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise each layer's vector to unit length; zero vectors stay zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms < eps, vectors, vectors / np.maximum(norms, eps))


@dataclass
class DiffMeansResult:
    """Per-layer DiffMeans vectors plus the counts and norms used downstream."""

    vectors: np.ndarray  # (n_layers, hidden_dim), raw (un-normalised)
    layer_norms: np.ndarray  # (n_layers,) L2 norm per layer
    n_positive: int
    n_negative: int

    @classmethod
    def from_stacks(cls, positive: np.ndarray, negative: np.ndarray) -> "DiffMeansResult":
        vectors = diffmeans_per_layer(positive, negative)
        return cls(
            vectors=vectors,
            layer_norms=np.linalg.norm(vectors, axis=-1),
            n_positive=int(np.asarray(positive).shape[0]),
            n_negative=int(np.asarray(negative).shape[0]),
        )

    @property
    def n_layers(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.vectors.shape[1])

    def unit_vectors(self) -> np.ndarray:
        return unit_normalize(self.vectors)
