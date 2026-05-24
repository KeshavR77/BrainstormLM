from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedder. All implementations must return L2-normalized (n, d) float32 arrays."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns: np.ndarray of shape (len(texts), embedding_dim), L2-normalized, float32.
        """

    @staticmethod
    def l2_normalize(emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (emb / norms).astype(np.float32)
