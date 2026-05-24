from __future__ import annotations
import numpy as np
from .base import BaseEmbedder

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


class LocalEmbedder(BaseEmbedder):
    """
    SentenceTransformer-based embedder. No API key required.
    Default model: BAAI/bge-large-en-v1.5 (1024 dims).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)
