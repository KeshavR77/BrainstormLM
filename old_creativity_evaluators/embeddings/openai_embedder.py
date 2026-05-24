from __future__ import annotations
import math
import os
import time
import numpy as np
from .base import BaseEmbedder

_DEFAULT_MODEL = "text-embedding-3-large"
_MAX_TOKENS_PER_BATCH = 200_000
_CHARS_PER_TOKEN = 4
_MAX_CHUNK_CHARS = 32_000


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-large embedder (1536 dims).
    Batches requests and retries with exponential backoff.
    Logic adapted from MAS_Diversity compute_vendi_extended.py.
    """

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self._api_key)
        all_embeddings: list[np.ndarray] = []

        # Build batches by estimated token count
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0
        for text in texts:
            estimated = len(text) // _CHARS_PER_TOKEN + 1
            if current_batch and current_tokens + estimated > _MAX_TOKENS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(text)
            current_tokens += estimated
        if current_batch:
            batches.append(current_batch)

        for batch in batches:
            # Chunk oversized individual texts
            safe_batch: list[str] = []
            chunk_map: list[int] = []  # index into safe_batch for each original text
            for text in batch:
                if len(text) <= _MAX_CHUNK_CHARS:
                    chunk_map.append(len(safe_batch))
                    safe_batch.append(text)
                else:
                    chunks = [
                        text[i : i + _MAX_CHUNK_CHARS]
                        for i in range(0, len(text), _MAX_CHUNK_CHARS)
                    ]
                    chunk_map.append(len(safe_batch))
                    safe_batch.extend(chunks)

            raw = self._call_with_backoff(client, safe_batch)

            # Reconstruct per-original-text embeddings
            idx = 0
            for orig_idx, text in enumerate(batch):
                if len(text) <= _MAX_CHUNK_CHARS:
                    all_embeddings.append(np.array(raw[idx], dtype=np.float32))
                    idx += 1
                else:
                    n_chunks = math.ceil(len(text) / _MAX_CHUNK_CHARS)
                    chunk_embs = np.array(raw[idx : idx + n_chunks], dtype=np.float32)
                    mean_emb = chunk_embs.mean(axis=0)
                    all_embeddings.append(mean_emb)
                    idx += n_chunks

        emb = np.stack(all_embeddings, axis=0)
        return self.l2_normalize(emb)

    def _call_with_backoff(self, client, texts: list[str]) -> list[list[float]]:
        delay = 1.0
        for attempt in range(6):
            try:
                response = client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as exc:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")
