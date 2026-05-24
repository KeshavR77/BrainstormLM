"""Divergent Semantic Integration (DSI).

Johnson, Kaufman, Baker, Patterson, Barbot, Green et al. (2022), *Behavior Research
Methods*, DOI 10.3758/s13428-022-01986-2. Reference implementation + validation:
osf.io/ath2s. DSI explained up to 72% of variance in human creativity ratings on
narratives — our single strongest model-free signal.

Method (BERT-variant, the paper's best configuration):
  * embed the response with ``bert-base-uncased``;
  * take the hidden states from layers 6 and 7 and **mean-pool the two layers** per token;
  * **keep stopwords and punctuation** (the BERT DSI variant scored best with them in);
  * DSI = mean pairwise cosine *distance* across all token embeddings.

The heavy BERT path is loaded lazily and can be swapped for any ``embed_fn`` callable
(``text -> (tokens, embeddings)``), which is how the tests run offline.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from ..types import DSIResult

# A token-level embedder: text -> (token strings, (n_tokens, hidden) float array).
EmbedFn = Callable[[str], "tuple[list[str], np.ndarray]"]

# Lightweight "does this look like source code?" markers (the response is still scored;
# this only sets a debug flag, since code redistributes the semantic space oddly).
_CODE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"^\s*(def|class|import|from|return|for|while|if)\b", re.MULTILINE),
    re.compile(r"[{};]\s*$", re.MULTILINE),
    re.compile(r"\b(print|range|len|np|self)\s*\("),
)


def _looks_like_code(text: str) -> bool:
    hits = sum(1 for pat in _CODE_PATTERNS if pat.search(text))
    return hits >= 2


def mean_pairwise_cosine_distance(emb: np.ndarray) -> float:
    """Mean cosine distance over all unordered token pairs.

    Returns 0.0 for fewer than two tokens (no pairs) or when every token shares a direction.
    Zero-norm rows contribute cosine similarity 0 (distance 1) rather than dividing by zero.
    """
    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)
    sim = normed @ normed.T
    iu = np.triu_indices(n, k=1)
    dist = 1.0 - sim[iu]
    return float(np.clip(dist, 0.0, 2.0).mean())


class DSIScorer:
    """Scores one response's DSI.

    Parameters
    ----------
    model_name, layers, device:
        Configure the lazily-built BERT embedder (used only when ``embed_fn`` is None).
    min_tokens:
        Responses with fewer tokens return ``score=None, reason="too_short"``.
    chunk_size:
        For very long responses, compute DSI per consecutive chunk of this many tokens and
        average — keeps the pairwise matrix tractable. ``None`` disables chunking.
    embed_fn:
        Inject a custom embedder (text -> (tokens, embeddings)). Primarily for testing.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        layers: tuple[int, int] = (6, 7),
        device: str | None = None,
        min_tokens: int = 5,
        chunk_size: int | None = None,
        embed_fn: EmbedFn | None = None,
    ):
        self.model_name = model_name
        self.layers = layers
        self.device = device
        self.min_tokens = min_tokens
        self.chunk_size = chunk_size
        self._embed_fn = embed_fn
        self._bert = None  # lazily-built (tokenizer, model) tuple

    def score(self, text: str) -> DSIResult:
        tokens, emb = self._embed(text)
        n_tokens = len(tokens)
        is_code = _looks_like_code(text)

        if n_tokens < self.min_tokens:
            return DSIResult(
                score=None,
                n_tokens=n_tokens,
                layer_indices=self.layers,
                reason="too_short",
                is_code=is_code,
            )

        if self.chunk_size and n_tokens > self.chunk_size:
            chunk_scores = [
                mean_pairwise_cosine_distance(emb[i : i + self.chunk_size])
                for i in range(0, n_tokens, self.chunk_size)
                if emb[i : i + self.chunk_size].shape[0] >= 2
            ]
            score = float(np.mean(chunk_scores))
            n_chunks = len(chunk_scores)
        else:
            score = mean_pairwise_cosine_distance(emb)
            n_chunks = 1

        return DSIResult(
            score=score,
            n_tokens=n_tokens,
            layer_indices=self.layers,
            is_code=is_code,
            n_chunks=n_chunks,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> tuple[list[str], np.ndarray]:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        return self._bert_embed(text)

    def _bert_embed(self, text: str) -> tuple[list[str], np.ndarray]:
        """Real path: mean-pool BERT layers 6+7 per (non-special) token. Loaded lazily."""
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        if self._bert is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
            model.eval()
            if self.device:
                model.to(self.device)
            self._bert = (tokenizer, model)
        tokenizer, model = self._bert

        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, return_special_tokens_mask=True
        )
        special = enc.pop("special_tokens_mask")[0].bool()
        if self.device:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        # hidden_states: tuple of (num_layers + 1) tensors, each (1, seq, hidden).
        lo, hi = self.layers
        pooled = (out.hidden_states[lo] + out.hidden_states[hi]) / 2.0  # (1, seq, hidden)
        pooled = pooled[0]
        keep = ~special  # drop [CLS]/[SEP] etc.; keep stopwords + punctuation
        emb = pooled[keep].detach().cpu().numpy().astype(np.float64)
        ids = enc["input_ids"][0][keep].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return tokens, emb
