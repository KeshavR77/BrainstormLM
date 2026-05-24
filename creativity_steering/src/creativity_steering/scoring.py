"""Creativity scoring adapter.

The derivation filter and the patching evaluation both need a single creativity number
per response (and per collection). We isolate that behind ``CreativityScorer`` so the
underlying evaluator can be swapped later (the user explicitly wants this option):

  * ``EmbeddingCreativityScorer`` (factory names ``"embedding"`` and ``"real"``) — a
    self-contained embedding + lexical per-item scorer (prompt distance + distinct-2 + low
    repetition). Needs the ``embedding`` extra (sentence-transformers); imported lazily so the
    rest of the package stays offline. This is the per-item scorer used by the *patching*
    evaluation. The former vendored ``creativity_evaluator`` wrapper has been retired — the
    richer two-stage signal (DSI + LLM-as-judge) now lives in the sibling
    ``creativity_evaluators`` package and is used for *derivation* filtering via
    ``derivation/evaluator_filter.py``.
  * ``HeuristicScorer`` (factory name ``"heuristic"``) — dependency-free lexical proxy. The
    default for offline/mock runs and the whole test suite.
  * ``FakeScorer`` — returns caller-supplied scores; for deterministic unit tests.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _distinct_2(text: str) -> float:
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return len(set(bigrams)) / len(bigrams)


def _bigram_repetition(text: str) -> float:
    """Within-text bigram repetition rate in [0, 1]; higher = more repetitive."""
    toks = _tokens(text)
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks[:-1], toks[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


class CreativityScorer(ABC):
    """A scorer turns text into a creativity number in [0, 1]."""

    @abstractmethod
    def score_single(self, text: str, prompt: str) -> float:
        """Creativity/novelty of one response against its prompt."""

    @abstractmethod
    def score_collection(self, texts: list[str], prompt: str | None = None) -> float:
        """Diversity of a collection of responses to the same prompt."""


class HeuristicScorer(CreativityScorer):
    """Offline lexical proxy — no model downloads."""

    def score_single(self, text: str, prompt: str) -> float:  # noqa: ARG002 — prompt unused
        toks = _tokens(text)
        if len(toks) < 2:
            return 0.0
        # distinct-2 is the discriminative signal; a gentle length gate only penalises
        # near-empty text (so a long repetitive answer still scores low).
        gate = min(len(toks) / 20.0, 1.0)
        return float(max(0.0, min(1.0, _distinct_2(text) * gate)))

    def score_collection(self, texts: list[str], prompt: str | None = None) -> float:
        if not texts:
            return 0.0
        # Cross-text bigram distinctness as a cheap diversity stand-in.
        joined_bigrams: list[tuple[str, str]] = []
        for t in texts:
            toks = _tokens(t)
            joined_bigrams.extend(zip(toks[:-1], toks[1:]))
        if not joined_bigrams:
            return 0.0
        return len(set(joined_bigrams)) / len(joined_bigrams)


class FakeScorer(CreativityScorer):
    """Returns scores from a callable keyed on text — for tests."""

    def __init__(self, single_fn: Callable[[str], float], collection: float = 0.5) -> None:
        self._single = single_fn
        self._collection = collection

    def score_single(self, text: str, prompt: str) -> float:  # noqa: ARG002
        return float(self._single(text))

    def score_collection(self, texts: list[str], prompt: str | None = None) -> float:
        return float(self._collection)


class EmbeddingCreativityScorer(CreativityScorer):
    """Self-contained embedding + lexical scorer (no dependency on the removed eval pkg).

    ``score_single`` mirrors the former evaluator's no-judge single-response formula:
    50% prompt→response cosine distance + 30% distinct-2 + 20% (1 − bigram repetition).
    ``score_collection`` uses mean pairwise embedding distance as a diversity proxy.

    Needs ``sentence-transformers`` (the ``real`` extra); imported lazily so the rest of the
    package stays offline. The model is loaded once and cached.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _embed(self, texts: list[str]):
        import numpy as np

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return np.asarray(
            self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float32,
        )

    def score_single(self, text: str, prompt: str) -> float:
        if not text:
            return 0.0
        import numpy as np

        emb = self._embed([prompt, text])
        dist = float(1.0 - np.dot(emb[0], emb[1]))  # cosine distance, embeddings L2-normalised
        dist_norm = min(max(dist, 0.0) / 2.0, 1.0)
        novelty_lex = max(0.0, 1.0 - _bigram_repetition(text))
        score = 0.5 * dist_norm + 0.3 * _distinct_2(text) + 0.2 * novelty_lex
        return float(max(0.0, min(1.0, score)))

    def score_collection(self, texts: list[str], prompt: str | None = None) -> float:  # noqa: ARG002
        if len(texts) < 2:
            return 0.0
        import numpy as np

        emb = self._embed(list(texts))
        sims = emb @ emb.T
        n = sims.shape[0]
        off_diag = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        return float(max(0.0, min(1.0, 1.0 - off_diag)))


def make_scorer(name: str) -> CreativityScorer:
    """Factory for the per-item *patching-eval* scorer.

    ``"embedding"`` / ``"real"`` → ``EmbeddingCreativityScorer`` (the former vendored "real"
    evaluator has been retired); ``"heuristic"`` → ``HeuristicScorer``. (``"fake"`` is built
    directly in tests.) Derivation *filtering* with the full two-stage stack is configured
    separately via ``FilterConfig.method == "evaluator_stack"`` (see
    ``derivation/evaluator_filter.py``).
    """
    name = name.lower()
    if name in ("real", "embedding"):
        return EmbeddingCreativityScorer()
    if name == "heuristic":
        return HeuristicScorer()
    raise ValueError(f"unknown scorer {name!r}; use 'real', 'embedding', or 'heuristic'")
