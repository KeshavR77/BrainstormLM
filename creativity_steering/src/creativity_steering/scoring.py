"""Creativity scoring adapter.

The derivation filter and the patching evaluation both need a single creativity number
per response (and per collection). We isolate that behind ``CreativityScorer`` so the
underlying evaluator can be swapped later (the user explicitly wants this option):

  * ``RealCreativityScorer`` — wraps ``creativity_evaluator.CreativityEvaluator`` (the
    repo's existing evaluator). Loads a sentence-transformer, so it is used in real runs
    and the demo, not in fast unit tests.
  * ``HeuristicScorer`` — dependency-free lexical proxy (distinct-2 + low repetition +
    length). Lets the whole pipeline run fully offline.
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


class RealCreativityScorer(CreativityScorer):
    """Wraps the repo's ``creativity_evaluator.CreativityEvaluator`` (lazy-constructed)."""

    def __init__(self, judge=None) -> None:  # noqa: ANN001 — optional prism judge
        self._judge = judge
        self._evaluator = None

    def _ev(self):
        if self._evaluator is None:
            from creativity_evaluator.evaluator import CreativityEvaluator

            self._evaluator = CreativityEvaluator(judge=self._judge)
        return self._evaluator

    def score_single(self, text: str, prompt: str) -> float:
        return float(self._ev().evaluate_single(text, prompt).score)

    def score_collection(self, texts: list[str], prompt: str | None = None) -> float:
        return float(self._ev().evaluate(texts, prompt).score)


def make_scorer(name: str) -> CreativityScorer:
    """Factory: ``"real"`` | ``"heuristic"``. (``"fake"`` is constructed directly in tests.)"""
    name = name.lower()
    if name == "real":
        return RealCreativityScorer()
    if name == "heuristic":
        return HeuristicScorer()
    raise ValueError(f"unknown scorer {name!r}; use 'real' or 'heuristic'")
