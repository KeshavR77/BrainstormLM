"""Lexical / syntactic redundancy.

Two sub-scores, both measuring *redundancy* (so the composite subtracts the result —
higher == more boilerplate == less creative):

(a) Syntactic-template overlap (Shaib et al. 2024): fraction of the response's POS 4-grams
    that match a precomputed set of high-frequency LLM templates. Needs spaCy; degrades
    gracefully to 0.0 with ``template_overlap_available=False`` when spaCy is absent.

(b) Slop overlap (EQ-Bench v3 style): rate of known GPT-isms per token (see
    ``stage2.slop_list``). This is the more reliable signal, so it is weighted 2x.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..stage2.slop_list import SLOP_PHRASES
from ..types import LexicalResult

_TEMPLATE_FILE = Path(__file__).parent / "data" / "pos_templates.txt"
_WORD = re.compile(r"\w+")


@lru_cache(maxsize=1)
def load_pos_templates() -> frozenset[tuple[str, ...]]:
    """Load the static POS 4-gram template set (cached). Lines starting with # are ignored."""
    templates: set[tuple[str, ...]] = set()
    for line in _TEMPLATE_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        templates.add(tuple(line.split()))
    return frozenset(templates)


DEFAULT_POS_TEMPLATES = load_pos_templates()


@lru_cache(maxsize=512)
def _phrase_regex(phrase: str) -> re.Pattern[str]:
    # Escape, then let any run of whitespace match, and require word boundaries so
    # "realm" does not fire inside "overwhelmed".
    body = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def slop_rate(text: str, phrases: list[str]) -> tuple[float, list[str]]:
    """Return ``(occurrences / n_word_tokens, [matched_phrase per occurrence])``.

    Counts non-overlapping occurrences of each phrase. ``n_word_tokens`` is the number of
    ``\\w+`` tokens; an empty / token-less response yields ``(0.0, [])``.
    """
    n_tokens = len(_WORD.findall(text))
    if n_tokens == 0:
        return 0.0, []
    hits: list[str] = []
    for phrase in phrases:
        n = len(_phrase_regex(phrase).findall(text))
        hits.extend([phrase.lower()] * n)
    return len(hits) / n_tokens, hits


def pos_template_overlap(
    text: str,
    templates: frozenset[tuple[str, ...]],
    nlp=None,
) -> tuple[float, bool]:
    """Fraction of POS 4-grams in ``text`` that match ``templates``.

    Returns ``(overlap, available)``. ``available`` is False (and overlap 0.0) when spaCy or
    its ``en_core_web_sm`` model cannot be loaded, so callers degrade gracefully.
    """
    if nlp is None:
        try:
            import spacy  # type: ignore

            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            return 0.0, False
    pos = [tok.pos_ for tok in nlp(text)]
    if len(pos) < 4:
        return 0.0, True
    grams = [tuple(pos[i : i + 4]) for i in range(len(pos) - 3)]
    matches = sum(1 for g in grams if g in templates)
    return matches / len(grams), True


class LexicalScorer:
    """Combines slop overlap (weighted 2x) and POS-template overlap into one redundancy score.

    ``combined`` is the weighted *average* ``(template_weight·overlap + slop_weight·rate) /
    (template_weight + slop_weight)`` so it stays on a comparable scale regardless of which
    signals are available; the composite z-scores it within the batch anyway.
    """

    def __init__(
        self,
        slop_phrases: list[str] | None = None,
        templates: frozenset[tuple[str, ...]] | None = None,
        spacy_model: str = "en_core_web_sm",
        slop_weight: float = 2.0,
        template_weight: float = 1.0,
        nlp=None,
    ):
        self.slop_phrases = slop_phrases if slop_phrases is not None else SLOP_PHRASES
        self.templates = templates if templates is not None else DEFAULT_POS_TEMPLATES
        self.spacy_model = spacy_model
        self.slop_weight = slop_weight
        self.template_weight = template_weight
        self._nlp = nlp

    def score(self, text: str) -> LexicalResult:
        rate, hits = slop_rate(text, self.slop_phrases)
        overlap, available = pos_template_overlap(text, self.templates, nlp=self._nlp)
        denom = self.template_weight + self.slop_weight
        combined = (self.template_weight * overlap + self.slop_weight * rate) / denom
        return LexicalResult(
            template_overlap=overlap,
            slop_rate=rate,
            combined=combined,
            slop_hits=hits,
            template_overlap_available=available,
        )
