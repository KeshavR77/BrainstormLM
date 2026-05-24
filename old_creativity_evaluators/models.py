from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StructuralMetrics:
    """Embedding-based diversity metrics (from MAS_Diversity, Chen et al. ACL 2026)."""
    vendi_score: float
    """Effective number of distinct ideas. Higher = more diverse."""
    order_parameter: float
    """Alignment around a mean direction. 0 = diverse, 1 = fully converged."""
    pairwise_cosine_distance: float
    """Mean pairwise cosine distance. Higher = more spread."""


@dataclass
class LexicalMetrics:
    """Token/n-gram diversity metrics (MAS_Diversity + shrooms-llm)."""
    self_bleu: float
    """Average BLEU of each text against the others. Lower = more lexically diverse."""
    wdistinct_3: float
    """IDF-weighted trigram distinctness across all texts. Higher = more diverse."""
    distinct_1: float
    """Unique unigrams / total unigrams across all texts. Higher = more diverse."""
    distinct_2: float
    """Unique bigrams / total bigrams across all texts. Higher = more diverse."""
    bigram_repetition: float
    """Mean within-text bigram repetition rate. Lower = less repetitive."""


@dataclass
class PrismScores:
    """
    5-dimension creativity scores for a COLLECTION (CreativityPrism, Hou et al. 2025).
    Scored 1–5 by an LLM judge evaluating the collection as a whole.
    """
    originality: float
    fluency: float
    flexibility: float
    elaboration: float
    usefulness: float
    raw_response: str = field(default="", repr=False)
    """Full text returned by the judge before JSON parsing."""

    @property
    def quality(self) -> float:
        return (self.elaboration + self.usefulness) / 2.0

    @property
    def novelty(self) -> float:
        return self.originality

    @property
    def diversity(self) -> float:
        return (self.fluency + self.flexibility) / 2.0


@dataclass
class SingleTextMetrics:
    """Within-text lexical metrics for a single response."""
    distinct_1: float
    distinct_2: float
    bigram_repetition: float


@dataclass
class SinglePrismScores:
    """
    3-dimension creativity scores for a SINGLE response.
    Fluency and flexibility are omitted — they require a collection.
    """
    originality: float
    elaboration: float
    usefulness: float
    raw_response: str = field(default="", repr=False)
    """Full text returned by the judge before JSON parsing."""

    @property
    def quality(self) -> float:
        return (self.elaboration + self.usefulness) / 2.0

    @property
    def novelty(self) -> float:
        return self.originality


@dataclass
class EvaluationResult:
    """
    Diversity evaluation for a COLLECTION of responses.

    ``score`` is the primary output — a single [0, 1] float summarising
    how diverse the collection is.  All other fields are the components
    that feed into it plus the optional Prism judge breakdown.
    """
    score: float
    """
    Composite diversity score [0, 1].
    Weighted blend: 40% Vendi (log-normalised) + 20% (1 − order_parameter)
                  + 20% wdistinct_3 + 20% distinct_2.
    """
    n_texts: int
    structural: StructuralMetrics
    lexical: LexicalMetrics
    prism: PrismScores | None

    def summary(self) -> dict:
        """Flat dict — score first, then all component breakdowns."""
        d: dict = {
            "score": self.score,
            "n_texts": self.n_texts,
            # structural
            "vendi_score": self.structural.vendi_score,
            "order_parameter": self.structural.order_parameter,
            "pairwise_cosine_distance": self.structural.pairwise_cosine_distance,
            # lexical
            "self_bleu": self.lexical.self_bleu,
            "wdistinct_3": self.lexical.wdistinct_3,
            "distinct_1": self.lexical.distinct_1,
            "distinct_2": self.lexical.distinct_2,
            "bigram_repetition": self.lexical.bigram_repetition,
        }
        if self.prism is not None:
            d.update({
                "prism_originality": self.prism.originality,
                "prism_fluency": self.prism.fluency,
                "prism_flexibility": self.prism.flexibility,
                "prism_elaboration": self.prism.elaboration,
                "prism_usefulness": self.prism.usefulness,
                "prism_quality": self.prism.quality,
                "prism_novelty": self.prism.novelty,
                "prism_diversity": self.prism.diversity,
                "prism_response": self.prism.raw_response,
            })
        return d


@dataclass
class SingleResponseResult:
    """
    Creativity evaluation for a SINGLE response against its prompt.

    ``score`` is the primary output — a single [0, 1] float summarising
    how novel/creative the response is.
    """
    score: float
    """
    Composite novelty/creativity score [0, 1].
    Without prism: 50% prompt_distance (normalised) + 30% distinct_2
                 + 20% (1 − bigram_repetition).
    With prism:    30% prompt_distance + 20% distinct_2
                 + 10% (1 − bigram_repetition) + 40% originality (normalised).
    """
    prompt_distance: float
    """Cosine distance between prompt and response embeddings [0, 2].
    Higher = response diverged more from the prompt."""
    lexical: SingleTextMetrics
    prism: SinglePrismScores | None

    def summary(self) -> dict:
        """Flat dict — score first, then all component breakdowns."""
        d: dict = {
            "score": self.score,
            "prompt_distance": self.prompt_distance,
            "distinct_1": self.lexical.distinct_1,
            "distinct_2": self.lexical.distinct_2,
            "bigram_repetition": self.lexical.bigram_repetition,
        }
        if self.prism is not None:
            d.update({
                "prism_originality": self.prism.originality,
                "prism_elaboration": self.prism.elaboration,
                "prism_usefulness": self.prism.usefulness,
                "prism_quality": self.prism.quality,
                "prism_novelty": self.prism.novelty,
                "prism_response": self.prism.raw_response,
            })
        return d


@dataclass
class RoundEvaluationResult:
    """
    Tracks collection diversity across interaction rounds to detect collapse
    (MAS_Diversity, Chen et al. ACL 2026).
    """
    per_round: list[EvaluationResult]
    collapse_detected: bool
    convergence_round: int | None
    """First round index where order_parameter > 0.7. None if no collapse."""

    def scores(self) -> list[float]:
        return [r.score for r in self.per_round]

    def order_parameters(self) -> list[float]:
        return [r.structural.order_parameter for r in self.per_round]

    def vendi_scores(self) -> list[float]:
        return [r.structural.vendi_score for r in self.per_round]
