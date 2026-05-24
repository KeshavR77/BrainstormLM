"""
CreativityEvaluator — main entry point.

Two evaluation modes:
  evaluate(texts, prompt)         — collection: diversity across N responses
  evaluate_single(text, prompt)   — single: novelty/creativity of one response
  evaluate_rounds(rounds, prompt) — round tracking: detect diversity collapse over time

Both modes return an object where .score is the primary [0, 1] output and
.summary() gives the full flat breakdown as a dict.
"""

from __future__ import annotations
import math

import numpy as np

from .embeddings.base import BaseEmbedder
from .embeddings.local_embedder import LocalEmbedder
from .metrics.structural import vendi_score, order_parameter, pairwise_cosine_distance
from .metrics.lexical import (
    self_bleu, wdistinct_n, distinct_n, mean_bigram_repetition, bigram_repetition,
)
from .models import (
    EvaluationResult,
    LexicalMetrics,
    RoundEvaluationResult,
    SingleResponseResult,
    SingleTextMetrics,
    StructuralMetrics,
)
from .prism.judge import BaseJudge

_COLLAPSE_THRESHOLD = 0.7


class CreativityEvaluator:
    """
    Two evaluation modes — both return an object where .score is the primary output:

    1. COLLECTION — diversity across multiple responses to the same prompt:
        evaluator = CreativityEvaluator()
        result = evaluator.evaluate(["idea A", "idea B", "idea C"])
        result.score          # [0, 1] diversity score
        result.summary()      # full flat dict with all components

    2. SINGLE — novelty/creativity of one response vs its prompt:
        result = evaluator.evaluate_single("my response", prompt="Generate startup ideas")
        result.score          # [0, 1] novelty score
        result.summary()      # full flat dict with all components

    Add a judge for LLM-scored dimensions:
        from creativity_evaluator.prism import ClaudeJudge
        evaluator = CreativityEvaluator(judge=ClaudeJudge())
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        judge: BaseJudge | None = None,
    ):
        self._embedder = embedder or LocalEmbedder()
        self._judge = judge

    def embed(self, texts: list[str]) -> np.ndarray:
        """Sentence embeddings (L2-normalized), same pipeline as ``evaluate`` / ``evaluate_single``."""
        return self._embedder.embed(texts)

    # ------------------------------------------------------------------
    # Collection mode
    # ------------------------------------------------------------------

    def evaluate(
        self,
        texts: list[str],
        context_prompt: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate diversity across a collection of responses.

        score formula (no prism):
          40% Vendi (log-normalised to [0,1]) + 20% (1 − order_parameter)
          + 20% wdistinct_3 + 20% distinct_2

        score formula (with prism):
          30% Vendi + 15% (1 − order_parameter) + 15% wdistinct_3
          + 10% distinct_2 + 30% prism.diversity (fluency+flexibility, normalised)
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        emb = self._embedder.embed(texts)

        structural = StructuralMetrics(
            vendi_score=vendi_score(emb),
            order_parameter=order_parameter(emb),
            pairwise_cosine_distance=pairwise_cosine_distance(emb),
        )
        lexical = LexicalMetrics(
            self_bleu=self_bleu(texts),
            wdistinct_3=wdistinct_n(texts, n=3),
            distinct_1=distinct_n(texts, n=1),
            distinct_2=distinct_n(texts, n=2),
            bigram_repetition=mean_bigram_repetition(texts),
        )

        prism = self._judge.score(texts, context_prompt) if self._judge else None
        score = _collection_score(structural, lexical, prism)

        return EvaluationResult(
            score=score,
            n_texts=len(texts),
            structural=structural,
            lexical=lexical,
            prism=prism,
        )

    # ------------------------------------------------------------------
    # Single-response mode
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        text: str,
        prompt: str,
    ) -> SingleResponseResult:
        """
        Evaluate novelty/creativity of a single response against its prompt.

        score formula (no prism):
          50% prompt_distance (normalised to [0,1]) + 30% distinct_2
          + 20% (1 − bigram_repetition)

        score formula (with prism):
          30% prompt_distance + 20% distinct_2 + 10% (1 − bigram_repetition)
          + 40% prism.originality (normalised)
        """
        if not text:
            raise ValueError("text must be non-empty")
        if not prompt:
            raise ValueError("prompt must be non-empty")

        emb = self._embedder.embed([prompt, text])
        dist = float(1.0 - np.dot(emb[0], emb[1]))  # cosine distance, embs are L2-normalised

        lexical = SingleTextMetrics(
            distinct_1=distinct_n([text], n=1),
            distinct_2=distinct_n([text], n=2),
            bigram_repetition=bigram_repetition(text),
        )

        prism = self._judge.score_single(text, prompt) if self._judge else None
        score = _single_score(dist, lexical, prism)

        return SingleResponseResult(
            score=score,
            prompt_distance=dist,
            lexical=lexical,
            prism=prism,
        )

    # ------------------------------------------------------------------
    # Round-tracking mode
    # ------------------------------------------------------------------

    def evaluate_rounds(
        self,
        rounds: list[list[str]],
        context_prompt: str | None = None,
    ) -> RoundEvaluationResult:
        """Evaluate a collection per round and detect diversity collapse."""
        if not rounds:
            raise ValueError("rounds must be non-empty")

        per_round = [self.evaluate(r, context_prompt) for r in rounds]

        collapse_detected = False
        convergence_round: int | None = None
        for i, result in enumerate(per_round[1:], start=1):
            if result.structural.order_parameter > _COLLAPSE_THRESHOLD:
                collapse_detected = True
                convergence_round = i
                break

        return RoundEvaluationResult(
            per_round=per_round,
            collapse_detected=collapse_detected,
            convergence_round=convergence_round,
        )


# ------------------------------------------------------------------
# Score formulas
# ------------------------------------------------------------------

def _collection_score(
    structural: StructuralMetrics,
    lexical: LexicalMetrics,
    prism,
) -> float:
    vendi_norm = min(math.log(max(structural.vendi_score, 1.0)) / math.log(10), 1.0)
    order_div = max(0.0, min(1.0, 1.0 - structural.order_parameter))

    if prism is None:
        return (
            0.40 * vendi_norm
            + 0.20 * order_div
            + 0.20 * lexical.wdistinct_3
            + 0.20 * lexical.distinct_2
        )
    # prism.diversity = mean(fluency, flexibility), both on 1–5 → normalise to [0,1]
    prism_div_norm = (prism.diversity - 1.0) / 4.0
    return (
        0.30 * vendi_norm
        + 0.15 * order_div
        + 0.15 * lexical.wdistinct_3
        + 0.10 * lexical.distinct_2
        + 0.30 * prism_div_norm
    )


def _single_score(
    prompt_distance: float,
    lexical: SingleTextMetrics,
    prism,
) -> float:
    dist_norm = min(prompt_distance / 2.0, 1.0)  # cosine distance range is [0, 2]
    novelty_lex = max(0.0, min(1.0, 1.0 - lexical.bigram_repetition))

    if prism is None:
        return (
            0.50 * dist_norm
            + 0.30 * lexical.distinct_2
            + 0.20 * novelty_lex
        )
    # prism.originality on 1–5 → normalise to [0, 1]
    orig_norm = (prism.originality - 1.0) / 4.0
    return (
        0.30 * dist_norm
        + 0.20 * lexical.distinct_2
        + 0.10 * novelty_lex
        + 0.40 * orig_norm
    )
