"""Self-BLEU diversity metric (Zhu et al., 'Texygen', 2018).

For a set of N model outputs to the same prompt, Self-BLEU treats each output
in turn as the hypothesis and the remaining N-1 as references, then averages
the resulting BLEU-n scores. A high Self-BLEU means the outputs share many
n-grams (mode collapse). A low Self-BLEU means outputs are lexically diverse.

This is the standard complementary lexical metric to the embedding-based
Vendi score already in ``brainstormlm.eval.diversity``.

Reference: Zhu et al., "Texygen: A Benchmarking Platform for Text Generation
Models", SIGIR 2018.
"""

from __future__ import annotations

from typing import Sequence


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def self_bleu(texts: Sequence[str], *, n: int = 4) -> float:
    """Mean BLEU-n of each text against the other texts as references.

    Args:
      texts: a set of outputs generated for the same prompt.
      n: max n-gram order for BLEU (default 4, the standard).

    Returns:
      A float in [0, 1]. Higher = more mode collapse.
    """
    if len(texts) < 2:
        return 0.0

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    tokenized = [_tokenize(t) for t in texts]
    weights = tuple([1.0 / n] * n)
    smooth = SmoothingFunction().method1

    scores: list[float] = []
    for i, hyp in enumerate(tokenized):
        if not hyp:
            continue
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i and tokenized[j]]
        if not refs:
            continue
        scores.append(sentence_bleu(refs, hyp, weights=weights, smoothing_function=smooth))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
