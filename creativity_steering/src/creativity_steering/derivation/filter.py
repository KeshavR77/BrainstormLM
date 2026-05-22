"""Evaluator-based filtering — turn raw responses into a *verified* contrast subset.

The project's core idea: don't apply DiffMeans to whatever the creative/neutral framings
happen to produce, but to responses the creativity evaluator actually *verifies*. A
creative-framed response only becomes a positive if it scores creative enough; a
neutral-framed response only becomes a negative if it scores un-creative enough.
Everything ambiguous is dropped, so the two pools are cleanly separated in creativity.
"""

from __future__ import annotations

from collections.abc import Iterable

from creativity_steering.derivation.records import ResponseExample, VerifiedSubset
from creativity_steering.scoring import CreativityScorer


def label_and_filter(
    examples: Iterable[ResponseExample],
    scorer: CreativityScorer,
    *,
    creative_min: float,
    neutral_max: float,
    min_per_class: int = 0,
) -> VerifiedSubset:
    """Score every example, then keep only confidently-creative / confidently-not pools.

    Mutates each example's ``.score`` in place (it is useful provenance to persist).
    Raises ``ValueError`` if ``min_per_class`` is set and a pool falls short — the guard
    against deriving DiffMeans on too little data.
    """
    if not 0.0 <= neutral_max <= creative_min <= 1.0:
        raise ValueError(
            f"need 0 <= neutral_max ({neutral_max}) <= creative_min ({creative_min}) <= 1"
        )

    positives: list[ResponseExample] = []
    negatives: list[ResponseExample] = []
    dropped = 0

    for ex in examples:
        ex.score = scorer.score_single(ex.response, ex.prompt)
        if ex.framing == "creative" and ex.score >= creative_min:
            positives.append(ex)
        elif ex.framing == "neutral" and ex.score <= neutral_max:
            negatives.append(ex)
        else:
            dropped += 1

    if min_per_class > 0 and (len(positives) < min_per_class or len(negatives) < min_per_class):
        raise ValueError(
            f"verified subset too small: {len(positives)} positives / {len(negatives)} "
            f"negatives < min_per_class={min_per_class}. Gather more prompts/samples or "
            f"relax thresholds (creative_min={creative_min}, neutral_max={neutral_max})."
        )

    return VerifiedSubset(positives=positives, negatives=negatives, n_dropped=dropped)
