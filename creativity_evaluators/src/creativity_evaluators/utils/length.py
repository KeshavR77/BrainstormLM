"""Token counting and length-matched contrast partitioning.

Length-matching is **critical hygiene** for the DiffMeans use case: if the creative and
not-creative sets differ systematically in length, the steering vector ends up encoding
verbosity rather than creativity. ``length_matched_partition`` takes the top-quintile
creative responses and bottom-quintile neutral responses and greedily matches them by token
count, dropping anything it cannot pair within tolerance.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np

from ..types import EvalResult

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str, tokenizer: Callable[[str], list] | None = None) -> int:
    """Count tokens. Default is a simple word+punctuation regex; pass the *generator's*
    tokenizer (any ``str -> list``) for an exact match to the model that produced the text."""
    if tokenizer is not None:
        return len(tokenizer(text))
    return len(_TOKEN.findall(text))


def _select_quintile(results: list[EvalResult], quintile: int) -> list[EvalResult]:
    """Return the members of ``results`` whose creativity_score falls in the given quintile
    (1 = bottom 20%, 5 = top 20%)."""
    if not results:
        return []
    scores = np.array([r.creativity_score for r in results], dtype=float)
    lo = float(np.percentile(scores, (quintile - 1) * 20))
    hi = float(np.percentile(scores, quintile * 20))
    if quintile == 5:
        return [r for r in results if r.creativity_score >= lo]
    if quintile == 1:
        return [r for r in results if r.creativity_score <= hi]
    return [r for r in results if lo <= r.creativity_score <= hi]


def length_matched_partition(
    creative_pool: list[EvalResult],
    neutral_pool: list[EvalResult],
    tolerance: float = 0.10,
    creative_quintile: int = 5,
    neutral_quintile: int = 1,
) -> tuple[list[EvalResult], list[EvalResult]]:
    """Return ``(creative_subset, neutral_subset)``, length-matched within ``tolerance``.

    ``creative_subset`` is the top-quintile-by-creativity creative responses; ``neutral_subset``
    is the bottom-quintile neutral responses. Greedy matching: for each creative (highest score
    first) find the nearest unused neutral whose token count is within ``±tolerance`` (relative
    to the creative's token count). Unmatched creatives are dropped; the two returned lists are
    equal length. The number of drops is logged.
    """
    creatives = sorted(
        _select_quintile(creative_pool, creative_quintile),
        key=lambda r: r.creativity_score,
        reverse=True,
    )
    neutrals = _select_quintile(neutral_pool, neutral_quintile)
    used = [False] * len(neutrals)
    matched_c: list[EvalResult] = []
    matched_n: list[EvalResult] = []
    drops = 0

    for c in creatives:
        nc = c.n_tokens
        best_j: int | None = None
        best_diff: float | None = None
        for j, n in enumerate(neutrals):
            if used[j]:
                continue
            diff = abs(n.n_tokens - nc)
            if nc > 0 and diff <= tolerance * nc and (best_diff is None or diff < best_diff):
                best_diff, best_j = diff, j
        if best_j is None:
            drops += 1
            continue
        used[best_j] = True
        matched_c.append(c)
        matched_n.append(neutrals[best_j])

    logger.info(
        "length_matched_partition: %d pairs matched, %d creatives dropped, %d neutrals unused",
        len(matched_c),
        drops,
        used.count(False),
    )
    return matched_c, matched_n
