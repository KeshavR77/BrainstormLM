"""Conditional-Vendi — diversity conditioned on the prompt (arXiv:2411.02817).

Pooling every response into one unconditional Vendi score inflates diversity
just because the prompts differ — "prompt echo". Conditional-Vendi instead
averages the *within-prompt* Vendi score across prompts, isolating how much
genuine variety the model produces per prompt. Higher = more genuinely diverse
generations; bounded below by 1.0 (each non-empty group contributes >= 1).

Reuses :func:`brainstormlm.eval.diversity.vendi_score` per group, so the same
embedding kernel and caching apply.
"""

from __future__ import annotations

from typing import Sequence

from brainstormlm.eval.diversity import vendi_score


def conditional_vendi(per_prompt_texts: Sequence[Sequence[str]]) -> dict[str, float]:
    """Mean within-prompt Vendi score across prompt groups.

    Returns ``{mean_within_vendi, per_prompt, n_prompts}``. Empty input → zeros.
    """
    groups = [list(g) for g in per_prompt_texts]
    if not groups:
        return {"mean_within_vendi": 0.0, "per_prompt": [], "n_prompts": 0}
    per_prompt = [float(vendi_score(g)) for g in groups]
    mean = sum(per_prompt) / len(per_prompt)
    return {
        "mean_within_vendi": float(mean),
        "per_prompt": per_prompt,
        "n_prompts": len(groups),
    }
