"""Multi-judge ensemble.

Per the MDPI Applied Sciences (2025) consensus result, a 3-judge ensemble reaches ~97%
macro-F1 vs ~80% for any single judge. The default ensemble is
``[AnthropicJudge, OpenAIJudge, GoogleJudge]`` (configurable).

Bias controls reflected here:
  * **Judge-family exclusion.** A judge is dropped when its ``family`` matches the family of
    the model that *generated* the responses, so a model never grades its own siblings. For
    our DiffMeans use case the generator is a Qwen model, so by default no frontier judge is
    excluded; the guard matters when we later test on frontier-model outputs. Pass
    ``generator_family`` to activate it.
  * **Position-swap.** Only meaningful for *pairwise* judging. We do single-response scoring,
    which has no positions, so there is nothing to swap — and mean aggregation across judges
    is order-invariant, which is the corresponding bias control here.

Aggregation: mean across judges per dimension, then mean across dimensions for the overall
scalar. Inter-judge agreement (mean pairwise Spearman across the judges' 5-dim score vectors)
is reported as a health check; a value below ``warn_threshold`` is logged as a warning.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

from ..types import DIMENSIONS, EnsembleResult, JudgeResult
from .judge import Judge

logger = logging.getLogger(__name__)


def exclude_same_family(judges: Sequence[Judge], generator_family: str | None) -> list[Judge]:
    """Drop judges whose ``family`` equals ``generator_family`` (no-op when it is None)."""
    if not generator_family:
        return list(judges)
    return [j for j in judges if j.family != generator_family]


def aggregate_scores(results: Sequence[JudgeResult]) -> tuple[dict[str, float], float]:
    """Mean per dimension across judges, then mean of those for the overall scalar."""
    per_dimension_mean = {
        dim: float(np.mean([r.scores[dim] for r in results])) for dim in DIMENSIONS
    }
    overall = float(np.mean(list(per_dimension_mean.values())))
    return per_dimension_mean, overall


def mean_pairwise_spearman(results: Sequence[JudgeResult]) -> float:
    """Mean pairwise Spearman across judges' 5-dim score vectors.

    Returns NaN with fewer than two judges, or when every pair is undefined (e.g. a judge
    gives a constant vector, which has no rank variance).
    """
    vectors = [[r.scores[dim] for dim in DIMENSIONS] for r in results]
    if len(vectors) < 2:
        return math.nan
    corrs: list[float] = []
    with warnings.catch_warnings():
        # A constant score vector has no rank variance -> NaN, which we filter below.
        warnings.simplefilter("ignore", ConstantInputWarning)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                rho, _ = spearmanr(vectors[i], vectors[j])
                if rho is not None and math.isfinite(rho):
                    corrs.append(float(rho))
    return float(np.mean(corrs)) if corrs else math.nan


class Ensemble:
    """Runs a set of judges over one (prompt, response) and aggregates their verdicts."""

    def __init__(
        self,
        judges: Sequence[Judge],
        *,
        generator_family: str | None = None,
        warn_threshold: float = 0.5,
    ):
        self.judges = exclude_same_family(judges, generator_family)
        self.warn_threshold = warn_threshold

    def score(self, prompt: str, response: str) -> EnsembleResult:
        per_judge: list[JudgeResult] = []
        for judge in self.judges:
            try:
                per_judge.append(judge.score(prompt, response))
            except Exception as exc:  # one judge failing must not sink the ensemble
                # ERROR, not WARNING: a dropped judge silently degrades the run to a smaller
                # ensemble (the "Claude vanished after rate-limiting" case), so make it loud.
                logger.error(
                    "judge %s failed and was skipped: %s", getattr(judge, "name", "?"), exc
                )

        if not per_judge:
            return EnsembleResult(
                per_judge=[],
                per_dimension_mean={dim: math.nan for dim in DIMENSIONS},
                overall=math.nan,
                inter_judge_spearman=math.nan,
                n_judges_succeeded=0,
            )

        per_dimension_mean, overall = aggregate_scores(per_judge)
        agreement = mean_pairwise_spearman(per_judge)
        if math.isfinite(agreement) and agreement < self.warn_threshold:
            logger.warning(
                "low inter-judge agreement (Spearman=%.3f < %.2f); distrust this score",
                agreement,
                self.warn_threshold,
            )
        return EnsembleResult(
            per_judge=per_judge,
            per_dimension_mean=per_dimension_mean,
            overall=overall,
            inter_judge_spearman=agreement,
            n_judges_succeeded=len(per_judge),
        )
