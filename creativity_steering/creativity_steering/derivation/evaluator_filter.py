"""Evaluator-stack filtering — the verified-contrast idea in its strongest form.

Instead of a per-item heuristic, score *every* generated response with the two-stage
``creativity_evaluators`` stack (Stage 1 model-free signals — DSI, lexical/slop, perplexity,
reward — plus an optional Stage 2 LLM-as-judge ensemble), then keep only the highest-scoring
creative-framed responses and the lowest-scoring neutral-framed responses. Those clean pools
are what DiffMeans sees, so the steering vector contrasts *actually-creative* against
*actually-not-creative* text rather than just the two framings.

``creativity_evaluators`` is a sibling package at the repo root (not pip-installed in this
standalone package), so we put it on ``sys.path`` lazily. Its composite is batch-relative
(z-scored within the batch), which is why selection is by **quantile** rather than a fixed
threshold.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from creativity_steering.derivation.records import ResponseExample, VerifiedSubset


def _ensure_creativity_evaluators_importable() -> None:
    """Put the sibling ``creativity_evaluators`` package on ``sys.path`` if needed."""
    try:
        import creativity_evaluators  # noqa: F401

        return
    except ImportError:
        pass
    # This file: <repo>/creativity_steering/creativity_steering/derivation/evaluator_filter.py
    # The sibling package uses a ``src`` layout: <repo>/creativity_evaluators/src/creativity_evaluators
    repo_root = Path(__file__).resolve().parents[3]
    sibling_src = repo_root / "creativity_evaluators" / "src"
    if (sibling_src / "creativity_evaluators" / "__init__.py").exists():
        sys.path.insert(0, str(sibling_src))


def select_by_quantile(
    pool: list[ResponseExample], *, quantile: float, keep: str
) -> list[ResponseExample]:
    """Keep examples on one side of a within-pool score quantile.

    ``keep="high"`` keeps scores at/above the ``quantile``-th score; ``keep="low"`` keeps
    scores at/below it. Empty pool -> empty list.
    """
    if not pool:
        return []
    scores = np.array([ex.score for ex in pool], dtype=float)
    threshold = float(np.quantile(scores, quantile))
    if keep == "high":
        return [ex for ex in pool if ex.score >= threshold]
    if keep == "low":
        return [ex for ex in pool if ex.score <= threshold]
    raise ValueError(f"keep must be 'high' or 'low', got {keep!r}")


def select_by_threshold(
    pool: list[ResponseExample], *, threshold: float, keep: str
) -> list[ResponseExample]:
    """Keep examples on one side of an absolute score ``threshold``.

    ``keep="high"`` keeps scores ``>= threshold`` (strongly creative); ``keep="low"`` keeps
    scores ``<= threshold`` (weakly creative). Unlike ``select_by_quantile``, the cut is an
    explicit score, not a within-pool fraction — so the pool size is whatever clears the bar.
    """
    if keep == "high":
        return [ex for ex in pool if ex.score is not None and ex.score >= threshold]
    if keep == "low":
        return [ex for ex in pool if ex.score is not None and ex.score <= threshold]
    raise ValueError(f"keep must be 'high' or 'low', got {keep!r}")


def _select_one_per_prompt(
    pool: list[ResponseExample], *, quantile: float, keep: str, require_quality: bool
) -> list[ResponseExample]:
    """Take the ``quantile`` tail by **novelty**, then at most one response per prompt.

    ``keep="high"`` keeps the top ``1 - quantile`` tail (e.g. ``quantile=0.9`` → top 10%) and,
    per prompt, the single most-novel survivor; ``keep="low"`` keeps the bottom ``quantile``
    tail and the single least-novel survivor. Selection is on ``ex.novelty`` (the batch-relative
    Stage-1 novelty), not the gated composite, so the dull pole is ordered by how un-novel it is
    rather than being collapsed by the quality gate. When ``require_quality`` is set, candidates
    must have ``ex.quality_ok`` (used for the creative/positive pole so incoherent high-temp
    responses never enter the contrast). The per-prompt cap stops one topic from dominating.
    """
    novs = [ex.novelty for ex in pool if ex.novelty is not None]
    if not novs:
        return []
    threshold = float(np.quantile(np.array(novs, dtype=float), quantile))
    if keep == "high":
        cand = [ex for ex in pool if ex.novelty is not None and ex.novelty >= threshold]
    elif keep == "low":
        cand = [ex for ex in pool if ex.novelty is not None and ex.novelty <= threshold]
    else:
        raise ValueError(f"keep must be 'high' or 'low', got {keep!r}")
    if require_quality:
        cand = [ex for ex in cand if ex.quality_ok]
    best: dict[str, ResponseExample] = {}
    for ex in cand:
        cur = best.get(ex.prompt)
        if cur is None:
            best[ex.prompt] = ex
        elif keep == "high" and ex.novelty > cur.novelty:
            best[ex.prompt] = ex
        elif keep == "low" and ex.novelty < cur.novelty:
            best[ex.prompt] = ex
    return list(best.values())


def _length_match(
    positives: list[ResponseExample],
    negatives: list[ResponseExample],
    tolerance: float,
) -> tuple[list[ResponseExample], list[ResponseExample]]:
    """Greedily pair each positive with the nearest unused negative within ``±tolerance``
    (relative to the positive's token count). Unmatched positives are dropped."""
    from creativity_evaluators.utils.length import count_tokens

    neg_tokens = [count_tokens(ex.response) for ex in negatives]
    used = [False] * len(negatives)
    matched_pos: list[ResponseExample] = []
    matched_neg: list[ResponseExample] = []
    for pos in positives:
        nc = count_tokens(pos.response)
        best_j, best_diff = None, None
        for j, nt in enumerate(neg_tokens):
            if used[j]:
                continue
            diff = abs(nt - nc)
            if nc > 0 and diff <= tolerance * nc and (best_diff is None or diff < best_diff):
                best_diff, best_j = diff, j
        if best_j is None:
            continue
        used[best_j] = True
        matched_pos.append(pos)
        matched_neg.append(negatives[best_j])
    return matched_pos, matched_neg


def evaluate_and_filter(
    examples: Iterable[ResponseExample],
    *,
    eval_config=None,  # creativity_evaluators.EvalConfig; built from the kwargs if None
    scorers=None,  # creativity_evaluators.Scorers; built from eval_config if None (real run)
    creative_quantile: float = 0.8,
    neutral_quantile: float = 0.2,
    creative_score_min: float | None = None,
    neutral_score_max: float | None = None,
    split_by: str = "framing",
    enable_judge: bool = True,
    generator_family: str | None = "qwen",
    stage1_floor_percentile: float = 20.0,
    length_match: bool = False,
    length_match_tolerance: float = 0.10,
    enable_reward: bool = False,
    use_reward_fallback: bool = False,
    stage1_device: str | None = None,
    min_per_class: int = 0,
    one_per_prompt: bool = False,
    quality_floor: float | None = None,
) -> VerifiedSubset:
    """Score every example with the two-stage stack, then keep top-creative / bottom-neutral.

    Mutates each example's ``.score`` (gated composite), ``.novelty`` (batch-relative Stage-1
    novelty), and ``.quality_ok`` (raw judge pragmaticality cleared ``quality_floor``) in place
    as provenance. Raises ``ValueError`` if ``min_per_class`` is set and a kept pool falls short.

    When ``one_per_prompt`` is set (the v3 framed path), each pool is the top/bottom novelty tail
    (``creative_quantile`` / ``neutral_quantile``) capped at one response per prompt, with the
    positive pole additionally gated on ``quality_floor`` — see ``_select_one_per_prompt``.
    """
    _ensure_creativity_evaluators_importable()
    from creativity_evaluators import EvalConfig, PromptCategory
    from creativity_evaluators.pipeline import evaluate_batch

    examples = list(examples)
    if not examples:
        return VerifiedSubset(positives=[], negatives=[], n_dropped=0)

    if eval_config is None:
        eval_config = EvalConfig(
            enable_stage2=enable_judge,
            generator_family=generator_family,
            stage1_percentile_floor=stage1_floor_percentile,
            run_reward=enable_reward,
            use_reward_fallback=use_reward_fallback,
            stage1_device=stage1_device,
            composite_quality_floor=quality_floor,
        )

    records = [
        {
            "prompt": ex.prompt,
            "response": ex.response,
            "category": (
                PromptCategory.CREATIVE if ex.framing == "creative" else PromptCategory.NEUTRAL
            ),
        }
        for ex in examples
    ]
    results = evaluate_batch(records, eval_config, scorers=scorers)
    for ex, result in zip(examples, results):
        ex.score = float(result.creativity_score)
        ex.novelty = float(result.components.get("stage1_composite", result.creativity_score))
        prag = result.ensemble.per_dimension_mean.get("Pragmaticality") if result.ensemble else None
        ex.quality_ok = (
            (prag is not None and float(prag) >= quality_floor)
            if quality_floor is not None
            else None
        )
    import numpy as _np
    all_scores = _np.array([ex.score for ex in examples], dtype=float)

    def _pct(q: float) -> float:
        return float(_np.percentile(all_scores, q))

    print(f"[evaluator_filter] composite score distribution (n={len(all_scores)}): "
          f"min={all_scores.min():.3f} p10={_pct(10):.3f} p25={_pct(25):.3f} "
          f"p50={_pct(50):.3f} p75={_pct(75):.3f} p90={_pct(90):.3f} max={all_scores.max():.3f}",
          flush=True)

    if one_per_prompt and split_by != "score":
        # v3 framed path: top/bottom novelty tail, capped at one response per prompt; the positive
        # pole is additionally quality-gated. Selecting on novelty (not the gated composite) keeps
        # the dull pole ordered by un-novelty rather than collapsed onto the quality penalty.
        creatives = [ex for ex in examples if ex.framing == "creative"]
        neutrals = [ex for ex in examples if ex.framing == "neutral"]
        nov = _np.array([ex.novelty for ex in examples if ex.novelty is not None], dtype=float)
        if nov.size:
            print(f"[evaluator_filter] novelty distribution (n={nov.size}): "
                  f"p10={float(_np.percentile(nov,10)):.3f} p50={float(_np.percentile(nov,50)):.3f} "
                  f"p90={float(_np.percentile(nov,90)):.3f}", flush=True)
        n_quality_ok = sum(1 for ex in creatives if ex.quality_ok)
        positives = _select_one_per_prompt(
            creatives, quantile=creative_quantile, keep="high",
            require_quality=quality_floor is not None,
        )
        negatives = _select_one_per_prompt(
            neutrals, quantile=neutral_quantile, keep="low", require_quality=False,
        )
        print(f"[evaluator_filter] one_per_prompt: {len(positives)} positives "
              f"(of {len(creatives)} creative, {n_quality_ok} quality-ok) / {len(negatives)} "
              f"negatives (of {len(neutrals)} neutral); quality_floor={quality_floor}", flush=True)
    elif split_by == "score":
        # Single-context: ignore framing entirely; split the whole pool.
        # Absolute-score thresholds when both are set; else quantile cuts on the unified pool
        # (creative_quantile = lower edge of "creative" tail, neutral_quantile = upper edge of
        # "dull" tail). Quantile mode gives a deterministic count proportional to pool size and
        # a guaranteed separating band, which matters when the score distribution is skewed and
        # absolute thresholds end up landing in noise (cf. v2's 0.05/0.00 collapse).
        if creative_score_min is not None and neutral_score_max is not None:
            positives = select_by_threshold(examples, threshold=creative_score_min, keep="high")
            negatives = select_by_threshold(examples, threshold=neutral_score_max, keep="low")
        else:
            positives = select_by_quantile(examples, quantile=creative_quantile, keep="high")
            negatives = select_by_quantile(examples, quantile=neutral_quantile, keep="low")
    else:
        creatives = [ex for ex in examples if ex.framing == "creative"]
        neutrals = [ex for ex in examples if ex.framing == "neutral"]
        # Explicit absolute-score selection when both thresholds are set; else quantile.
        if creative_score_min is not None and neutral_score_max is not None:
            positives = select_by_threshold(creatives, threshold=creative_score_min, keep="high")
            negatives = select_by_threshold(neutrals, threshold=neutral_score_max, keep="low")
        else:
            positives = select_by_quantile(creatives, quantile=creative_quantile, keep="high")
            negatives = select_by_quantile(neutrals, quantile=neutral_quantile, keep="low")

    print(f"[evaluator_filter] post-threshold: {len(positives)} positives / {len(negatives)} negatives (of {len(examples)} examples)", flush=True)
    if length_match:
        positives, negatives = _length_match(positives, negatives, length_match_tolerance)
        print(f"[evaluator_filter] post-length-match: {len(positives)} positives / {len(negatives)} negatives (tolerance={length_match_tolerance})", flush=True)

    dropped = len(examples) - len(positives) - len(negatives)

    if min_per_class > 0 and (len(positives) < min_per_class or len(negatives) < min_per_class):
        if creative_score_min is not None and neutral_score_max is not None:
            cut = (
                f"the score thresholds (creative_score_min={creative_score_min}, "
                f"neutral_score_max={neutral_score_max})"
            )
        else:
            cut = (
                f"the quantiles (creative_quantile={creative_quantile}, "
                f"neutral_quantile={neutral_quantile})"
            )
        raise ValueError(
            f"verified subset too small: {len(positives)} positives / {len(negatives)} "
            f"negatives < min_per_class={min_per_class}. Gather more prompts/samples or relax "
            f"{cut}."
        )

    return VerifiedSubset(positives=positives, negatives=negatives, n_dropped=dropped)
