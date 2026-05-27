"""Top-level pipeline: Stage 1 -> Stage-2 gating -> Stage 2 -> composite -> EvalResult.

``evaluate_batch`` is the real entry point because the composite z-scores every signal
against the batch. ``evaluate`` is a thin single-record wrapper; note that with a batch of
one, every z-score is 0 by construction, so a single-record creativity_score is degenerate —
use a batch for meaningful relative scores.

Stage 1 runs sequentially (GPU-bound model inference). Stage 2 runs across the passing
records in a thread pool (``n_workers``), since judge calls are network-bound; the shared
disk cache makes repeats free.
"""

from __future__ import annotations

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .composite import compute_composite_batch, stage1_composite_batch
from .stage1.dsi import DSIScorer
from .stage1.lexical import LexicalScorer
from .stage1.perplexity import PerplexityScorer
from .stage1.reward_model import RewardModelScorer
from .stage2.ensemble import Ensemble
from .stage2.judge import PROVIDER_REGISTRY
from .types import EvalConfig, EvalResult, LexicalResult, PromptCategory
from .utils.caching import DiskCache
from .utils.length import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class Scorers:
    """The constructed scorer objects. Build from config via ``build_scorers`` or inject
    (e.g. in tests / for custom backends)."""

    dsi: DSIScorer | None
    lexical: LexicalScorer | None
    ppl: PerplexityScorer | None
    reward: RewardModelScorer | None
    ensemble: Ensemble | None


def build_scorers(config: EvalConfig) -> Scorers:
    """Construct the real scorers (lazy models / API judges) from an ``EvalConfig``."""
    cache = DiskCache(config.cache_dir)
    dsi = (
        DSIScorer(
            model_name=config.dsi_model,
            layers=tuple(config.dsi_layers),
            min_tokens=config.dsi_min_tokens,
            chunk_size=config.dsi_chunk_size,
            device=config.stage1_device,
        )
        if config.run_dsi
        else None
    )
    lexical = LexicalScorer() if config.run_lexical else None
    ppl = (
        PerplexityScorer(model_name=config.ppl_model, device=config.stage1_device)
        if config.run_ppl
        else None
    )
    reward = (
        RewardModelScorer(
            model_name=config.reward_model,
            fallback_model=config.reward_model_fallback,
            use_fallback=config.use_reward_fallback,
            device=config.stage1_device,
            cache=cache,
        )
        if config.run_reward
        else None
    )
    ensemble = None
    if config.enable_stage2:
        judges = [
            PROVIDER_REGISTRY[spec.provider](
                model=spec.model,
                name=spec.name,
                cache=cache,
                temperature=config.judge_temperature,
                max_retries=config.judge_max_retries,
                truncate_chars=config.judge_truncate_chars,
                timeout=config.judge_timeout,
            )
            for spec in config.judges
        ]
        ensemble = Ensemble(
            judges,
            generator_family=config.generator_family,
            warn_threshold=config.inter_judge_warn_threshold,
        )
    return Scorers(dsi=dsi, lexical=lexical, ppl=ppl, reward=reward, ensemble=ensemble)


def _coerce_category(value) -> PromptCategory:
    if isinstance(value, PromptCategory):
        return value
    return PromptCategory(str(value).lower())


def evaluate(
    prompt: str,
    response: str,
    category: PromptCategory,
    config: EvalConfig,
    *,
    scorers: Scorers | None = None,
) -> EvalResult:
    """Evaluate one (prompt, response). See module note on single-record degeneracy."""
    record = {"prompt": prompt, "response": response, "category": category}
    return evaluate_batch([record], config, n_workers=1, scorers=scorers)[0]


def evaluate_batch(
    records: list[dict],
    config: EvalConfig,
    n_workers: int = 4,
    *,
    scorers: Scorers | None = None,
) -> list[EvalResult]:
    """Score a batch of ``{prompt, response, category}`` records into ``EvalResult``s."""
    if not records:
        return []
    if scorers is None:
        scorers = build_scorers(config)

    n = len(records)
    prompts = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]
    categories = [_coerce_category(r["category"]) for r in records]

    # --- Stage 1 (sequential) ---
    dsi_results = []
    lex_results: list[LexicalResult] = []
    ppl_results = []
    reward_vals: list[float] = []
    n_tokens_list: list[int] = []
    for prompt, response in zip(prompts, responses):
        n_tokens_list.append(count_tokens(response))
        dsi_results.append(scorers.dsi.score(response) if scorers.dsi else None)
        lex_results.append(
            scorers.lexical.score(response) if scorers.lexical else LexicalResult(0.0, 0.0, 0.0)
        )
        ppl_results.append(scorers.ppl.score(response) if scorers.ppl else None)
        reward_vals.append(scorers.reward.score(prompt, response) if scorers.reward else 0.0)

    def _dsi(i):
        return dsi_results[i].score if dsi_results[i] is not None else None

    def _ppl(i):
        return ppl_results[i].nll_per_token if ppl_results[i] is not None else None

    # --- Stage-2 gating on the Stage-1-only composite ---
    s1_records = [
        {
            "dsi": _dsi(i),
            "ppl": _ppl(i),
            "lexical_redundancy": lex_results[i].combined,
            "reward": reward_vals[i],
        }
        for i in range(n)
    ]
    s1_composite = stage1_composite_batch(s1_records)

    run_s2 = [False] * n
    if config.enable_stage2 and scorers.ensemble is not None:
        floor = float(np.percentile(s1_composite, config.stage1_percentile_floor))
        run_s2 = [s1_composite[i] >= floor for i in range(n)]

    # --- Stage 2 (parallel over passing records) ---
    ensembles: list = [None] * n
    pass_idx = [i for i in range(n) if run_s2[i]]
    if pass_idx:

        def _run(i):
            return scorers.ensemble.score(prompts[i], responses[i])

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
            for i, ens in zip(pass_idx, ex.map(_run, pass_idx)):
                ensembles[i] = ens

    # --- Composite ---
    full_records = []
    for i in range(n):
        ens = ensembles[i]
        full_records.append(
            {
                "dsi": _dsi(i),
                "ppl": _ppl(i),
                "lexical_redundancy": lex_results[i].combined,
                "reward": reward_vals[i],
                "originality_s2": ens.per_dimension_mean["Originality"] if ens else None,
                "pragmaticality_s2": ens.per_dimension_mean["Pragmaticality"] if ens else None,
                "elaboration_s2": ens.per_dimension_mean["Elaboration"] if ens else None,
            }
        )
    finals, components = compute_composite_batch(
        full_records,
        creativity_weight=config.composite_creativity_weight,
        elaboration_weight=config.composite_elaboration_weight,
    )

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    config_hash = config.config_hash

    results: list[EvalResult] = []
    for i in range(n):
        comp = components[i]
        comp["stage1_composite"] = float(s1_composite[i])
        if dsi_results[i] is not None:
            comp["dsi_n_tokens"] = float(dsi_results[i].n_tokens)
        results.append(
            EvalResult(
                prompt=prompts[i],
                response=responses[i],
                category=categories[i],
                dsi=_dsi(i),
                lexical=lex_results[i],
                ppl=_ppl(i) if _ppl(i) is not None else float("nan"),
                reward=reward_vals[i],
                ensemble=ensembles[i],
                creativity_score=finals[i],
                n_tokens=n_tokens_list[i],
                stage1_only=ensembles[i] is None,
                timestamp=timestamp,
                config_hash=config_hash,
                components=comp,
            )
        )
    return results
