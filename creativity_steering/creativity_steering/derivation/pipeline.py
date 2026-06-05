"""Derivation pipeline: gather → filter → DiffMeans at *every* layer.

This is the spec's "Vector derivation" phase end to end:

  1. generate responses to the creative and neutral framings, caching per-layer activations;
  2. filter to an evaluator-verified subset (confidently-creative vs confidently-not);
  3. DiffMeans at every layer, for the primary capture position and any extra positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.config import ExperimentConfig
from creativity_steering.derivation.diffmeans import DiffMeansResult
from creativity_steering.derivation.filter import label_and_filter
from creativity_steering.derivation.generate import gather_responses, gather_responses_unframed
from creativity_steering.derivation.prompts import PromptSet, default_prompt_set, load_prompt_set
from creativity_steering.derivation.records import ResponseExample, VerifiedSubset
from creativity_steering.models.backend import ModelBackend
from creativity_steering.scoring import CreativityScorer


@dataclass
class DerivationOutput:
    """Everything the derivation phase produces."""

    vector_set: SteeringVectorSet  # primary capture position
    per_position: dict[str, SteeringVectorSet]  # position -> vector set (incl. primary)
    subset: VerifiedSubset
    n_examples: int
    # The full scored pool (every gathered example with its composite score and per-position
    # activations), kept so the contrast set can be re-filtered at different cutoffs without
    # re-running generation. Populated only when filtering uses the evaluator stack (the only
    # path that scores every example), and only persisted when save_scored_pool=True.
    examples: list[ResponseExample] | None = None

    def summary(self) -> dict:
        return {
            "model_id": self.vector_set.model_id,
            "primary_capture_position": self.vector_set.capture_position,
            "positions": sorted(self.per_position),
            "n_examples": self.n_examples,
            "n_positive": self.subset.n_positive,
            "n_negative": self.subset.n_negative,
            "n_dropped": self.subset.n_dropped,
            "n_layers": self.vector_set.n_layers,
            "hidden_dim": self.vector_set.hidden_dim,
        }


def derive_vectors(
    backend: ModelBackend,
    config: ExperimentConfig,
    scorer: CreativityScorer,
    prompt_set: PromptSet | None = None,
    *,
    eval_scorers=None,  # creativity_evaluators.Scorers for the evaluator_stack method (testing)
) -> DerivationOutput:
    if prompt_set is None:
        prompt_set = (
            load_prompt_set(config.topics_file) if config.topics_file else default_prompt_set()
        )
    positions = config.all_capture_positions

    fc = config.filtering
    gen = config.generation
    is_evaluator = fc.method in ("evaluator_stack", "score_split")

    def _gather(topics_chunk, *, seed: int) -> list[ResponseExample]:
        """Generate the (framed or unframed) responses for one chunk of topics."""
        if fc.method == "score_split":
            return gather_responses_unframed(
                backend, list(topics_chunk), positions=positions,
                samples_per_prompt=gen.samples_per_prompt, max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature, seed=seed, sampling=gen.unframed_sampling,
            )
        chunk_set = PromptSet(
            prompt_set.creative_system, prompt_set.neutral_system, tuple(topics_chunk)
        )
        return gather_responses(
            backend, chunk_set, positions=positions,
            samples_per_prompt=gen.samples_per_prompt, max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature, seed=seed, sampling=gen.sampling_by_framing(),
        )

    def _filter(examples, *, min_per_class: int):
        if is_evaluator:
            # ``evaluator_stack`` splits by framing; ``score_split`` splits the unified pool by
            # score. Sibling-package import stays lazy.
            from creativity_steering.derivation.evaluator_filter import evaluate_and_filter

            return evaluate_and_filter(
                examples, scorers=eval_scorers,
                creative_quantile=fc.creative_quantile, neutral_quantile=fc.neutral_quantile,
                creative_score_min=fc.creative_score_min, neutral_score_max=fc.neutral_score_max,
                split_by="score" if fc.method == "score_split" else "framing",
                enable_judge=fc.enable_judge, generator_family=fc.generator_family,
                stage1_floor_percentile=fc.stage1_floor_percentile,
                length_match=fc.length_match, length_match_tolerance=fc.length_match_tolerance,
                enable_reward=fc.enable_reward, use_reward_fallback=fc.use_reward_fallback,
                stage1_device=config.device, min_per_class=min_per_class,
                one_per_prompt=fc.one_per_prompt, quality_floor=fc.quality_floor,
            )
        return label_and_filter(
            examples, scorer, creative_min=fc.creative_min, neutral_max=fc.neutral_max,
            min_per_class=min_per_class,
        )

    target = fc.target_per_class if is_evaluator else 0
    if target and target > 0:
        # Top-up loop: generate topics in chunks and re-filter the GROWING pool each round
        # (judges are cached, so the re-score is cheap) until both pools reach ``target`` or the
        # bank is exhausted. Never raises mid-loop (min_per_class=0), so the expensive generation
        # is always available to save — the fix for v3 run #1's pre-save abort.
        topics = list(prompt_set.topics)
        chunk = max(1, fc.topup_chunk_size)
        examples: list[ResponseExample] = []
        subset = None
        rnd = 0
        i = 0
        while i < len(topics):
            batch = topics[i : i + chunk]
            i += chunk
            examples.extend(_gather(batch, seed=gen.seed + rnd))
            subset = _filter(examples, min_per_class=0)
            rnd += 1
            print(
                f"[derive] top-up round {rnd}: {min(i, len(topics))}/{len(topics)} topics, "
                f"{len(examples)} examples -> {subset.n_positive} pos / {subset.n_negative} neg "
                f"(target {target})",
                flush=True,
            )
            if subset.n_positive >= target and subset.n_negative >= target:
                break
        if subset is None:
            subset = _filter(examples, min_per_class=0)
        if subset.n_positive < target or subset.n_negative < target:
            print(
                f"[derive] WARNING: topic bank exhausted ({len(topics)} topics) before reaching "
                f"target_per_class={target}: got {subset.n_positive} pos / {subset.n_negative} "
                f"neg. Building vectors from what we have; expand the topic bank to hit the target.",
                flush=True,
            )
    else:
        examples = _gather(prompt_set.topics, seed=gen.seed)
        subset = _filter(examples, min_per_class=fc.min_per_class)

    per_position: dict[str, SteeringVectorSet] = {}
    for position in positions:
        pos, neg = subset.stacks(position)
        result = DiffMeansResult.from_stacks(pos, neg)
        per_position[position] = SteeringVectorSet.from_diffmeans(
            result,
            model_id=backend.model_id,
            capture_position=position,
            creative_system=prompt_set.creative_system,
            neutral_system=prompt_set.neutral_system,
        )

    # Whether to retain the full scored pool on the output (caller decides what to do with it).
    examples_out = examples if fc.method in ("evaluator_stack", "score_split") else None

    return DerivationOutput(
        vector_set=per_position[config.capture_position],
        per_position=per_position,
        subset=subset,
        n_examples=len(examples),
        examples=examples_out,
    )


def save_derivation(
    output: DerivationOutput,
    out_dir: str | Path,
    *,
    save_scored_pool: bool = False,
) -> Path:
    """Write one vector set per capture position under ``out_dir/vectors/<position>/``.

    Also persists the verified contrast set (the actual DiffMeans inputs) to
    ``contrast_set.jsonl`` — text only (activations live in the vector sets) — so the pools
    can be inspected and re-filtered without re-generating.

    When ``save_scored_pool=True`` and ``output.examples`` is populated, also writes the FULL
    scored pool (every gathered example, not just the verified subset) so the contrast set
    can be re-filtered at different cutoffs / capture positions without re-running generation
    or scoring:

    * ``scored_pool.jsonl`` — one line per example: ``{example_id, framing, prompt, response, score}``
      (the same row index used for the activations arrays below)
    * ``activations/<position>.npz`` — a single ``acts`` array of shape
      ``(n_examples, n_layers, hidden_dim)`` per capture position; the first axis is the
      example_id from ``scored_pool.jsonl``.
    """
    import json

    import numpy as np

    base = Path(out_dir)
    root = base / "vectors"
    for position, vs in output.per_position.items():
        vs.save(root / position)

    (base / "derivation_summary.json").write_text(
        json.dumps(output.summary(), indent=2, sort_keys=True)
    )

    lines = [
        json.dumps(
            {
                "pool": pool,
                "framing": ex.framing,
                "prompt": ex.prompt,
                "response": ex.response,
                "score": ex.score,
            }
        )
        for pool, examples in (
            ("positive", output.subset.positives),
            ("negative", output.subset.negatives),
        )
        for ex in examples
    ]
    base.mkdir(parents=True, exist_ok=True)
    (base / "contrast_set.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))

    if save_scored_pool and output.examples is not None:
        examples = output.examples
        scored_lines = [
            json.dumps(
                {
                    "example_id": i,
                    "framing": ex.framing,
                    "prompt": ex.prompt,
                    "response": ex.response,
                    "score": ex.score,
                    # Persisted so the framed novelty/quality selection can be re-cut offline
                    # (different quantile / length tolerance) without re-running generation.
                    "novelty": ex.novelty,
                    "quality_ok": ex.quality_ok,
                }
            )
            for i, ex in enumerate(examples)
        ]
        (base / "scored_pool.jsonl").write_text(
            "\n".join(scored_lines) + ("\n" if scored_lines else "")
        )

        acts_dir = base / "activations"
        acts_dir.mkdir(parents=True, exist_ok=True)
        positions = sorted(output.per_position)
        for position in positions:
            # Stack (n_examples, n_layers, hidden_dim). Examples that lack the position raise
            # — that's a programming error, every cached example should carry every position.
            stack = np.stack([ex.activations[position] for ex in examples])
            np.savez_compressed(acts_dir / f"{position}.npz", acts=stack)

    return root
