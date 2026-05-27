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
from creativity_steering.derivation.generate import gather_responses
from creativity_steering.derivation.prompts import PromptSet, default_prompt_set
from creativity_steering.derivation.records import VerifiedSubset
from creativity_steering.models.backend import ModelBackend
from creativity_steering.scoring import CreativityScorer


@dataclass
class DerivationOutput:
    """Everything the derivation phase produces."""

    vector_set: SteeringVectorSet  # primary capture position
    per_position: dict[str, SteeringVectorSet]  # position -> vector set (incl. primary)
    subset: VerifiedSubset
    n_examples: int

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
    prompt_set = prompt_set or default_prompt_set()
    positions = config.all_capture_positions

    examples = gather_responses(
        backend,
        prompt_set,
        positions=positions,
        samples_per_prompt=config.generation.samples_per_prompt,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        seed=config.generation.seed,
        sampling=config.generation.sampling_by_framing(),
    )

    fc = config.filtering
    if fc.method == "evaluator_stack":
        # Score with the two-stage creativity_evaluators stack; keep top-creative /
        # bottom-neutral by quantile. The sibling-package import stays lazy in here.
        from creativity_steering.derivation.evaluator_filter import evaluate_and_filter

        subset = evaluate_and_filter(
            examples,
            scorers=eval_scorers,
            creative_quantile=fc.creative_quantile,
            neutral_quantile=fc.neutral_quantile,
            creative_score_min=fc.creative_score_min,
            neutral_score_max=fc.neutral_score_max,
            enable_judge=fc.enable_judge,
            generator_family=fc.generator_family,
            stage1_floor_percentile=fc.stage1_floor_percentile,
            length_match=fc.length_match,
            length_match_tolerance=fc.length_match_tolerance,
            use_reward_fallback=fc.use_reward_fallback,
            stage1_device=config.device,
            min_per_class=fc.min_per_class,
        )
    else:
        subset = label_and_filter(
            examples,
            scorer,
            creative_min=fc.creative_min,
            neutral_max=fc.neutral_max,
            min_per_class=fc.min_per_class,
        )

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

    return DerivationOutput(
        vector_set=per_position[config.capture_position],
        per_position=per_position,
        subset=subset,
        n_examples=len(examples),
    )


def save_derivation(output: DerivationOutput, out_dir: str | Path) -> Path:
    """Write one vector set per capture position under ``out_dir/vectors/<position>/``.

    Also persists the verified contrast set (the actual DiffMeans inputs) to
    ``contrast_set.jsonl`` — text only (activations live in the vector sets) — so the pools
    can be inspected and re-filtered without re-generating.
    """
    import json

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
    return root
