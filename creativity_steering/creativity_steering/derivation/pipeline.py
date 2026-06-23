"""Derivation pipeline: gather → filter → DiffMeans at *every* layer.

This is the spec's "Vector derivation" phase end to end:

  1. generate responses to the creative and neutral framings, caching per-layer activations;
  2. filter to an evaluator-verified subset (confidently-creative vs confidently-not);
  3. DiffMeans at every layer, for the primary capture position and any extra positions.

Crash-safety: in the top-up loop (``target_per_class > 0``) the full state — datapoints +
evaluated metrics (``scored_pool.jsonl``) + per-position activations + the current vectors — is
checkpointed to ``out_dir`` **after every round**, with atomic writes. So a crash, OOM, or a
stopped pod loses at most the in-flight round, and whatever was generated is on disk and
transferable. On relaunch with the same ``out_dir`` the derive resumes from the checkpoint
(deterministic seeds → judge-cache hits), instead of starting over.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file + rename, so a crash mid-write can't corrupt the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    """``np.savez_compressed`` via a temp file + rename (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # file object → numpy won't re-append .npz
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def write_scored_pool(base: Path, examples: list[ResponseExample], positions) -> None:
    """Persist the full scored pool: datapoints + evaluated metrics + per-position activations.

    * ``scored_pool.jsonl`` — one row per example: example_id, framing, prompt, response, and the
      evaluated metrics (composite ``score``, ``novelty``, ``quality_ok``).
    * ``activations/<position>.npz`` — an ``acts`` array ``(n_examples, n_layers, hidden_dim)``
      per capture position, row-aligned to ``scored_pool.jsonl``.

    Atomic writes throughout so an interrupted save never leaves a corrupt checkpoint.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "example_id": i,
                "framing": ex.framing,
                "prompt": ex.prompt,
                "response": ex.response,
                "score": ex.score,
                "novelty": ex.novelty,
                "quality_ok": ex.quality_ok,
            }
        )
        for i, ex in enumerate(examples)
    ]
    _atomic_write_text(base / "scored_pool.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    acts_dir = base / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    for position in positions:
        stack = np.stack([ex.activations[position] for ex in examples])
        _atomic_savez(acts_dir / f"{position}.npz", acts=stack)


def load_checkpoint_pool(out_dir, positions) -> list[ResponseExample]:
    """Reload a checkpointed scored pool into ``ResponseExample``s for resume. ``[]`` if absent
    or incomplete (any requested position missing / row-count mismatch → start fresh)."""
    if out_dir is None:
        return []
    base = Path(out_dir)
    pool_path = base / "scored_pool.jsonl"
    if not pool_path.exists():
        return []
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line]
    if not rows:
        return []
    acts: dict[str, np.ndarray] = {}
    for p in positions:
        npz = base / "activations" / f"{p}.npz"
        if not npz.exists():
            return []
        acts[p] = np.load(npz)["acts"]
        if acts[p].shape[0] != len(rows):
            return []
    examples: list[ResponseExample] = []
    for i, row in enumerate(rows):
        ex = ResponseExample(
            framing=row["framing"], system="", prompt=row["prompt"], response=row["response"],
            activations={p: acts[p][i] for p in positions}, score=row.get("score"),
        )
        ex.novelty = row.get("novelty")
        ex.quality_ok = row.get("quality_ok")
        examples.append(ex)
    return examples


def derive_vectors(
    backend: ModelBackend,
    config: ExperimentConfig,
    scorer: CreativityScorer,
    prompt_set: PromptSet | None = None,
    *,
    eval_scorers=None,  # creativity_evaluators.Scorers for the evaluator_stack method (testing)
    out_dir: str | Path | None = None,  # when set, the top-up loop checkpoints/resumes here
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

    def _build_per_position(subset) -> dict[str, SteeringVectorSet]:
        per: dict[str, SteeringVectorSet] = {}
        for position in positions:
            pos, neg = subset.stacks(position)
            per[position] = SteeringVectorSet.from_diffmeans(
                DiffMeansResult.from_stacks(pos, neg),
                model_id=backend.model_id, capture_position=position,
                creative_system=prompt_set.creative_system,
                neutral_system=prompt_set.neutral_system,
            )
        return per

    def _checkpoint(examples, subset) -> None:
        """Persist the round's full state to out_dir (datapoints+metrics+activations always;
        vectors+summary+contrast when both pools are non-empty). Atomic, so always transferable."""
        if out_dir is None:
            return
        base = Path(out_dir)
        write_scored_pool(base, examples, positions)
        if subset.n_positive > 0 and subset.n_negative > 0:
            per = _build_per_position(subset)
            ckpt = DerivationOutput(
                vector_set=per[positions[0]], per_position=per, subset=subset,
                n_examples=len(examples), examples=None,
            )
            save_derivation(ckpt, base, save_scored_pool=False)  # vectors+summary+contrast only

    target = fc.target_per_class if is_evaluator else 0
    if target and target > 0:
        # Top-up loop with per-round checkpointing + resume.
        topics = list(prompt_set.topics)
        chunk = max(1, fc.topup_chunk_size)
        examples = load_checkpoint_pool(out_dir, positions)
        topics_done = len({ex.prompt for ex in examples})
        i = topics_done
        rnd = topics_done // chunk
        subset = _filter(examples, min_per_class=0) if examples else None
        if examples:
            print(
                f"[derive] resumed from checkpoint: {len(examples)} examples / {topics_done} "
                f"topics done -> {subset.n_positive} pos / {subset.n_negative} neg "
                f"(target {target})",
                flush=True,
            )
        while (
            (subset is None or subset.n_positive < target or subset.n_negative < target)
            and i < len(topics)
        ):
            batch = topics[i : i + chunk]
            i += chunk
            examples.extend(_gather(batch, seed=gen.seed + rnd))
            rnd += 1
            subset = _filter(examples, min_per_class=0)
            print(
                f"[derive] top-up round {rnd}: {min(i, len(topics))}/{len(topics)} topics, "
                f"{len(examples)} examples -> {subset.n_positive} pos / {subset.n_negative} neg "
                f"(target {target})",
                flush=True,
            )
            _checkpoint(examples, subset)
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

    per_position = _build_per_position(subset)
    examples_out = examples if is_evaluator else None

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
    scored pool (datapoints + metrics + per-position activations) via ``write_scored_pool`` so
    the contrast can be re-cut at different cutoffs / positions without re-running generation.
    """
    base = Path(out_dir)
    root = base / "vectors"
    for position, vs in output.per_position.items():
        vs.save(root / position)

    base.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        base / "derivation_summary.json",
        json.dumps(output.summary(), indent=2, sort_keys=True),
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
    _atomic_write_text(base / "contrast_set.jsonl", "\n".join(lines) + ("\n" if lines else ""))

    if save_scored_pool and output.examples is not None:
        write_scored_pool(base, output.examples, sorted(output.per_position))

    return root
