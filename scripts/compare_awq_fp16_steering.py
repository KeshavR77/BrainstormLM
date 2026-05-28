#!/usr/bin/env python3
"""Compare creative steering quality: AWQ-derived vs FP16-derived vectors.

Full extraction (32 prompts) → validation (20 prompts × 3 arms) → quantitative
metrics (Distinct-2, Self-BLEU, mean length, lexical overlap).

Usage on RunPod::

    export HF_HOME=/workspace/.cache/huggingface
    cd /workspace/BrainstormLM
    python3 scripts/compare_awq_fp16_steering.py \
        --awq-model-path Qwen/Qwen2.5-14B-Instruct-AWQ

    # Skip extraction if AWQ vectors already derived:
    python3 scripts/compare_awq_fp16_steering.py \
        --awq-model-path Qwen/Qwen2.5-14B-Instruct-AWQ --skip-extract

    # Quick mode (8 validation prompts instead of 20):
    python3 scripts/compare_awq_fp16_steering.py \
        --awq-model-path Qwen/Qwen2.5-14B-Instruct-AWQ --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Validation prompts — diverse domains, distinct from the 32 extraction prompts
# ---------------------------------------------------------------------------

VALIDATION_PROMPTS = [
    # Strategy / business
    "How could a small e-commerce brand compete with Amazon in a niche category?",
    "Design a loyalty program that doesn't rely on discounts or points.",
    "What business model could make investigative journalism sustainable?",
    "Propose a strategy for a B2B SaaS company to break into the Japanese market.",
    "How should a mid-size law firm position itself against AI-native legal startups?",
    # Creative / design
    "Concept a rebrand for a 50-year-old regional airline.",
    "Pitch a visual identity for an AI-powered music composition tool.",
    "Design an onboarding experience for a meditation app aimed at skeptics.",
    "Sketch a campaign for a carbon-offset marketplace targeting frequent flyers.",
    "Concept a launch event for a luxury electric bicycle brand.",
    # Systems / technical
    "How might a city redesign public transit to increase ridership by 3x?",
    "Propose an architecture for a real-time collaborative design tool like Figma.",
    "Design a system that matches volunteer tutors with students in rural areas.",
    "How could a hospital reduce ER wait times by 40% without adding staff?",
    "Propose a supply-chain transparency system for a fast-fashion retailer.",
    # Wildcard / lateral
    "Invent a new sport that can be played in a standard office conference room.",
    "Design a museum exhibit that explains quantum computing to 10-year-olds.",
    "Propose a neighborhood composting system that actually scales.",
    "How could a public library reinvent itself for 2030?",
    "Concept a dating app where users can't see photos.",
]


# ---------------------------------------------------------------------------
# Metrics (local, no API calls — runs on RunPod without OpenAI key)
# ---------------------------------------------------------------------------

def _bigrams(text: str) -> list[tuple[str, str]]:
    tokens = re.findall(r"\w+", text.lower())
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def distinct_2(texts: list[str]) -> float:
    """Distinct-2: fraction of unique bigrams over total bigrams."""
    total: list[tuple[str, str]] = []
    for t in texts:
        total.extend(_bigrams(t))
    if not total:
        return 1.0
    return len(set(total)) / len(total)


def self_bleu(texts: list[str], n: int = 4) -> float:
    """Self-BLEU: lower = more diverse."""
    if len(texts) < 2:
        return 0.0
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    tokenized = [t.lower().split() for t in texts]
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
    return float(sum(scores) / len(scores)) if scores else 0.0


def vocab_richness(texts: list[str]) -> float:
    """Type-token ratio across all texts."""
    tokens = []
    for t in texts:
        tokens.extend(re.findall(r"\w+", t.lower()))
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def pairwise_jaccard(texts: list[str]) -> float:
    """Mean pairwise Jaccard similarity of word sets. Lower = more diverse."""
    sets = [set(re.findall(r"\w+", t.lower())) for t in texts]
    if len(sets) < 2:
        return 0.0
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            sims.append(len(sets[i] & sets[j]) / len(union))
    return float(sum(sims) / len(sims)) if sims else 0.0


def compute_metrics(texts: list[str]) -> dict:
    """Compute all diversity + quality metrics for a set of outputs."""
    lengths = [len(t.split()) for t in texts]
    return {
        "n_outputs": len(texts),
        "mean_length_words": round(statistics.mean(lengths), 1),
        "std_length_words": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0.0,
        "distinct_2": round(distinct_2(texts), 4),
        "self_bleu_4": round(self_bleu(texts, n=4), 4),
        "vocab_richness": round(vocab_richness(texts), 4),
        "mean_jaccard_sim": round(pairwise_jaccard(texts), 4),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_arm(
    engine,
    prompts: list[str],
    do_steering: bool,
    arm_name: str,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate responses for all prompts under one steering configuration."""
    outputs = []
    for i, prompt in enumerate(prompts):
        text = engine.complete(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=max_new_tokens,
            do_steering=do_steering,
            temperature=0.8,
        )
        outputs.append(text)
        if (i + 1) % 5 == 0 or i == len(prompts) - 1:
            print(f"    [{arm_name}] {i+1}/{len(prompts)}")
    return outputs


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    awq_model_path: str,
    awq_vector_dir: Path,
    fp16_vector_dir: Path,
    prompts: list[str],
    out_dir: Path,
    max_new_tokens: int = 256,
) -> dict:
    """Run 3-arm comparison and compute metrics."""
    from brainstormlm.conceptial.engine.config import SteeringConfig
    from brainstormlm.conceptial.engine.pipeline import SteeringEngine, apply_layer_injections

    creative_alpha = 1.5
    technical_alpha = 0.86

    def make_injections(vector_dir: Path) -> list[dict]:
        return [
            {
                "layer": 17,
                "vector_path": str(vector_dir / "hyper_creative_purified_vector_L17.pt"),
                "start_alpha": creative_alpha,
                "decay_tokens": 168,
                "alpha_schedule": "exp_decay",
                "exp_zero_by_step": 168,
            },
            {
                "layer": 24,
                "vector_path": str(vector_dir / "hyper_technical_purified_vector_L24.pt"),
                "start_alpha": technical_alpha,
                "decay_tokens": 0,
                "alpha_schedule": "sigmoid_ramp",
                "sigmoid_mid_step": 52.0,
                "sigmoid_steepness": 0.095,
            },
        ]

    results: dict = {
        "model": awq_model_path,
        "awq_vector_dir": str(awq_vector_dir),
        "fp16_vector_dir": str(fp16_vector_dir),
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "creative_alpha": creative_alpha,
        "technical_alpha": technical_alpha,
        "prompts": prompts,
        "arms": {},
    }

    # --- ARM 1: Unsteered baseline ---
    print(f"\n{'='*60}")
    print("ARM 1/3: Unsteered baseline")
    print(f"{'='*60}")
    config_base = SteeringConfig(model_id=awq_model_path, torch_dtype="float16")
    engine = SteeringEngine(config_base)
    engine.load()

    t0 = time.time()
    unsteered = generate_arm(engine, prompts, do_steering=False, arm_name="unsteered")
    t_unsteered = time.time() - t0
    results["arms"]["unsteered"] = {
        "outputs": unsteered,
        "metrics": compute_metrics(unsteered),
        "elapsed_s": round(t_unsteered, 1),
    }
    print(f"  Done in {t_unsteered:.0f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- ARM 2: AWQ-derived vectors ---
    print(f"\n{'='*60}")
    print("ARM 2/3: AWQ-derived steering vectors")
    print(f"{'='*60}")
    config_awq = SteeringConfig(
        model_id=awq_model_path,
        torch_dtype="float16",
        layer_injections=make_injections(awq_vector_dir),
    )
    engine.config = config_awq
    apply_layer_injections(engine)

    t0 = time.time()
    awq_steered = generate_arm(engine, prompts, do_steering=True, arm_name="awq_vectors")
    t_awq = time.time() - t0
    results["arms"]["awq_vectors"] = {
        "outputs": awq_steered,
        "metrics": compute_metrics(awq_steered),
        "elapsed_s": round(t_awq, 1),
    }
    print(f"  Done in {t_awq:.0f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- ARM 3: FP16-derived vectors ---
    fp16_creative = fp16_vector_dir / "hyper_creative_purified_vector_L17.pt"
    fp16_technical = fp16_vector_dir / "hyper_technical_purified_vector_L24.pt"
    if fp16_creative.exists() and fp16_technical.exists():
        print(f"\n{'='*60}")
        print("ARM 3/3: FP16-derived steering vectors (cross-precision)")
        print(f"{'='*60}")
        config_fp16 = SteeringConfig(
            model_id=awq_model_path,
            torch_dtype="float16",
            layer_injections=make_injections(fp16_vector_dir),
        )
        engine.config = config_fp16
        apply_layer_injections(engine)

        t0 = time.time()
        fp16_steered = generate_arm(engine, prompts, do_steering=True, arm_name="fp16_vectors")
        t_fp16 = time.time() - t0
        results["arms"]["fp16_vectors"] = {
            "outputs": fp16_steered,
            "metrics": compute_metrics(fp16_steered),
            "elapsed_s": round(t_fp16, 1),
        }
        print(f"  Done in {t_fp16:.0f}s")
    else:
        print(f"\n  ⚠ FP16 vectors not found at {fp16_vector_dir}, skipping arm 3")

    # --- Cleanup GPU ---
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison_results.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Full results saved to {report_path}")

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("METRICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<22} {'Unsteered':>12} {'AWQ Vectors':>12} {'FP16 Vectors':>12}")
    print("-" * 60)
    metric_keys = ["mean_length_words", "distinct_2", "self_bleu_4", "vocab_richness", "mean_jaccard_sim"]
    labels = {
        "mean_length_words": "Mean length (words)",
        "distinct_2": "Distinct-2 ↑",
        "self_bleu_4": "Self-BLEU-4 ↓",
        "vocab_richness": "Vocab richness ↑",
        "mean_jaccard_sim": "Jaccard sim ↓",
    }
    for key in metric_keys:
        row = f"{labels[key]:<22}"
        for arm in ["unsteered", "awq_vectors", "fp16_vectors"]:
            if arm in results["arms"]:
                row += f" {results['arms'][arm]['metrics'][key]:>11}"
            else:
                row += f" {'n/a':>11}"
        print(row)

    print("\n  (↑ = higher is better for diversity, ↓ = lower is better)")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--awq-model-path", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ",
                        help="HuggingFace model ID or local path for AWQ model.")
    parser.add_argument("--awq-vectors", type=Path, default=ROOT / "data" / "steering_14b_awq",
                        help="Directory with AWQ-derived steering vectors.")
    parser.add_argument("--fp16-vectors", type=Path, default=ROOT / "data" / "steering_14b",
                        help="Directory with FP16-derived steering vectors.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "contrast_pairs_14b.json",
                        help="Contrast pairs config for vector extraction.")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "awq_fp16_comparison",
                        help="Output directory for results.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip vector re-extraction (use existing AWQ vectors).")
    parser.add_argument("--force-extract", action="store_true",
                        help="Force re-extraction even if vectors exist.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 8 validation prompts instead of 20.")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max tokens per generation (default 256).")
    args = parser.parse_args()

    prompts = VALIDATION_PROMPTS[:8] if args.quick else VALIDATION_PROMPTS

    # --- Step 1: Ensure AWQ vectors exist (full 32-prompt extraction) ---
    awq_creative = args.awq_vectors / "hyper_creative_purified_vector_L17.pt"
    awq_technical = args.awq_vectors / "hyper_technical_purified_vector_L24.pt"
    need_extract = args.force_extract or not (awq_creative.exists() and awq_technical.exists())

    if need_extract and not args.skip_extract:
        print(f"\n{'='*60}")
        print("Step 0: Full 32-prompt AWQ vector extraction")
        print(f"{'='*60}")

        # Remove existing vectors if force-extract
        if args.force_extract:
            for p in [awq_creative, awq_technical]:
                if p.exists():
                    p.unlink()
                    print(f"  Removed existing: {p.name}")

        # Import and run extraction from the main pipeline script
        sys.path.insert(0, str(ROOT / "scripts"))
        from quantize_and_steer_14b import extract_vectors
        extract_vectors(
            model_id=args.awq_model_path,
            out_dir=args.awq_vectors,
            config_path=args.config,
            creative_layer=17,
            technical_layer=24,
            limit_prompts=None,  # Full 32 prompts
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif need_extract and args.skip_extract:
        print(f"\n  ⚠ AWQ vectors not found at {args.awq_vectors} and --skip-extract set.")
        print("  Run without --skip-extract first to derive vectors.")
        return 1
    else:
        print(f"\n  AWQ vectors found at {args.awq_vectors}, skipping extraction.")

    # --- Step 2: Run comparison ---
    run_comparison(
        awq_model_path=args.awq_model_path,
        awq_vector_dir=args.awq_vectors,
        fp16_vector_dir=args.fp16_vectors,
        prompts=prompts,
        out_dir=args.out,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n✓ Comparison complete. Results: {args.out / 'comparison_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
