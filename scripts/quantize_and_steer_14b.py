#!/usr/bin/env python3
"""Quantize 14B → AWQ 4-bit, re-derive steering vectors, validate quality.

End-to-end pipeline for RunPod A6000/A100. Produces:
  1. Quantized model checkpoint (Qwen2.5-14B-Instruct-AWQ)
  2. Steering vectors derived on quantized activations
  3. Side-by-side quality comparison (FP16 vectors vs AWQ vectors)

Usage on RunPod::

    cd /workspace/BrainstormLM
    uv sync
    pip install autoawq
    export HF_TOKEN=hf_...

    # Full pipeline (quantize + extract + validate):
    uv run python scripts/quantize_and_steer_14b.py

    # Skip quantization if model already quantized:
    uv run python scripts/quantize_and_steer_14b.py --skip-quantize \
        --awq-model-path /workspace/models/Qwen2.5-14B-Instruct-AWQ

    # Quick smoke test (3 prompts instead of 32):
    uv run python scripts/quantize_and_steer_14b.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Step 1: Quantize
# ---------------------------------------------------------------------------

def quantize_model(model_id: str, output_path: Path) -> None:
    """Quantize a HuggingFace model to AWQ 4-bit."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Step 1: Quantizing {model_id} → AWQ 4-bit")
    print(f"{'='*60}")

    if output_path.exists() and (output_path / "config.json").exists():
        print(f"  Skipping: {output_path} already exists.")
        return

    t0 = time.time()
    print("  Loading FP16 model for quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print("  Quantizing (group_size=128, 4-bit)... this takes 10-15 min")
    model.quantize(tokenizer, quant_config=quant_config)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 2: Extract steering vectors on quantized model
# ---------------------------------------------------------------------------

def extract_vectors(
    model_id: str,
    out_dir: Path,
    config_path: Path,
    creative_layer: int = 17,
    technical_layer: int = 24,
    limit_prompts: int | None = None,
) -> None:
    """Re-derive steering vectors on the quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainstormlm.conceptial.engine.vector_purify import hyper_purify_vector

    print(f"\n{'='*60}")
    print(f"Step 2: Extracting steering vectors from {model_id}")
    print(f"{'='*60}")

    # Check if vectors already exist
    creative_path = out_dir / f"hyper_creative_purified_vector_L{creative_layer}.pt"
    technical_path = out_dir / f"hyper_technical_purified_vector_L{technical_layer}.pt"
    if creative_path.exists() and technical_path.exists():
        print(f"  Vectors already exist at {out_dir}. Skipping extraction.")
        return

    cfg = json.loads(config_path.read_text())
    framings = cfg["framings"]
    prompts: list[str] = cfg["prompts"]
    if limit_prompts:
        prompts = prompts[:limit_prompts]
        print(f"  Limited to {limit_prompts} prompts (smoke mode)")
    neg_banks_creative: list[str] = cfg["negative_banks_for_creative"]
    neg_banks_technical: list[str] = cfg["negative_banks_for_technical"]

    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    print(f"  Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")

    def render_pair(system: str, user: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def last_token_hidden(text: str, layer_idx: int) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        v = outputs.hidden_states[layer_idx][0, -1, :].detach().to("cpu").float()
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return v

    def derive_caa(pos_system: str, neg_system: str, layer_idx: int, desc: str) -> torch.Tensor:
        deltas = []
        for i, prompt in enumerate(prompts):
            pos_h = last_token_hidden(render_pair(pos_system, prompt), layer_idx)
            neg_h = last_token_hidden(render_pair(neg_system, prompt), layer_idx)
            deltas.append(pos_h - neg_h)
            if (i + 1) % 8 == 0:
                print(f"    [{desc}] L{layer_idx}: {i+1}/{len(prompts)} pairs")
        v = torch.stack(deltas, dim=0).mean(0)
        return v / (torch.linalg.norm(v) + 1e-12)

    # Raw vectors
    print(f"\n  Deriving creative CAA @ L{creative_layer}...")
    raw_creative = derive_caa(framings["creative"], framings["generic"], creative_layer, "creative")

    print(f"\n  Deriving technical CAA @ L{technical_layer}...")
    raw_technical = derive_caa(framings["technical"], framings["generic"], technical_layer, "technical")

    # Negative banks
    print("\n  Deriving negative-bank vectors...")
    neg_vecs_c: dict[str, torch.Tensor] = {}
    neg_vecs_t: dict[str, torch.Tensor] = {}
    all_modes = sorted(set(neg_banks_creative) | set(neg_banks_technical))
    for mode in all_modes:
        if mode in neg_banks_creative:
            neg_vecs_c[mode] = derive_caa(framings[mode], framings["generic"], creative_layer, f"neg-{mode}")
        if mode in neg_banks_technical:
            neg_vecs_t[mode] = derive_caa(framings[mode], framings["generic"], technical_layer, f"neg-{mode}")

    # Purify
    print("\n  Hyper-purifying creative vector...")
    purified_c, c_report = hyper_purify_vector(
        raw_creative, list(neg_vecs_c.values()),
        min_source_cosine=0.15, max_negative_cosine=0.08,
    )
    print(f"    retained={c_report.retained_source_cosine:.4f} "
          f"neg_before={c_report.max_abs_negative_cosine_before:.4f} "
          f"neg_after={c_report.max_abs_negative_cosine_after:.4f}")

    print("\n  Hyper-purifying technical vector...")
    tech_neg_bank = list(neg_vecs_t.values()) + [purified_c]
    purified_t, t_report = hyper_purify_vector(
        raw_technical, tech_neg_bank,
        min_source_cosine=0.15, max_negative_cosine=0.08,
    )
    print(f"    retained={t_report.retained_source_cosine:.4f} "
          f"neg_before={t_report.max_abs_negative_cosine_before:.4f} "
          f"neg_after={t_report.max_abs_negative_cosine_after:.4f}")

    # Save
    for name, vec, layer in [
        ("creative", purified_c, creative_layer),
        ("technical", purified_t, technical_layer),
    ]:
        bundle = {"vector": vec, "layer_index": layer}
        pt_path = out_dir / f"hyper_{name}_purified_vector_L{layer}.pt"
        torch.save(bundle, str(pt_path))
        print(f"  Saved: {pt_path.relative_to(ROOT)}")

    print("  Vector extraction complete.")


# ---------------------------------------------------------------------------
# Step 3: Validate — generate with AWQ+steering, compare to FP16 baseline
# ---------------------------------------------------------------------------

def validate_quality(
    awq_model_path: str,
    awq_vector_dir: Path,
    fp16_vector_dir: Path,
    n_prompts: int = 5,
    out_dir: Path | None = None,
) -> None:
    """Generate side-by-side: AWQ+AWQ-vectors vs AWQ+FP16-vectors vs unsteered."""
    print(f"\n{'='*60}")
    print("Step 3: Quality validation")
    print(f"{'='*60}")

    from brainstormlm.conceptial.engine.config import SteeringConfig
    from brainstormlm.conceptial.engine.pipeline import SteeringEngine, apply_layer_injections

    test_prompts = [
        "How could a small e-commerce brand compete with Amazon in a niche category?",
        "Design a loyalty program that doesn't rely on discounts or points.",
        "What business model could make investigative journalism sustainable?",
        "How might a city redesign public transit to increase ridership by 3x?",
        "Propose a strategy for a B2B SaaS company to break into the Japanese market.",
    ][:n_prompts]

    results: dict[str, list[str]] = {"unsteered": [], "awq_vectors": [], "fp16_vectors": []}

    # --- Unsteered baseline ---
    print("\n  [1/3] Generating unsteered baseline...")
    config_base = SteeringConfig(model_id=awq_model_path, torch_dtype="float16")
    engine = SteeringEngine(config_base)
    engine.load()
    for i, prompt in enumerate(test_prompts):
        text = engine.complete(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=200, do_steering=False, temperature=0.8,
        )
        results["unsteered"].append(text)
        print(f"    [{i+1}/{len(test_prompts)}] {text[:80]}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- AWQ vectors ---
    print("\n  [2/3] Generating with AWQ-derived vectors...")
    config_awq = SteeringConfig(
        model_id=awq_model_path,
        torch_dtype="float16",
        layer_injections=[
            {"layer": 17, "vector_path": str(awq_vector_dir / "hyper_creative_purified_vector_L17.pt"),
             "start_alpha": 1.5, "decay_tokens": 168,
             "alpha_schedule": "exp_decay", "exp_zero_by_step": 168},
            {"layer": 24, "vector_path": str(awq_vector_dir / "hyper_technical_purified_vector_L24.pt"),
             "start_alpha": 0.86, "decay_tokens": 0,
             "alpha_schedule": "sigmoid_ramp", "sigmoid_mid_step": 52.0,
             "sigmoid_steepness": 0.095},
        ],
    )
    # Reuse loaded model
    engine.config = config_awq
    apply_layer_injections(engine)
    for i, prompt in enumerate(test_prompts):
        text = engine.complete(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=200, do_steering=True, temperature=0.8,
        )
        results["awq_vectors"].append(text)
        print(f"    [{i+1}/{len(test_prompts)}] {text[:80]}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- FP16 vectors on AWQ model (should be worse) ---
    print("\n  [3/3] Generating with FP16-derived vectors (mismatch test)...")
    fp16_creative = fp16_vector_dir / "hyper_creative_purified_vector_L17.pt"
    fp16_technical = fp16_vector_dir / "hyper_technical_purified_vector_L24.pt"
    if fp16_creative.exists() and fp16_technical.exists():
        config_fp16 = SteeringConfig(
            model_id=awq_model_path,
            torch_dtype="float16",
            layer_injections=[
                {"layer": 17, "vector_path": str(fp16_creative),
                 "start_alpha": 1.5, "decay_tokens": 168,
                 "alpha_schedule": "exp_decay", "exp_zero_by_step": 168},
                {"layer": 24, "vector_path": str(fp16_technical),
                 "start_alpha": 0.86, "decay_tokens": 0,
                 "alpha_schedule": "sigmoid_ramp", "sigmoid_mid_step": 52.0,
                 "sigmoid_steepness": 0.095},
            ],
        )
        engine.config = config_fp16
        apply_layer_injections(engine)
        for i, prompt in enumerate(test_prompts):
            text = engine.complete(
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                max_new_tokens=200, do_steering=True, temperature=0.8,
            )
            results["fp16_vectors"].append(text)
            print(f"    [{i+1}/{len(test_prompts)}] {text[:80]}...")
    else:
        print(f"    Skipped: FP16 vectors not found at {fp16_vector_dir}")

    # --- Save results to disk ---
    save_dir = out_dir or (ROOT / "artifacts" / "awq_validation")
    save_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": awq_model_path,
        "awq_vector_dir": str(awq_vector_dir),
        "fp16_vector_dir": str(fp16_vector_dir),
        "prompts": test_prompts,
        "results": results,
    }
    report_path = save_dir / "validation_results.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Results saved to {report_path}")

    # --- Print comparison ---
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Prompt {i+1}: {prompt[:60]}... ---")
        print(f"\n  UNSTEERED:\n    {results['unsteered'][i][:200]}")
        print(f"\n  AWQ VECTORS (should be best):\n    {results['awq_vectors'][i][:200]}")
        if results["fp16_vectors"]:
            print(f"\n  FP16 VECTORS (mismatch — expect degraded):\n    {results['fp16_vectors'][i][:200]}")

    print(f"\n{'='*60}")
    print("Next: run the eval harness on each set to get creativity/logic scores:")
    print(f"  uv run python scripts/score_outputs.py --input {report_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct",
                        help="HuggingFace model ID to quantize.")
    parser.add_argument("--awq-model-path", type=Path, default=None,
                        help="Path to already-quantized model (skip quantization).")
    parser.add_argument("--out-model", type=Path, default=Path("/workspace/models/Qwen2.5-14B-Instruct-AWQ"),
                        help="Where to save the quantized model.")
    parser.add_argument("--out-vectors", type=Path, default=ROOT / "data" / "steering_14b_awq",
                        help="Where to save AWQ-derived steering vectors.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "contrast_pairs_14b.json",
                        help="Contrast pairs config for vector derivation.")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip quantization (use --awq-model-path).")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip vector extraction (use existing vectors).")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip quality validation.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 prompts for extraction, 2 for validation.")
    parser.add_argument("--out-results", type=Path, default=ROOT / "artifacts" / "awq_validation",
                        help="Where to save validation results JSON.")
    parser.add_argument("--creative-layer", type=int, default=17)
    parser.add_argument("--technical-layer", type=int, default=24)
    args = parser.parse_args()

    awq_path = args.awq_model_path or args.out_model
    limit = 3 if args.smoke else None
    n_val = 2 if args.smoke else 5

    # Step 1
    if not args.skip_quantize and args.awq_model_path is None:
        quantize_model(args.model_id, args.out_model)

    # Step 2
    if not args.skip_extract:
        extract_vectors(
            model_id=str(awq_path),
            out_dir=args.out_vectors,
            config_path=args.config,
            creative_layer=args.creative_layer,
            technical_layer=args.technical_layer,
            limit_prompts=limit,
        )
        # Free extraction model before validate_quality loads a second copy
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 3
    if not args.skip_validate:
        validate_quality(
            awq_model_path=str(awq_path),
            awq_vector_dir=args.out_vectors,
            fp16_vector_dir=ROOT / "data" / "steering_14b",
            n_prompts=n_val,
            out_dir=args.out_results,
        )

    print("\n✓ Pipeline complete.")
    print(f"  Quantized model: {awq_path}")
    print(f"  Steering vectors: {args.out_vectors}")
    print("  Config: config/conceptial_14b_awq.json")
    print("\n  To serve: update config/conceptial_14b_awq.json base_model")
    print(f"  to point at {awq_path}, then:")
    print("    uv run uvicorn brainstormlm.api.server:app --host 0.0.0.0 --port 8001")
    return 0


if __name__ == "__main__":
    sys.exit(main())
