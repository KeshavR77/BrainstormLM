#!/usr/bin/env python3
"""Extract CAA steering vectors + hyper-purified directions for Qwen2.5-14B-Instruct.

Runs on a RunPod A100 80GB pod. Ports the 3B steering pipeline (Corpius 1) to
14B by:

1. Building (positive, generic) contrast pairs from
   ``config/contrast_pairs_14b.json`` under the locked system-prompt framings.
2. Forwarding each pair through Qwen2.5-14B-Instruct with
   ``output_hidden_states=True``, capturing the last-token residual at the
   target layer.
3. Computing the mean (positive - generic) delta → raw CAA vector.
4. Repeating for the 5 negative-mode framings to build a negative bank.
5. Calling ``hyper_purify_vector`` (SVD-orthogonal projection) to project the
   raw vector orthogonal to the negative subspace, with retained-source-cosine
   guardrails.
6. Writing ``data/steering_14b/hyper_{creative,technical}_purified_vector_L{19,24}.pt``
   plus matching ``..._manifest.json`` files in the same shape as the 3B
   artifacts under ``data/steering/``.

Layer choice — depth-scaled from the 3B baseline:
- 3B (36 layers): creative=L14 (0.389), technical=L18 (0.500).
- 14B (48 layers): creative=L19 (0.396), technical=L24 (0.500).

Override with ``--creative-layer / --technical-layer`` if a downstream sweep
finds a better operating point.

Usage on a RunPod A100 80GB pod::

    cd /workspace/BrainstormLM
    export HF_TOKEN=hf_...
    uv run python scripts/extract_14b_vectors.py
    # ~10-15 min wall-clock for 32 pairs × 7 framings on A100 80GB
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from brainstormlm.conceptial.engine.vector_purify import (  # noqa: E402
    hyper_purify_vector,
)


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _render_pair(tokenizer, system: str, user: str) -> str:
    """Apply the Qwen2.5 chat template ending at the assistant turn boundary."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _last_token_hiddens(model, tokenizer, text: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """Forward ``text``, return ``{layer_idx: last-token residual (cpu fp32)}``."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    out: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        v = outputs.hidden_states[layer_idx][0, -1, :].detach().to("cpu").float()
        out[layer_idx] = v
    # Free GPU memory eagerly — hidden_states tuple is big at 14B.
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _derive_caa_vector(
    *,
    model,
    tokenizer,
    pos_system: str,
    neg_system: str,
    prompts: list[str],
    layer_idx: int,
    desc: str,
) -> torch.Tensor:
    """Mean of (pos - neg) last-token residuals at ``layer_idx``."""
    deltas: list[torch.Tensor] = []
    for i, prompt in enumerate(prompts):
        pos_text = _render_pair(tokenizer, pos_system, prompt)
        neg_text = _render_pair(tokenizer, neg_system, prompt)
        pos_h = _last_token_hiddens(model, tokenizer, pos_text, [layer_idx])[layer_idx]
        neg_h = _last_token_hiddens(model, tokenizer, neg_text, [layer_idx])[layer_idx]
        deltas.append(pos_h - neg_h)
        if (i + 1) % 8 == 0:
            print(f"    [{desc}] L{layer_idx}: {i+1}/{len(prompts)} pairs")
    v = torch.stack(deltas, dim=0).mean(0)
    # L2-normalize to unit length (matches 3B artifact convention).
    v = v / (torch.linalg.norm(v) + 1e-12)
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "contrast_pairs_14b.json",
    )
    parser.add_argument("--creative-layer", type=int, default=19)
    parser.add_argument("--technical-layer", type=int, default=24)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "steering_14b",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Process only the first N prompts (for fast smoke tests).",
    )
    parser.add_argument(
        "--no-purify",
        action="store_true",
        help="Skip orthogonal-projection purification; save raw CAA vectors.",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    framings = cfg["framings"]
    prompts: list[str] = cfg["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    neg_banks_creative: list[str] = cfg["negative_banks_for_creative"]
    neg_banks_technical: list[str] = cfg["negative_banks_for_technical"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== 14B steering-vector extraction ===")
    print(f"Model:           {args.model_id}")
    print(f"Domain prompts:  {len(prompts)}")
    print(f"Creative layer:  L{args.creative_layer}")
    print(f"Technical layer: L{args.technical_layer}")
    print(f"Negative banks (creative):  {neg_banks_creative}")
    print(f"Negative banks (technical): {neg_banks_technical}")
    print(f"Purify:          {not args.no_purify}")
    print(f"Out dir:         {args.out_dir.relative_to(ROOT)}")

    device = _resolve_device()
    print(f"\nLoading {args.model_id} on {device} … (~120s on A100)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    # ── Raw CAA vectors (positive − generic) ──────────────────────────────────
    print("\n[1/3] Deriving creative CAA vector (positive vs generic) …")
    raw_creative = _derive_caa_vector(
        model=model,
        tokenizer=tokenizer,
        pos_system=framings["creative"],
        neg_system=framings["generic"],
        prompts=prompts,
        layer_idx=args.creative_layer,
        desc="creative",
    )

    print("\n[2/3] Deriving technical CAA vector (technical vs generic) …")
    raw_technical = _derive_caa_vector(
        model=model,
        tokenizer=tokenizer,
        pos_system=framings["technical"],
        neg_system=framings["generic"],
        prompts=prompts,
        layer_idx=args.technical_layer,
        desc="technical",
    )

    # ── Negative banks: each derived as (negative_mode − generic) ────────────
    print("\n[3/3] Deriving negative-bank vectors at both layers …")
    neg_vectors_creative: dict[str, torch.Tensor] = {}
    neg_vectors_technical: dict[str, torch.Tensor] = {}
    all_neg_modes = sorted(set(neg_banks_creative) | set(neg_banks_technical))
    for mode in all_neg_modes:
        print(f"\n  Negative '{mode}' @ L{args.creative_layer} and L{args.technical_layer} …")
        # One pass for each layer so we can keep memory low.
        if mode in neg_banks_creative:
            v_c = _derive_caa_vector(
                model=model,
                tokenizer=tokenizer,
                pos_system=framings[mode],
                neg_system=framings["generic"],
                prompts=prompts,
                layer_idx=args.creative_layer,
                desc=f"neg-{mode}-c",
            )
            neg_vectors_creative[mode] = v_c
        if mode in neg_banks_technical:
            v_t = _derive_caa_vector(
                model=model,
                tokenizer=tokenizer,
                pos_system=framings[mode],
                neg_system=framings["generic"],
                prompts=prompts,
                layer_idx=args.technical_layer,
                desc=f"neg-{mode}-t",
            )
            neg_vectors_technical[mode] = v_t

    # ── Hyper-purification (orthogonal projection against negative subspace) ──
    if args.no_purify:
        purified_creative = raw_creative
        purified_technical = raw_technical
        creative_report = None
        technical_report = None
    else:
        print("\n[4/4] Hyper-purifying creative vector …")
        purified_creative, creative_report = hyper_purify_vector(
            raw_creative,
            list(neg_vectors_creative.values()),
            min_source_cosine=0.15,
            max_negative_cosine=0.08,
        )
        print(
            f"  retained={creative_report.retained_source_cosine:.4f}  "
            f"neg_before={creative_report.max_abs_negative_cosine_before:.4f}  "
            f"neg_after={creative_report.max_abs_negative_cosine_after:.4f}  "
            f"rank_removed={creative_report.removed_rank}"
        )
        print("\n  Hyper-purifying technical vector …")
        # Technical purification adds the creative vector to the negative bank so
        # that "technical" doesn't smuggle in creative directionality.
        tech_negatives = list(neg_vectors_technical.values()) + [purified_creative.detach().clone()]
        purified_technical, technical_report = hyper_purify_vector(
            raw_technical,
            tech_negatives,
            min_source_cosine=0.15,
            max_negative_cosine=0.08,
        )
        print(
            f"  retained={technical_report.retained_source_cosine:.4f}  "
            f"neg_before={technical_report.max_abs_negative_cosine_before:.4f}  "
            f"neg_after={technical_report.max_abs_negative_cosine_after:.4f}  "
            f"rank_removed={technical_report.removed_rank}"
        )

    # ── Save artifacts ────────────────────────────────────────────────────────
    creative_path = (
        args.out_dir / f"hyper_creative_purified_vector_L{args.creative_layer}.pt"
    )
    technical_path = (
        args.out_dir / f"hyper_technical_purified_vector_L{args.technical_layer}.pt"
    )
    # NOTE: key is `layer_index` (not `layer`) to match load_steering_checkpoint's contract.
    torch.save(
        {"vector": purified_creative.float(), "layer_index": int(args.creative_layer)},
        creative_path,
    )
    torch.save(
        {"vector": purified_technical.float(), "layer_index": int(args.technical_layer)},
        technical_path,
    )

    def _manifest(
        path: Path,
        layer: int,
        bank: str,
        report,
        neg_modes: list[str],
        purified: bool,
    ) -> dict:
        m: dict = {
            "checkpoint_path": str(path),
            "layer_index": layer,
            "model_id": args.model_id,
            "bank": bank,
            "hyper_purified": purified,
            "derived_at": _utc_now_iso(),
            "n_contrast_pairs": len(prompts),
            "normalization": "l2_unit",
            "storage_dtype": "float32",
            "framings": {
                "positive": framings["creative" if bank.endswith("creative_purified") else "technical"],
                "neutral": framings["generic"],
                "negatives": {mode: framings[mode] for mode in neg_modes},
            },
        }
        if report is not None:
            m["hyper_purify_report"] = {
                "retained_source_cosine": report.retained_source_cosine,
                "max_abs_negative_cosine_before": report.max_abs_negative_cosine_before,
                "max_abs_negative_cosine_after": report.max_abs_negative_cosine_after,
                "removed_rank": report.removed_rank,
                "n_negative_vectors": report.n_negative_vectors,
                "singular_values": report.singular_values,
            }
        return m

    creative_manifest = _manifest(
        creative_path,
        args.creative_layer,
        "hyper_creative_purified",
        creative_report,
        neg_banks_creative,
        not args.no_purify,
    )
    technical_manifest = _manifest(
        technical_path,
        args.technical_layer,
        "hyper_technical_purified",
        technical_report,
        neg_banks_technical,
        not args.no_purify,
    )
    (args.out_dir / f"hyper_creative_purified_vector_L{args.creative_layer}_manifest.json").write_text(
        json.dumps(creative_manifest, indent=2)
    )
    (args.out_dir / f"hyper_technical_purified_vector_L{args.technical_layer}_manifest.json").write_text(
        json.dumps(technical_manifest, indent=2)
    )

    print(f"\nWrote {creative_path.relative_to(ROOT)}")
    print(f"Wrote {technical_path.relative_to(ROOT)}")
    print("\nNext step (locally):")
    print(f"  scp -r root@<pod-ip>:/workspace/BrainstormLM/{args.out_dir.relative_to(ROOT)}/ \\")
    print("      ~/Desktop/Projects/BrainstormLM/data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
