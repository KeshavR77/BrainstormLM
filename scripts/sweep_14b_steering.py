#!/usr/bin/env python3
"""Sweep creative layer × alpha to find the best Conceptial 14B operating point.

Runs on a RunPod A100 80GB pod, AFTER ``scripts/extract_14b_vectors.py``.
Resolved the two open caveats in the 14B port (run 2026-05-18):

1. ``creative_start_alpha`` — sweep selected α₀=1.5 (down from 3B's 2.62).
2. ``creative_layer`` — sweep selected L17 (depth 0.354), beating the
   depth-scaled guess of L19 (Vendi 3.32 vs 2.26).

The sweep:

- **Step A — Multi-layer extraction (one forward pass per contrast pair).**
  Derives creative vectors at L17, L19, L21, L23 in the same forward pass
  the single-layer extractor would have used for one. Persists each to
  ``data/steering_14b/sweep/hyper_creative_purified_vector_L{layer}.pt``
  (+ manifest).
- **Step B — Grid sweep over the new logo brief.** For each
  (creative_layer × creative_alpha) cell, generate N=5 samples with
  iterative anti-mode + memory + min-p sampling.
- **Step C — Score each cell.** Vendi score on the 5 outputs (semantic
  diversity), near-dup pair rate, plus a coherence sanity filter (length
  ≥ 80 chars, repeated-line fraction < 0.3, no degenerate token runs).
- **Step D — Pick the winner.** Maximize Vendi subject to passing the
  coherence filter for ≥ 4/5 samples. Write
  ``data/steering_14b/sweep/chosen_config.json`` — the downstream
  ``runpod_batch_14b.py`` reads this and overrides the defaults in
  ``config/conceptial_14b.json``.

Defaults give 4 layers × 4 alphas × 5 samples = 80 samples, ~25-35 min on A100.

Usage::

    cd /workspace/BrainstormLM
    export HF_TOKEN=hf_...
    uv run python scripts/sweep_14b_steering.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from brainstormlm.conceptial.anti_mode import (  # noqa: E402
    extract_summary,
    format_anti_mode,
)
from brainstormlm.conceptial.engine.config import SteeringConfig  # noqa: E402
from brainstormlm.conceptial.engine.pipeline import (  # noqa: E402
    SteeringEngine,
    load_steering_checkpoint,
)
from brainstormlm.conceptial.engine.steering_presets import (  # noqa: E402
    apply_v10_parametric_preset,
)
from brainstormlm.conceptial.engine.vector_purify import (  # noqa: E402
    hyper_purify_vector,
    purify_vector,
)


DEFAULT_CREATIVE_LAYERS = [17, 19, 21, 23]
DEFAULT_CREATIVE_ALPHAS = [1.5, 2.0, 2.62, 3.0]
DEFAULT_TECHNICAL_LAYER = 24
DEFAULT_TECHNICAL_ALPHA = 0.86


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _render(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _last_token_hidden_at_layers(model, tokenizer, text: str, layers: list[int]) -> dict[int, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    result = {L: out.hidden_states[L][0, -1, :].detach().to("cpu").float() for L in layers}
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def derive_multi_layer_creative(
    *,
    model,
    tokenizer,
    framings: dict[str, str],
    prompts: list[str],
    layers: list[int],
    negative_modes: list[str],
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """One forward pass per (system, prompt); capture all candidate layers.

    Returns:
      raw_creative[layer]  — mean-difference (creative - generic)
      neg_vectors[layer][mode] — mean-difference (mode - generic)
    """
    creative_acc: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    neg_acc: dict[int, dict[str, list[torch.Tensor]]] = {
        L: {m: [] for m in negative_modes} for L in layers
    }

    for i, p in enumerate(prompts):
        # neutral baseline
        gen_h = _last_token_hidden_at_layers(
            model, tokenizer, _render(tokenizer, framings["generic"], p), layers
        )
        # positive
        cre_h = _last_token_hidden_at_layers(
            model, tokenizer, _render(tokenizer, framings["creative"], p), layers
        )
        for L in layers:
            creative_acc[L].append(cre_h[L] - gen_h[L])
        # negatives
        for mode in negative_modes:
            mode_h = _last_token_hidden_at_layers(
                model, tokenizer, _render(tokenizer, framings[mode], p), layers
            )
            for L in layers:
                neg_acc[L][mode].append(mode_h[L] - gen_h[L])
        if (i + 1) % 8 == 0:
            print(f"    multi-layer extract: {i+1}/{len(prompts)} prompts")

    raw_creative: dict[int, torch.Tensor] = {}
    neg_vectors: dict[int, dict[str, torch.Tensor]] = {}
    for L in layers:
        v = torch.stack(creative_acc[L], dim=0).mean(0)
        raw_creative[L] = v / (torch.linalg.norm(v) + 1e-12)
        neg_vectors[L] = {}
        for mode in negative_modes:
            m = torch.stack(neg_acc[L][mode], dim=0).mean(0)
            neg_vectors[L][mode] = m / (torch.linalg.norm(m) + 1e-12)
    return raw_creative, neg_vectors


def _save_creative_vector(
    out_dir: Path,
    layer: int,
    vector: torch.Tensor,
    report,
    framings: dict,
    negative_modes: list[str],
    n_pairs: int,
    model_id: str,
) -> Path:
    out_path = out_dir / f"hyper_creative_purified_vector_L{layer}.pt"
    torch.save({"vector": vector.float(), "layer_index": int(layer)}, out_path)
    manifest = {
        "checkpoint_path": str(out_path),
        "layer_index": int(layer),
        "model_id": model_id,
        "bank": "hyper_creative_purified",
        "hyper_purified": True,
        "derived_at": _utc_now(),
        "n_contrast_pairs": int(n_pairs),
        "normalization": "l2_unit",
        "storage_dtype": "float32",
        "framings": {
            "positive": framings["creative"],
            "neutral": framings["generic"],
            "negatives": {m: framings[m] for m in negative_modes},
        },
    }
    if report is not None:
        manifest["hyper_purify_report"] = {
            "retained_source_cosine": report.retained_source_cosine,
            "max_abs_negative_cosine_before": report.max_abs_negative_cosine_before,
            "max_abs_negative_cosine_after": report.max_abs_negative_cosine_after,
            "removed_rank": report.removed_rank,
            "n_negative_vectors": report.n_negative_vectors,
            "singular_values": report.singular_values,
        }
    (out_dir / f"hyper_creative_purified_vector_L{layer}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return out_path


# ─── Metrics ────────────────────────────────────────────────────────────────


def _coherent(text: str) -> bool:
    """Cheap coherence filter: not too short, no degenerate repetition."""
    t = text.strip()
    if len(t) < 80:
        return False
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not lines:
        return False
    if len(lines) >= 2:
        unique_lines = len(set(lines))
        if unique_lines / len(lines) < 0.7:
            return False
    # Degenerate token runs: same word repeating
    words = re.findall(r"\w+", t.lower())
    if len(words) >= 10:
        triples = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if triples:
            most_common_frac = max(triples.count(tr) for tr in set(triples)) / len(triples)
            if most_common_frac > 0.15:
                return False
    return True


def _vendi_score(texts: list[str]) -> float:
    """Lightweight Vendi: exp(entropy of normalized singular values of similarity matrix).

    Uses MiniLM-L6-v2 embeddings.
    """
    if len(texts) < 2:
        return 1.0
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # Similarity matrix → eigenvalues of K/n
    import numpy as np
    K = embs @ embs.T
    n = K.shape[0]
    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals = eigvals / eigvals.sum()
    H = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(H))


def _near_dup_rate(texts: list[str], threshold: float = 0.92) -> float:
    if len(texts) < 2:
        return 0.0
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sim = embs @ embs.T
    n = sim.shape[0]
    pairs = n * (n - 1) // 2
    dup_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                dup_pairs += 1
    return dup_pairs / pairs if pairs > 0 else 0.0


# ─── Sweep cell generator ───────────────────────────────────────────────────


def generate_cell(
    *,
    engine: SteeringEngine,
    brief: str,
    n: int,
    creative_vec_path: Path,
    technical_vec_path: Path,
    creative_layer: int,
    technical_layer: int,
    creative_alpha: float,
    technical_alpha: float,
    schedule: dict,
    base_temp: float,
    temp_jitter: float,
    max_new_tokens: int,
) -> list[dict]:
    """Generate N samples with a specific (layer, alpha) cell. Applies iterative anti-mode."""
    # Reset preset with cell parameters
    apply_v10_parametric_preset(
        engine,
        creative_vec_path.parent,
        "v17",
        creative_vector_path=creative_vec_path,
        technical_vector_path=technical_vec_path,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        steering_math=str(schedule.get("steering_math", "additive")),
        creative_start_alpha=float(creative_alpha),
        exp_zero_by_step=int(schedule["exp_zero_by_step"]),
        technical_start_alpha=float(technical_alpha),
        sigmoid_mid_step=float(schedule["sigmoid_mid_step"]),
        sigmoid_steepness=float(schedule["sigmoid_steepness"]),
    )

    prior_summaries: list[str] = []
    prior_first_token_hiddens: list[torch.Tensor] = []
    outputs: list[dict] = []
    for k in range(n):
        if prior_first_token_hiddens:
            v_bundle = load_steering_checkpoint(creative_vec_path, map_location="cpu")
            mean_h = torch.stack(
                [h.float().reshape(-1) for h in prior_first_token_hiddens], dim=0
            ).mean(0)
            v_iter = purify_vector(v_bundle["vector"].float(), [mean_h])
            p_param = next(engine.model.parameters())
            v_iter = v_iter.to(device=p_param.device, dtype=p_param.dtype)
            current = dict(engine.steering_vectors or {})
            current[creative_layer] = v_iter
            engine.set_steering_vectors(current)

        prompt = format_anti_mode(
            user_text=brief,
            prior_summaries=prior_summaries,
            exemplars=None,
            output_mode="generic",
        )
        temp = float(base_temp + (((k * 0.61803) % 1.0 - 0.5) * temp_jitter * 2))
        first_token_buf: list[torch.Tensor] = []
        text = engine.complete(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=temp,
            do_sample=True,
            do_steering=True,
            use_ghost_rejection=False,
            capture_first_decode_residual=(creative_layer, first_token_buf),
        )
        text = text.strip()
        if first_token_buf:
            prior_first_token_hiddens.append(first_token_buf[0].detach().to("cpu").float())
        summary = extract_summary(text) or text.split("\n")[0][:80]
        prior_summaries.append(summary)
        outputs.append({"k": k, "text": text, "summary": summary, "temperature": temp})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--corpius-config",
        type=Path,
        default=ROOT / "config" / "conceptial_14b.json",
    )
    parser.add_argument(
        "--contrast-config",
        type=Path,
        default=ROOT / "config" / "contrast_pairs_14b.json",
    )
    parser.add_argument(
        "--brief-id",
        default="logo-stanford-engineering",
        help="Prompt id from config/benchmark_creative_briefs.json to sweep on.",
    )
    parser.add_argument(
        "--brief-config",
        type=Path,
        default=ROOT / "config" / "benchmark_creative_briefs.json",
    )
    parser.add_argument(
        "--creative-layers",
        default=",".join(str(L) for L in DEFAULT_CREATIVE_LAYERS),
        help="Comma-separated creative-layer candidates (default: 17,19,21,23).",
    )
    parser.add_argument(
        "--creative-alphas",
        default=",".join(str(a) for a in DEFAULT_CREATIVE_ALPHAS),
        help="Comma-separated creative-alpha candidates (default: 1.5,2.0,2.62,3.0).",
    )
    parser.add_argument("--n", type=int, default=5, help="Samples per cell.")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--base-temp", type=float, default=0.8)
    parser.add_argument("--temp-jitter", type=float, default=0.05)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "steering_14b" / "sweep",
    )
    parser.add_argument(
        "--coherence-min-pass",
        type=int,
        default=4,
        help="Min coherent outputs out of N for a cell to qualify as winner.",
    )
    parser.add_argument(
        "--limit-prompts-for-extraction",
        type=int,
        default=None,
        help="Use only the first N contrast prompts for multi-layer extraction (smoke test).",
    )
    args = parser.parse_args()

    layers = [int(x) for x in args.creative_layers.split(",")]
    alphas = [float(x) for x in args.creative_alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load configs ─────────────────────────────────────────────────────────
    corpius = json.loads(args.corpius_config.read_text())
    contrast = json.loads(args.contrast_config.read_text())
    briefs = json.loads(args.brief_config.read_text())
    framings = contrast["framings"]
    contrast_prompts = contrast["prompts"]
    if args.limit_prompts_for_extraction:
        contrast_prompts = contrast_prompts[: args.limit_prompts_for_extraction]
    neg_modes = contrast["negative_banks_for_creative"]

    brief_text = next(
        (p["brief"] for p in briefs["prompts"] if p["id"] == args.brief_id), None
    )
    if brief_text is None:
        raise SystemExit(f"Unknown brief id: {args.brief_id}")

    technical_layer = int(corpius["layers"]["technical_layer"])
    technical_vec_path = ROOT / corpius["steering_vectors"]["technical"]
    if not technical_vec_path.exists():
        raise SystemExit(
            f"Missing technical vector at {technical_vec_path}. Run extract_14b_vectors.py first."
        )
    schedule = corpius["schedule"]

    print("\n=== 14B steering sweep ===")
    print(f"Model:              {args.model_id}")
    print(f"Brief id:           {args.brief_id}")
    print(f"Brief text:         {brief_text!r}")
    print(f"Creative layers:    {layers}")
    print(f"Creative alphas:    {alphas}")
    print(f"Cells × samples:    {len(layers)*len(alphas)} × {args.n} = {len(layers)*len(alphas)*args.n}")
    print(f"Technical layer α:  L{technical_layer} α={DEFAULT_TECHNICAL_ALPHA} (fixed)")
    print(f"Contrast pairs:     {len(contrast_prompts)}")

    # ── Load model ───────────────────────────────────────────────────────────
    device = _resolve_device()
    print(f"\nLoading {args.model_id} on {device} … (~120s on A100)")
    cfg_engine = SteeringConfig(model_id=args.model_id)
    cfg_engine.preferred_device = device
    engine = SteeringEngine(cfg_engine)
    engine.load()
    print("Model loaded.")
    model = engine.model
    tokenizer = engine.tokenizer

    # ── Step A — Multi-layer extraction ──────────────────────────────────────
    print("\n[A] Deriving creative + negative vectors at all candidate layers …")
    t_extract = time.time()
    raw_creative, neg_by_layer = derive_multi_layer_creative(
        model=model,
        tokenizer=tokenizer,
        framings=framings,
        prompts=contrast_prompts,
        layers=layers,
        negative_modes=neg_modes,
    )
    print(f"    extracted in {time.time() - t_extract:.1f}s")

    # Hyper-purify at each candidate layer and save.
    creative_vec_paths: dict[int, Path] = {}
    purify_reports: dict[int, dict] = {}
    for L in layers:
        print(f"  Purifying creative @ L{L} …")
        purified, report = hyper_purify_vector(
            raw_creative[L],
            list(neg_by_layer[L].values()),
            min_source_cosine=0.15,
            max_negative_cosine=0.08,
        )
        creative_vec_paths[L] = _save_creative_vector(
            args.out_dir,
            L,
            purified,
            report,
            framings,
            neg_modes,
            len(contrast_prompts),
            args.model_id,
        )
        purify_reports[L] = {
            "retained": report.retained_source_cosine,
            "neg_after": report.max_abs_negative_cosine_after,
            "rank_removed": report.removed_rank,
        }
        print(
            f"    retained={report.retained_source_cosine:.4f} "
            f"neg_after={report.max_abs_negative_cosine_after:.4f} "
            f"rank_removed={report.removed_rank}"
        )

    # ── Step B + C — Grid sweep ──────────────────────────────────────────────
    print("\n[B/C] Sweeping cells on the brief …")
    cell_results: list[dict] = []
    for L in layers:
        for alpha in alphas:
            t_cell = time.time()
            outputs = generate_cell(
                engine=engine,
                brief=brief_text,
                n=args.n,
                creative_vec_path=creative_vec_paths[L],
                technical_vec_path=technical_vec_path,
                creative_layer=L,
                technical_layer=technical_layer,
                creative_alpha=alpha,
                technical_alpha=DEFAULT_TECHNICAL_ALPHA,
                schedule=schedule,
                base_temp=args.base_temp,
                temp_jitter=args.temp_jitter,
                max_new_tokens=args.max_new_tokens,
            )
            texts = [o["text"] for o in outputs]
            coherent_flags = [_coherent(t) for t in texts]
            n_coherent = sum(coherent_flags)
            vendi = _vendi_score(texts)
            near_dup = _near_dup_rate(texts)
            cell = {
                "creative_layer": L,
                "creative_alpha": alpha,
                "vendi": vendi,
                "near_dup": near_dup,
                "n_coherent": n_coherent,
                "n_total": len(texts),
                "passes_coherence": n_coherent >= args.coherence_min_pass,
                "elapsed_s": time.time() - t_cell,
                "outputs": outputs,
                "coherent_flags": coherent_flags,
            }
            cell_results.append(cell)
            print(
                f"  L{L} α={alpha:>4.2f}: Vendi={vendi:.3f}  near_dup={near_dup:.3f}  "
                f"coherent={n_coherent}/{len(texts)}  "
                f"{'PASS' if cell['passes_coherence'] else 'FAIL'}  "
                f"({cell['elapsed_s']:.1f}s)"
            )

    # ── Step D — Pick winner ─────────────────────────────────────────────────
    qualifying = [c for c in cell_results if c["passes_coherence"]]
    if not qualifying:
        print("\nWARNING: no cell passed the coherence filter. Falling back to highest-Vendi cell.")
        winner = max(cell_results, key=lambda c: c["vendi"])
    else:
        winner = max(qualifying, key=lambda c: c["vendi"])

    print(f"\nWinner: L{winner['creative_layer']} α={winner['creative_alpha']:.2f}  "
          f"Vendi={winner['vendi']:.3f}  near_dup={winner['near_dup']:.3f}  "
          f"coherent={winner['n_coherent']}/{winner['n_total']}")

    # ── Persist results ──────────────────────────────────────────────────────
    sweep_results_path = args.out_dir / "sweep_results.json"
    sweep_results_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "brief_id": args.brief_id,
                "brief": brief_text,
                "n_per_cell": args.n,
                "purify_reports": purify_reports,
                "cells": cell_results,
                "winner": {
                    "creative_layer": winner["creative_layer"],
                    "creative_alpha": winner["creative_alpha"],
                    "vendi": winner["vendi"],
                    "near_dup": winner["near_dup"],
                    "n_coherent": winner["n_coherent"],
                },
                "generated_at": _utc_now(),
            },
            indent=2,
        )
    )
    chosen_path = args.out_dir / "chosen_config.json"
    chosen_path.write_text(
        json.dumps(
            {
                "creative_layer": winner["creative_layer"],
                "creative_alpha": winner["creative_alpha"],
                "creative_vector_path": str(
                    creative_vec_paths[winner["creative_layer"]].relative_to(ROOT)
                ),
                "technical_layer": technical_layer,
                "technical_alpha": DEFAULT_TECHNICAL_ALPHA,
                "technical_vector_path": str(technical_vec_path.relative_to(ROOT)),
                "selected_from": {
                    "creative_layers": layers,
                    "creative_alphas": alphas,
                    "n_per_cell": args.n,
                    "brief_id": args.brief_id,
                },
                "generated_at": _utc_now(),
            },
            indent=2,
        )
    )
    print(f"\nWrote {sweep_results_path.relative_to(ROOT)}")
    print(f"Wrote {chosen_path.relative_to(ROOT)}")
    print("\nNext: `uv run python scripts/runpod_batch_14b.py --n 8` will auto-pick this config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
