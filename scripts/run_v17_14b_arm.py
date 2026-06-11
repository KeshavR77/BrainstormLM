#!/usr/bin/env python
"""Corrected v17 steering test on 14B — uses the REAL production config, not a strawman.

The validation harness (``validate_diversity.py``) tested an arm labelled ``ortho_L17`` that applied
a SINGLE layer (L17) at a CONSTANT alpha. That is NOT how the product steers, and its verdict does
not stand. Production 14B steering is **dual-layer with scheduled alpha**, defined authoritatively by
``config/conceptial_14b.json`` ("Corpius 2 14B"):

    creative L17  exp_decay   (start_alpha, exp_zero_by_step)
    technical L24 sigmoid_ramp (start_alpha, sigmoid_mid_step, sigmoid_steepness)
    steering_math = additive

This script reads that config file directly (so it can never drift from what ships), applies it via
:class:`CloudCorpius` (the engine's own 14B path), and generates **steered vs unsteered on the same
model + briefs** so the comparison isolates the steering effect. Generations are written in the
harness format for scoring with ``results/pod_run_2026-06-04/recompute.py`` + the judge.

Modes
-----
``--verify-only``  CPU only. Proves the built dual-layer config matches ``config/conceptial_14b.json``
                   (loads checkpoints + runs the real preset's config-build; no model load).
(default)          Loads 14B via CloudCorpius and generates. Needs a CUDA GPU (~28GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_CONFIG = _REPO / "config" / "conceptial_14b.json"


def _load_prod_config() -> dict[str, Any]:
    """Authoritative 14B steering config (``config/conceptial_14b.json``). No hardcoded schedule."""
    cfg = json.loads(_CONFIG.read_text())
    sched = cfg["schedule"]
    return {
        "creative_vector": _REPO / cfg["steering_vectors"]["creative"],
        "technical_vector": _REPO / cfg["steering_vectors"]["technical"],
        "creative_layer": int(cfg["layers"]["creative_layer"]),
        "technical_layer": int(cfg["layers"]["technical_layer"]),
        "schedule": {
            "creative_start_alpha": float(sched["creative_start_alpha"]),
            "exp_zero_by_step": int(sched["exp_zero_by_step"]),
            "technical_start_alpha": float(sched["technical_start_alpha"]),
            "sigmoid_mid_step": float(sched["sigmoid_mid_step"]),
            "sigmoid_steepness": float(sched["sigmoid_steepness"]),
            "steering_math": str(sched["steering_math"]),
        },
    }


def verify_config() -> int:
    """Prove (CPU, no model) the built dual-layer config matches config/conceptial_14b.json."""
    import brainstormlm.conceptial.engine.steering_presets as sp
    from brainstormlm.conceptial.engine.config import SteeringConfig
    from brainstormlm.conceptial.engine.pipeline import load_steering_checkpoint

    pc = _load_prod_config()
    s = pc["schedule"]
    print("== verifying against config/conceptial_14b.json (authoritative production config) ==")
    print("schedule:", json.dumps(s))
    c = load_steering_checkpoint(pc["creative_vector"])
    t = load_steering_checkpoint(pc["technical_vector"])
    print(f"creative  : layer_index={int(c['layer_index'])} dim={tuple(c['vector'].shape)}  ({pc['creative_vector'].name})")
    print(f"technical : layer_index={int(t['layer_index'])} dim={tuple(t['vector'].shape)}  ({pc['technical_vector'].name})")
    assert int(c["layer_index"]) == pc["creative_layer"], f"creative layer {c['layer_index']} != {pc['creative_layer']}"
    assert int(t["layer_index"]) == pc["technical_layer"], f"technical layer {t['layer_index']} != {pc['technical_layer']}"

    class _StubEngine:
        def __init__(self) -> None:
            self.config = SteeringConfig(model_id="Qwen/Qwen2.5-14B-Instruct")

    stub = _StubEngine()
    orig_apply = sp.apply_layer_injections
    sp.apply_layer_injections = lambda *a, **k: None  # neutralise only the model-dependent apply step
    try:
        sp.apply_v8_idea_forward_phase_shift(
            stub,
            pc["creative_vector"].parent,
            creative_vector_path=pc["creative_vector"],
            technical_vector_path=pc["technical_vector"],
            creative_layer=pc["creative_layer"],
            technical_layer=pc["technical_layer"],
            **s,
        )
    finally:
        sp.apply_layer_injections = orig_apply

    inj = stub.config.layer_injections
    print("\nlayer_injections the engine WILL apply:")
    for entry in inj:
        print("  ", json.dumps(entry))

    assert len(inj) == 2, f"expected 2 layers (dual-injection), got {len(inj)}"
    creative = next(e for e in inj if int(e["layer"]) == pc["creative_layer"])
    technical = next(e for e in inj if int(e["layer"]) == pc["technical_layer"])
    assert creative["alpha_schedule"] == "exp_decay", creative["alpha_schedule"]
    assert technical["alpha_schedule"] == "sigmoid_ramp", technical["alpha_schedule"]
    assert abs(float(creative["start_alpha"]) - s["creative_start_alpha"]) < 1e-9
    assert int(creative["exp_zero_by_step"]) == s["exp_zero_by_step"]
    assert abs(float(technical["start_alpha"]) - s["technical_start_alpha"]) < 1e-9
    assert abs(float(technical["sigmoid_mid_step"]) - s["sigmoid_mid_step"]) < 1e-9
    assert abs(float(technical["sigmoid_steepness"]) - s["sigmoid_steepness"]) < 1e-9
    assert creative["steering_math"] == s["steering_math"], creative["steering_math"]
    assert technical["steering_math"] == s["steering_math"], technical["steering_math"]
    print(
        f"\nVERIFIED: matches config/conceptial_14b.json — "
        f"L{pc['creative_layer']} creative exp_decay(alpha={s['creative_start_alpha']} -> 0 by step "
        f"{s['exp_zero_by_step']}) + L{pc['technical_layer']} technical sigmoid_ramp"
        f"(alpha={s['technical_start_alpha']}, mid={s['sigmoid_mid_step']}, steep={s['sigmoid_steepness']}), "
        f"{s['steering_math']} math.\nThis is the production 14B mechanism, NOT the single-layer "
        f"constant-alpha strawman that validate_diversity.py's ortho_L17 arm used."
    )
    return 0


def run_generation(*, n_briefs: int, k: int, max_new_tokens: int, temperature: float, out: str) -> int:
    """Load 14B via CloudCorpius (config/conceptial_14b.json) and generate steered vs unsteered."""
    import torch

    sys.path.insert(0, str(_REPO / "scripts"))
    from validate_diversity import BUILTIN_BRIEFS  # same builtin briefs as the FP16 sweep

    from brainstormlm.conceptial.engine.cloud.cloud_corpius import CloudCorpius

    if not torch.cuda.is_available():
        print("ERROR: generation needs a CUDA GPU (~28GB for 14B). Use --verify-only on CPU.", file=sys.stderr)
        return 2

    pc = _load_prod_config()
    briefs = BUILTIN_BRIEFS[:n_briefs]
    cc = CloudCorpius(
        pc["creative_vector"].parent,
        pc["creative_vector"],
        pc["technical_vector"],
        model_id="Qwen/Qwen2.5-14B-Instruct",
        creative_layer=pc["creative_layer"],
        technical_layer=pc["technical_layer"],
        device_map=None,
        **pc["schedule"],
    )
    engine = cc.engine  # forces model load + applies the dual-layer steering
    inj = engine.config.layer_injections
    print("APPLIED layer_injections:", json.dumps(inj), flush=True)
    if len(inj) != 2:
        print("ABORT: not dual-layer — this would be a strawman, refusing to generate.", file=sys.stderr)
        return 3

    def _seeded_complete(prompt: str, seed: int, *, steered: bool) -> str:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return engine.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            do_steering=steered,
        )

    steered: list[list[str]] = []
    unsteered: list[list[str]] = []
    for bi, brief in enumerate(briefs):
        steered.append([_seeded_complete(brief, 1000 * bi + j, steered=True) for j in range(k)])
        unsteered.append([_seeded_complete(brief, 1000 * bi + j, steered=False) for j in range(k)])
        print(f"brief {bi + 1}/{len(briefs)} done", flush=True)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"v17_steered": steered, "base_unsteered": unsteered}, fh, indent=2)
    print(f"saved {len(briefs)} briefs x {k} (steered + unsteered) -> {out}")
    print("score with results/pod_run_2026-06-04/recompute.py (NB-Distinct/Vendi) + the judge (quality).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Corrected v17 steering test on 14B (real config/conceptial_14b.json, dual-layer L17+L24)."
    )
    ap.add_argument("--verify-only", action="store_true", help="CPU: prove the config matches conceptial_14b.json; no model load.")
    ap.add_argument("--n-briefs", type=int, default=12, help="First N builtin briefs (default 12).")
    ap.add_argument("--k", type=int, default=8, help="Outputs per brief per arm (default 8).")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="results/validate_v17_14b.generations.json")
    args = ap.parse_args(argv)
    if args.verify_only:
        return verify_config()
    return run_generation(
        n_briefs=args.n_briefs,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
