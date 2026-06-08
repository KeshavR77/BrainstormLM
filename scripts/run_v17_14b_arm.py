#!/usr/bin/env python
"""Corrected v17 steering test on 14B — uses the REAL ``conceptial`` engine, not a strawman.

The validation harness (``validate_diversity.py``) tested an arm labelled ``ortho_L17`` that
applied a SINGLE layer (L17) at a CONSTANT alpha with additive rescale. That is NOT how the
product steers. Production steering (``conceptial/engine``) is **dual-layer with scheduled alpha**:
a creative vector on an exp-decay ramp + a technical vector on a sigmoid ramp, spherical math.

This script runs that real mechanism on the 14B via :class:`CloudCorpius` (the engine's own
14B-on-RunPod path), using the v17 schedule and the 14B vectors that actually exist
(creative L17 + technical L24). It generates **steered vs unsteered on the same model and briefs**
so the comparison isolates the steering effect, and writes generations in the harness format so
they score with ``results/pod_run_2026-06-04/recompute.py`` + the judge.

Modes
-----
``--verify-only``  CPU only. Proves the real v17 dual-layer config is built from the 14B vectors
                   (loads the checkpoints + runs the real preset's config-build; no model load).
(default)          Loads 14B via CloudCorpius and generates. Needs a CUDA GPU (~28GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_STEERING_DIR = _REPO / "data" / "steering_14b"
_CREATIVE = _STEERING_DIR / "hyper_creative_purified_vector_L17.pt"
_TECHNICAL = _STEERING_DIR / "hyper_technical_purified_vector_L24.pt"
_CREATIVE_LAYER = 17
_TECHNICAL_LAYER = 24

# v17 schedule, verbatim from steering_presets.V10_PARAMETRIC_SCHEDULES["v17"]
# = (creative_start_alpha, exp_zero_by_step, technical_start_alpha, sigmoid_mid_step, sigmoid_steepness)
_V17 = {
    "creative_start_alpha": 2.05,
    "exp_zero_by_step": 132,
    "technical_start_alpha": 0.75,
    "sigmoid_mid_step": 58.0,
    "sigmoid_steepness": 0.11,
    "steering_math": "spherical",
}


def verify_config() -> int:
    """Prove (on CPU, no model) that the real v17 dual-layer config is built from the 14B vectors."""
    import brainstormlm.conceptial.engine.steering_presets as sp
    from brainstormlm.conceptial.engine.config import SteeringConfig
    from brainstormlm.conceptial.engine.pipeline import load_steering_checkpoint

    print("== verifying the REAL v17 dual-layer steering wiring (CPU; no model load) ==")
    c = load_steering_checkpoint(_CREATIVE)
    t = load_steering_checkpoint(_TECHNICAL)
    print(f"creative  : layer_index={int(c['layer_index'])} dim={tuple(c['vector'].shape)}  ({_CREATIVE.name})")
    print(f"technical : layer_index={int(t['layer_index'])} dim={tuple(t['vector'].shape)}  ({_TECHNICAL.name})")
    assert int(c["layer_index"]) == _CREATIVE_LAYER, f"creative layer {c['layer_index']} != {_CREATIVE_LAYER}"
    assert int(t["layer_index"]) == _TECHNICAL_LAYER, f"technical layer {t['layer_index']} != {_TECHNICAL_LAYER}"

    class _StubEngine:
        def __init__(self) -> None:
            self.config = SteeringConfig(model_id="Qwen/Qwen2.5-14B-Instruct")

    stub = _StubEngine()
    # Run the REAL preset's config-build; neutralise only the model-dependent apply step.
    orig_apply = sp.apply_layer_injections
    sp.apply_layer_injections = lambda *a, **k: None
    try:
        sp.apply_v8_idea_forward_phase_shift(
            stub,
            _STEERING_DIR,
            creative_vector_path=_CREATIVE,
            technical_vector_path=_TECHNICAL,
            creative_layer=_CREATIVE_LAYER,
            technical_layer=_TECHNICAL_LAYER,
            **_V17,
        )
    finally:
        sp.apply_layer_injections = orig_apply

    inj = stub.config.layer_injections
    print("\nlayer_injections the engine WILL apply:")
    for entry in inj:
        print("  ", json.dumps(entry))

    assert len(inj) == 2, f"expected 2 layers (dual-injection), got {len(inj)}"
    creative = next(e for e in inj if int(e["layer"]) == _CREATIVE_LAYER)
    technical = next(e for e in inj if int(e["layer"]) == _TECHNICAL_LAYER)
    assert creative["alpha_schedule"] == "exp_decay", creative["alpha_schedule"]
    assert technical["alpha_schedule"] == "sigmoid_ramp", technical["alpha_schedule"]
    assert abs(float(creative["start_alpha"]) - 2.05) < 1e-9
    assert int(creative["exp_zero_by_step"]) == 132
    assert abs(float(technical["start_alpha"]) - 0.75) < 1e-9
    assert abs(float(technical["sigmoid_mid_step"]) - 58.0) < 1e-9
    assert abs(float(technical["sigmoid_steepness"]) - 0.11) < 1e-9
    assert creative["steering_math"] == "spherical"
    assert technical["steering_math"] == "spherical"
    print(
        "\nVERIFIED: real v17 dual-layer steering — L17 creative exp_decay(alpha=2.05 -> 0 by step 132) "
        "+ L24 technical sigmoid_ramp(alpha=0.75, mid=58, steep=0.11), spherical math.\n"
        "This is the production mechanism, NOT the single-layer constant-alpha strawman that "
        "validate_diversity.py's ortho_L17 arm used."
    )
    return 0


def run_generation(*, n_briefs: int, k: int, max_new_tokens: int, temperature: float, out: str) -> int:
    """Load 14B via CloudCorpius (v17, L17+L24) and generate steered vs unsteered. Needs a CUDA GPU."""
    import torch

    sys.path.insert(0, str(_REPO / "scripts"))
    from validate_diversity import BUILTIN_BRIEFS  # the same builtin briefs as the FP16 sweep

    from brainstormlm.conceptial.engine.cloud.cloud_corpius import CloudCorpius

    if not torch.cuda.is_available():
        print("ERROR: generation needs a CUDA GPU (~28GB for 14B). Use --verify-only on CPU.", file=sys.stderr)
        return 2

    briefs = BUILTIN_BRIEFS[:n_briefs]
    cc = CloudCorpius(
        _STEERING_DIR,
        _CREATIVE,
        _TECHNICAL,
        model_id="Qwen/Qwen2.5-14B-Instruct",
        creative_layer=_CREATIVE_LAYER,
        technical_layer=_TECHNICAL_LAYER,
        device_map=None,
        **_V17,
    )
    engine = cc.engine  # forces model load + applies the v17 dual-layer steering
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
        description="Corrected v17 steering test on 14B (real conceptial engine, dual-layer L17+L24)."
    )
    ap.add_argument("--verify-only", action="store_true", help="CPU: prove the real v17 config is built; no model load.")
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
