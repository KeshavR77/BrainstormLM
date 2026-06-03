"""Real-model smoke for X1×X2: does orthogonalised-archetype per-sample steering fan a batch?

Runs the full steering wiring on a *small real* model (Qwen2.5-0.5B-Instruct via
``configs/qwen_0_5b_local.json``):

  1. build a tiny real archetype bank from captured creative-vs-dull activations (DiffMeans),
  2. fuse it with the orthonormal frame via ``build_orthogonal_archetype_dirs`` (the guarded
     "orthogonalise the bank" path),
  3. generate one steered continuation per orthogonalised mode with
     ``HFBackend.generate_steered_batch`` (greedy, so the *only* source of variation across the
     batch is the per-sample direction), against a zero-direction baseline batch,
  4. report set-level semantic spread (embedding mean-pairwise-cosine distance; raw Vendi if
     available) plus the orthogonalisation retention diagnostics.

    uv run --extra hf --extra embedding python scripts/smoke_semantic_spread.py \
        --config configs/qwen_0_5b_local.json

IMPORTANT — necessary, not sufficient. Steering vectors are non-identifiable (arXiv:2602.06801):
orthogonal *directions* do not guarantee distinct *outputs*. The spread numbers here are a
preliminary geometry check only. The fusion "earns its keep" only when it lifts Conditional-Vendi /
NoveltyBench-Distinct — and those are worktree-C deliverables (C2/C3) not yet built — so this smoke
asserts the mechanism, prints the spread, and explicitly does NOT claim success from spread alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from creativity_steering.config import CAPTURE_MEAN_RESPONSE, SamplingPreset, load_config
from creativity_steering.derivation.archetype_bank import ArchetypeBank
from creativity_steering.models.factory import build_backend
from creativity_steering.patching.steering import build_orthogonal_archetype_dirs
from creativity_steering.scoring import make_scorer

# A held-out brainstorm brief + a creative/dull contrast for the (tiny) DiffMeans bank.
BANK_BRIEF = "Suggest one idea for revitalizing an unused public lot."
CREATIVE = [
    "Flood it at dusk into a shallow mirror-pool so the skyline doubles underfoot and kids wade through reflected clouds.",
    "Hang a canopy of recycled-bottle prisms that scatter noon sun into a slow constellation crawling across the pavement.",
    "Grow living benches from mycelium that reshape each season and quietly compost themselves into next year's wildflowers.",
    "Tune a grove of resonant steel pipes to the prevailing wind so the lot hums a different chord every afternoon.",
]
DULL = [
    "Add some benches and a few trees so people have a place to sit.",
    "Put in a small playground and maybe a basketball court.",
    "Plant grass and add lighting so the area is safe at night.",
    "Install trash cans and signs to keep the space clean.",
]
STEER_BRIEF = "Pitch a concept for a small neighborhood museum."


def _build_bank(backend, archetype_layers: list[int]) -> ArchetypeBank:
    """Capture mean-response activations for the contrast and slice a few layers as archetype modes."""
    def stack(texts: list[str]) -> np.ndarray:
        rows = [
            backend.capture_activations(
                BANK_BRIEF, t, positions=[CAPTURE_MEAN_RESPONSE]
            )[CAPTURE_MEAN_RESPONSE]
            for t in texts
        ]
        return np.stack(rows)  # (n_examples, n_layers, hidden)

    pos, neg = stack(CREATIVE), stack(DULL)
    return ArchetypeBank.from_stacks(pos, neg, archetype_layers=archetype_layers)


def _maybe_vendi(texts: list[str]) -> float | None:
    """Best-effort raw Vendi Score from the sibling brainstormlm package (skip cleanly if absent)."""
    try:
        repo_src = Path(__file__).resolve().parents[2] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from brainstormlm.eval.diversity import vendi_score

        return float(vendi_score(texts))
    except Exception as exc:  # noqa: BLE001 — optional cross-package metric; never fatal
        print(f"  (raw Vendi unavailable: {type(exc).__name__}: {exc})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen_0_5b_local.json")
    ap.add_argument("--layer", type=int, default=None, help="injection layer (default: num_layers//2)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="steering strength, rescale=True. The 0.5B residual is fragile: its "
                         "coherence cliff for a single mid-layer injection is ~0.3-0.4; above ~0.6 "
                         "outputs degrade into token-salad (spread stays high but is meaningless).")
    ap.add_argument("--archetype-layers", type=int, nargs="+", default=None,
                    help="layers to use as archetype modes (default: four evenly-spaced mid layers)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: number of archetype modes")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = build_backend(cfg)
    n_layers, hidden = backend.num_layers, backend.hidden_dim
    layer = args.layer if args.layer is not None else n_layers // 2
    arch_layers = args.archetype_layers or [round(n_layers * f) for f in (0.25, 0.40, 0.55, 0.70)]
    print(f"model={backend.model_id}  n_layers={n_layers}  hidden={hidden}  inject_layer={layer}")
    print(f"archetype_layers={arch_layers}  alpha={args.alpha}")

    bank = _build_bank(backend, arch_layers)
    batch_size = args.batch_size or bank.k
    res = build_orthogonal_archetype_dirs(bank, batch_size)
    print("\n== orthogonalisation diagnostics (X1×X2) ==")
    print(f"  distinctness order : {res.order}")
    print(f"  retention cos      : {np.round(res.retention_cos, 3).tolist()}")
    print(f"  fell back to raw   : {res.fell_back or 'none'}")

    steered = backend.generate_steered_batch(
        STEER_BRIEF, dirs=res.dirs, layer=layer, alpha=args.alpha,
        max_new_tokens=args.max_new_tokens, sampling=SamplingPreset(temperature=0.0),
    )
    baseline = backend.generate_steered_batch(
        STEER_BRIEF, dirs=np.zeros_like(res.dirs), layer=layer, alpha=args.alpha,
        max_new_tokens=args.max_new_tokens, sampling=SamplingPreset(temperature=0.0),
    )

    scorer = make_scorer("real")
    steered_spread = scorer.score_collection(steered)
    baseline_spread = scorer.score_collection(baseline)

    print(f"\n== baseline batch (zero-direction, greedy) — sample [0] for coherence reference ==")
    print(f"  [0] {baseline[0].strip()[:160]}")
    print(f"\n== steered batch ({len(steered)} samples, greedy) ==")
    for i, t in enumerate(steered):
        print(f"  [{i}] {t.strip()[:160]}")
    print("\n== semantic spread (mean pairwise cosine distance; higher = more diverse) ==")
    print(f"  baseline (zero-direction) : {baseline_spread:.4f}")
    print(f"  steered  (orthogonal set) : {steered_spread:.4f}")
    v_base, v_steer = _maybe_vendi(baseline), _maybe_vendi(steered)
    if v_base is not None and v_steer is not None:
        print(f"  raw Vendi  baseline={v_base:.3f}  steered={v_steer:.3f}")
    print(f"  delta = {steered_spread - baseline_spread:+.4f}  "
          f"({'spread increased' if steered_spread > baseline_spread else 'no increase — see caveat'})")

    # Assert the MECHANISM only — never the spread (geometry can lie; the real gate is worktree-C
    # Conditional-Vendi / NoveltyBench-Distinct, not yet built).
    assert len(steered) == batch_size, "wrong number of steered outputs"
    assert all(t.strip() for t in steered), "a steered output was empty"
    assert not any(
        np.isnan(scorer_val) for scorer_val in (steered_spread, baseline_spread)
    ), "spread score was NaN"

    print(
        "\nNOTE: necessary-not-sufficient. Orthogonal directions != distinct outputs "
        "(2602.06801). This smoke validates the wiring + preliminary spread only; the fusion "
        "earns its keep when it lifts Conditional-Vendi / NoveltyBench-Distinct (worktree C, C2/C3)."
    )
    print("\nOK — batch steering produced one output per orthogonalised archetype mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
