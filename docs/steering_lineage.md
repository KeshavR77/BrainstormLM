# Steering lineage — which one ships

Two steering lineages exist in the repo. This note is the canonical statement of which is which,
so docs/config/scripts stop contradicting each other.

## Lineage B — DiffMeans v3 small-alpha  ← **this is what ships**

- **What:** single-layer activation steering with a small alpha, from a DiffMeans (CAA) vector
  over a *framed, quality-gated-novelty* contrast (118 positive / 120 negative).
- **Vectors:** `artifacts/qwen14b_run_v3/vectors/{mean_response,…}/` (48-layer, 5120-dim).
- **Serving:** `creativity_steering/.../serving/` via a package, default
  `creativity_steering/configs/ship_v3_bracket.json` (default variant **L17 @ α=0.2**; the
  steer→refine pick used for the real designer run is **L7 @ α=0.25**).
- **Evidence:** full 48-layer × α∈{0.2,0.3,0.4} sweep (144 cells) + 3-judge ensemble; the first
  replicated **beat-baseline** result (`artifacts/qwen14b_run_v3/RESULTS.md`). This is the line the
  2026-06-23 designer run generated on (`artifacts/designer_run/`), and the line the
  [Qwen2.5-32B port](port_to_qwen32b.md) re-derives.

## Lineage A — v17 "purified" CAA (scheduled, dual-layer)  ← older, script-only

- **What:** scheduled dual-layer steering (creative **L17** exp-decay + technical **L24** sigmoid),
  a `v10_parametric_preset_v17` inherited from the 3B stack.
- **Config:** `config/conceptial_14b.json` — still referenced by the `scripts/*_14b.py` CAA path
  (`extract_14b_vectors.py`, `sweep_14b_steering.py`, `runpod_batch_14b.py`,
  `run_feedback_loop.py`, `run_v17_14b_arm.py`). **Kept, not deleted** — those scripts depend on it.
- **Status:** the technical L24/α=0.86 was never empirically tuned on 14B (depth-scaled from 3B).
  A proper steered-vs-anti-mode head-to-head for this line never ran. Treat it as the historical
  CAA lineage, not the shipping steering.

## The demo path is neither

`src/brainstormlm/api/server.py` (the demo canvas backend) serves **curated `demo_packs`** with
project-scoped memory — it does **not** load a model or steer. See the deployment topology note.
The production brainstorm path is the steering serving harness (Lineage B) + the concept-frozen
refiner, driven by `creativity_steering/scripts/generate_briefs.py`.
