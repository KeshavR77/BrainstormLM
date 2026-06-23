#!/usr/bin/env python3
"""Run the 3-judge LLM ensemble OFF-POD on persisted sweep generations.

No GPU and no regeneration: Stage-1 novelty is recomputed deterministically from the saved
text (CPU/MPS), and the judges (Sonnet 4.6 + gpt-4o + gemini-2.5-flash) run over the API. All
cells are judged in one batch so the composite z-scoring is on one comparable scale.

Usage:
  python creativity_steering/scripts/judge_generations.py \
    --gens 'artifacts/qwen14b_run_v3/patching/mean_response/S1_chunk*_generations.jsonl' \
    --out  artifacts/qwen14b_run_v3/patching/mean_response/offpod_judged_all144.json \
    --device mps
  # smoke first:  add  --only-cells baseline L6_a0.25
"""
import argparse
import glob
import json
import sys

from creativity_steering.patching.judge_offpod import (
    judge_generations,
    load_generation_records,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True, help="generation jsonl files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality-floor", type=float, default=2.0)
    ap.add_argument("--device", default=None, help="stage1 device: mps|cpu|cuda (default auto)")
    ap.add_argument("--only-cells", nargs="*", default=None, help="restrict to these labels (smoke)")
    ap.add_argument("--n-workers", type=int, default=4)
    args = ap.parse_args()

    paths: list[str] = []
    for g in args.gens:
        paths.extend(sorted(glob.glob(g)) or [g])
    if not paths:
        sys.exit(f"no generation files matched: {args.gens}")
    recs = load_generation_records(paths)
    if args.only_cells:
        keep = set(args.only_cells)
        recs = [r for r in recs if r["cell"] in keep]
    n_cells = len({r["cell"] for r in recs})
    print(f"loaded {len(recs)} generations across {n_cells} cells from {len(paths)} files",
          file=sys.stderr, flush=True)

    from creativity_evaluators import EvalConfig

    cfg = EvalConfig(
        enable_stage2=True,
        generator_family="qwen",
        stage1_percentile_floor=0.0,   # judge every response (no top-k gating)
        run_reward=False,
        composite_quality_floor=args.quality_floor,
        stage1_device=args.device,
    )
    res = judge_generations(recs, eval_config=cfg, n_workers=args.n_workers)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    b = res["baseline"]
    print(f"baseline: composite={b['composite']:+.3f} stage1={b['stage1']:+.3f} "
          f"prag={b['pragmaticality']} n={b['n_samples']}", file=sys.stderr)
    print(f"wrote {len(res['cells'])} judged cells -> {args.out}", file=sys.stderr)
    for c in res["cells"][:8]:
        print(f"  {c['label']}: composite={c['composite']:+.3f} delta={c['delta']:+.3f} "
              f"prag={c['pragmaticality']} n={c['n_samples']}", file=sys.stderr)


if __name__ == "__main__":
    main()
