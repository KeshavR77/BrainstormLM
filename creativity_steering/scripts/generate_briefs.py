#!/usr/bin/env python3
"""Real steer -> refine generation over a briefs file (no mocks).

Loads Qwen2.5-14B once, then for each brief generates K steered samples (varying the seed) and,
with --refine, a concept-frozen OpenAI polish of each. Every sample is checkpointed to
``<out>/<brief_id>.jsonl`` so a crash/disconnect resumes (just re-run the same command).

  STEERING_PACKAGE / --package  steering package (default: configs/designer_run.json -> L7@0.25)
  --briefs                      briefs json (default: config/designer_briefs.json)
  --variant L7|L6|L17           which steering to apply (default L7 — the steer->refine pick)
  --samples K                   distinct directions per brief
  --refine                      concept-frozen OpenAI refine (uses OPENAI_API_KEY)
  --baseline                    also generate the unsteered arm (steered-vs-baseline A/B)

  PYTHONPATH=creativity_steering OPENAI_API_KEY=… \
  python creativity_steering/scripts/generate_briefs.py \
    --variant L7 --samples 12 --refine --baseline --out results/designer_run
"""
import argparse
import json
import os
from pathlib import Path

from creativity_steering.serving.batch import run_briefs
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.refiner import OpenAIRefiner
from creativity_steering.serving.service import SteeringService


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default=os.environ.get(
        "STEERING_PACKAGE", "creativity_steering/configs/designer_run.json"))
    ap.add_argument("--briefs", default="config/designer_briefs.json")
    ap.add_argument("--variant", default="L7")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", default="results/designer_run")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--no-seeds", action="store_true",
                    help="disable archetype-seed rotation (the anti-mode-collapse lens)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--refine-model", default=os.environ.get("REFINER_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    pkg = SteeringPackage.load(args.package)
    data = json.loads(Path(args.briefs).read_text())
    briefs = data.get("prompts") or data.get("briefs")
    refiner = OpenAIRefiner(model=args.refine_model) if args.refine else None

    print(f"[run] package={pkg.name} variant={args.variant} briefs={len(briefs)} "
          f"samples={args.samples} refine={args.refine} baseline={args.baseline} "
          f"out={args.out}", flush=True)
    print("[run] loading model (first load streams the 14B shards)…", flush=True)
    svc = SteeringService.from_package(pkg, refiner=refiner)

    summary = run_briefs(
        svc, briefs, variant=args.variant, samples=args.samples, out_dir=args.out,
        refine=args.refine, baseline=args.baseline, rotate_seeds=not args.no_seeds,
        max_new_tokens=args.max_new_tokens,
    )
    made = sum(s["made"] for s in summary)
    print(f"[run] done — {made} new samples across {len(summary)} briefs. "
          f"See {args.out}/IMAGE_PROMPTS.md", flush=True)


if __name__ == "__main__":
    main()
