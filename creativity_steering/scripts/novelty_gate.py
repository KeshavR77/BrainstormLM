"""Novelty gate — the cheap, judge-free sanity check a steering vector must pass.

The v2 post-mortem established the diagnostic: a *usable* creativity vector raises the
model-free Stage-1 novelty over the unsteered baseline for at least one layer at alpha=1.
v1's vector cleared this (L9 delta-novelty +1.42); v2's did not (every layer <= 0) — and v2
came back neutral. So before spending any LLM-judge budget on the fine / Exp-2 sweeps, run
this against the broad sweep's results: if no layer lifts novelty, the vector is a
noise/destructive direction and the downstream judged sweep is wasted money.

Reads a ``broad_sweep_results.json`` (``two_stage_cell_sweep`` output: a ``baseline`` cell and
a list of per-(layer,alpha) ``cells``, each carrying ``stage1`` = the batch-relative Stage-1
novelty composite). Prints the top layers by delta-novelty and exits non-zero if none beat
baseline.

Usage::

    python -m creativity_steering.scripts.novelty_gate \\
        artifacts/qwen14b_run_v3/patching/mean_response/broad_sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="path to a broad_sweep_results.json")
    ap.add_argument("--alpha", type=float, default=1.0, help="only count cells at this alpha")
    ap.add_argument("--top", type=int, default=8, help="how many top layers to print")
    ap.add_argument(
        "--min-delta", type=float, default=0.25,
        help="minimum delta-novelty (batch z-units) a layer must clear to pass. A trivial "
             "positive delta is noise on a z-scored batch — v1's best was +1.42, v2's +0.056, "
             "so 0.25 separates a real lift from noise.",
    )
    args = ap.parse_args()

    data = json.loads(open(args.results).read())
    base = float(data["baseline"]["stage1"])
    cells = [c for c in data["cells"] if abs(float(c.get("alpha", args.alpha)) - args.alpha) < 1e-9]
    scored = sorted(
        ({"layer": c["layer"], "stage1": float(c["stage1"]), "delta": float(c["stage1"]) - base}
         for c in cells),
        key=lambda c: c["delta"],
        reverse=True,
    )
    n_positive = sum(1 for c in scored if c["delta"] > 0)
    n_meaningful = sum(1 for c in scored if c["delta"] >= args.min_delta)

    print(f"[novelty_gate] baseline Stage-1 novelty = {base:+.4f} (alpha={args.alpha})")
    print(f"[novelty_gate] top {args.top} layers by delta-novelty:")
    for c in scored[: args.top]:
        print(f"    L{c['layer']:<3} stage1={c['stage1']:+.4f}  delta={c['delta']:+.4f}")
    print(f"[novelty_gate] {n_positive}/{len(scored)} layers beat baseline; "
          f"{n_meaningful}/{len(scored)} clear delta>={args.min_delta}")

    if n_meaningful == 0:
        verdict = "destructive" if n_positive == 0 else "no meaningful lift (within noise)"
        print(f"[novelty_gate] FAIL — {verdict}; vector does not raise novelty. Do not spend "
              "judge budget on the fine/Exp-2 sweeps; fix the contrast first.")
        return 1
    best = scored[0]
    print(f"[novelty_gate] PASS — {n_meaningful} layer(s) clear the bar; best is "
          f"L{best['layer']} (delta-novelty {best['delta']:+.4f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
