"""Re-derive steering vectors from a saved scored pool at arbitrary cutoffs.

Use after a derive run with ``save_scored_pool=True``. Reads
``<run>/scored_pool.jsonl`` and ``<run>/activations/<position>.npz``, picks the verified
contrast set via either explicit percentile cuts or absolute score thresholds, computes
DiffMeans per layer, and writes new vector sets under ``<run>/vectors/<position>/`` or a
custom ``--out`` directory.

This is the cheap, GPU-free re-derivation path: no generation, no scoring — just an
in-memory filter + a mean. Costs seconds.

Examples
--------
Top/bottom 10% of the pool, all capture positions, primary = last_response_token::

    python -m creativity_steering.scripts.refilter_v3 \\
        --run artifacts/qwen14b_run_v3 \\
        --top-percentile 10 --bot-percentile 10 \\
        --out artifacts/qwen14b_run_v3/refiltered/p10_n_auto

Explicit score thresholds, single position::

    python -m creativity_steering.scripts.refilter_v3 \\
        --run artifacts/qwen14b_run_v3 \\
        --creative-score-min 0.5 --neutral-score-max 0.1 \\
        --positions last_response_token \\
        --out artifacts/qwen14b_run_v3/refiltered/strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.derivation.diffmeans import DiffMeansResult


def load_scored_pool(run: Path) -> tuple[list[dict], list[str]]:
    """Return ``(examples, positions)`` from a derive run.

    ``examples`` is a list of dicts from ``scored_pool.jsonl`` in row order (example_id ==
    list index). ``positions`` is the sorted list of capture positions for which an
    ``activations/<position>.npz`` exists.
    """
    pool_path = run / "scored_pool.jsonl"
    if not pool_path.exists():
        sys.exit(f"no scored_pool.jsonl at {pool_path} (was save_scored_pool=true at derive?)")
    examples = [json.loads(line) for line in pool_path.read_text().splitlines() if line]
    acts_dir = run / "activations"
    positions = sorted(p.stem for p in acts_dir.glob("*.npz"))
    if not positions:
        sys.exit(f"no activations/*.npz under {acts_dir}")
    return examples, positions


def pick_indices(
    scores: np.ndarray,
    *,
    top_percentile: float | None,
    bot_percentile: float | None,
    creative_score_min: float | None,
    neutral_score_max: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(positive_idx, negative_idx, meta)``.

    Percentile mode (when ``top_percentile`` and ``bot_percentile`` are set): positives are
    scores at/above the ``(100 - top_percentile)``-th percentile, negatives at/below the
    ``bot_percentile``-th percentile. Threshold mode (when ``creative_score_min`` and
    ``neutral_score_max`` are set): straight absolute cuts on the score.
    """
    if top_percentile is not None and bot_percentile is not None:
        top_cut = float(np.percentile(scores, 100 - top_percentile))
        bot_cut = float(np.percentile(scores, bot_percentile))
        pos_idx = np.where(scores >= top_cut)[0]
        neg_idx = np.where(scores <= bot_cut)[0]
        meta = {
            "mode": "percentile",
            "top_percentile": top_percentile,
            "bot_percentile": bot_percentile,
            "top_cut": top_cut,
            "bot_cut": bot_cut,
        }
    elif creative_score_min is not None and neutral_score_max is not None:
        pos_idx = np.where(scores >= creative_score_min)[0]
        neg_idx = np.where(scores <= neutral_score_max)[0]
        meta = {
            "mode": "threshold",
            "creative_score_min": creative_score_min,
            "neutral_score_max": neutral_score_max,
        }
    else:
        sys.exit(
            "must set either --top-percentile + --bot-percentile, "
            "or --creative-score-min + --neutral-score-max"
        )
    return pos_idx, neg_idx, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="Derive run directory.")
    ap.add_argument("--top-percentile", type=float, help="Top tail percentile (e.g. 10 for top 10%).")
    ap.add_argument("--bot-percentile", type=float, help="Bottom tail percentile (e.g. 10 for bottom 10%).")
    ap.add_argument("--creative-score-min", type=float)
    ap.add_argument("--neutral-score-max", type=float)
    ap.add_argument(
        "--positions", nargs="*", help="Capture positions to refilter (default: all saved)."
    )
    ap.add_argument("--out", type=Path, help="Output directory (default: <run>/refiltered/auto).")
    ap.add_argument("--model-id", type=str, default="unknown", help="Model id for vector manifest.")
    args = ap.parse_args()

    examples, all_positions = load_scored_pool(args.run)
    positions = args.positions or all_positions
    for p in positions:
        if p not in all_positions:
            sys.exit(f"position {p!r} not in saved positions {all_positions}")

    scores = np.array([float(ex["score"]) for ex in examples])
    pos_idx, neg_idx, meta = pick_indices(
        scores,
        top_percentile=args.top_percentile,
        bot_percentile=args.bot_percentile,
        creative_score_min=args.creative_score_min,
        neutral_score_max=args.neutral_score_max,
    )

    n_pos, n_neg = len(pos_idx), len(neg_idx)
    print(f"[refilter] mode={meta['mode']}  n_pos={n_pos}  n_neg={n_neg}  n_total={len(examples)}")
    if n_pos == 0 or n_neg == 0:
        sys.exit("refilter produced an empty pool — relax the cut")

    out_root = args.out or (args.run / "refiltered" / "auto")
    out_root.mkdir(parents=True, exist_ok=True)

    for position in positions:
        npz = np.load(args.run / "activations" / f"{position}.npz")
        acts = npz["acts"]  # (n_examples, n_layers, hidden_dim)
        pos_stack = acts[pos_idx]
        neg_stack = acts[neg_idx]
        result = DiffMeansResult.from_stacks(pos_stack, neg_stack)
        vs = SteeringVectorSet.from_diffmeans(
            result,
            model_id=args.model_id,
            capture_position=position,
            creative_system="(refilter)",
            neutral_system="(refilter)",
        )
        vs.save(out_root / "vectors" / position)
        print(f"  [{position}] vectors saved -> {out_root / 'vectors' / position}")

    (out_root / "refilter_meta.json").write_text(
        json.dumps({**meta, "n_pos": n_pos, "n_neg": n_neg, "positions": positions}, indent=2)
    )
    print(f"[refilter] done -> {out_root}")


if __name__ == "__main__":
    main()
