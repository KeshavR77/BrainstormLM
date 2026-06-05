"""Re-cut a saved v3 derive pool into a new framed contrast — GPU-free, no regeneration.

After a derive run with ``save_scored_pool=True``, this reproduces the v3 framed selection
(top/bottom novelty quantile among quality-ok creatives / neutrals, one response per prompt)
from ``scored_pool.jsonl`` + ``activations/<position>.npz``, with the length-matching and the
target count under your control — then rebuilds the DiffMeans vector sets. It reuses the live
``_select_one_per_prompt`` / ``_length_match`` helpers, so the contrast matches what
``derive_vectors`` would have produced for the same settings.

This is how we turn the run #2 pool (saved at ~84/84 with length_match on) into a 100/pole
vector by re-cutting with length_match OFF — no GPU, seconds, nothing regenerated.

Example — length-off, exactly 100/pole, all saved positions::

    python -m creativity_steering.scripts.recut_v3 \\
        --run artifacts/qwen14b_run_v3 \\
        --no-length-match --target 100 \\
        --out artifacts/qwen14b_run_v3   # overwrites vectors/<position>/ in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.derivation.diffmeans import DiffMeansResult
from creativity_steering.derivation.evaluator_filter import _length_match, _select_one_per_prompt
from creativity_steering.derivation.records import ResponseExample, VerifiedSubset


def _load_pool(run: Path) -> tuple[list[dict], list[str]]:
    pool_path = run / "scored_pool.jsonl"
    if not pool_path.exists():
        sys.exit(f"no scored_pool.jsonl at {pool_path} (was save_scored_pool=true at derive?)")
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line]
    positions = sorted(p.stem for p in (run / "activations").glob("*.npz"))
    if not positions:
        sys.exit(f"no activations/*.npz under {run / 'activations'}")
    return rows, positions


def _cap_to_target(positives, negatives, target):
    """Keep the ``target`` most-novel positives and ``target`` least-novel negatives."""
    pos = sorted(positives, key=lambda e: e.novelty, reverse=True)[:target]
    neg = sorted(negatives, key=lambda e: e.novelty)[:target]
    return pos, neg


def recut(
    run: Path,
    out: Path,
    *,
    creative_quantile: float = 0.9,
    neutral_quantile: float = 0.1,
    length_match: bool = False,
    length_tolerance: float = 0.5,
    target: int | None = None,
    positions: list[str] | None = None,
) -> dict:
    rows, all_positions = _load_pool(run)
    positions = positions or all_positions
    for p in positions:
        if p not in all_positions:
            sys.exit(f"position {p!r} not in saved positions {all_positions}")

    acts = {p: np.load(run / "activations" / f"{p}.npz")["acts"] for p in positions}

    examples: list[ResponseExample] = []
    for i, row in enumerate(rows):
        ex = ResponseExample(
            framing=row["framing"], system="", prompt=row["prompt"], response=row["response"],
            activations={p: acts[p][i] for p in positions},
            score=row.get("score"),
        )
        ex.novelty = row.get("novelty")
        ex.quality_ok = row.get("quality_ok")
        examples.append(ex)

    creatives = [e for e in examples if e.framing == "creative"]
    neutrals = [e for e in examples if e.framing == "neutral"]
    positives = _select_one_per_prompt(
        creatives, quantile=creative_quantile, keep="high", require_quality=True
    )
    negatives = _select_one_per_prompt(
        neutrals, quantile=neutral_quantile, keep="low", require_quality=False
    )
    print(f"[recut] post-select: {len(positives)} pos / {len(negatives)} neg "
          f"(of {len(creatives)} creative, {len(neutrals)} neutral)")
    if length_match:
        positives, negatives = _length_match(positives, negatives, length_tolerance)
        print(f"[recut] post-length-match (tol={length_tolerance}): "
              f"{len(positives)} pos / {len(negatives)} neg")
    if target is not None:
        positives, negatives = _cap_to_target(positives, negatives, target)
        print(f"[recut] post-cap (target={target}): {len(positives)} pos / {len(negatives)} neg")

    if not positives or not negatives:
        sys.exit("recut produced an empty pool — relax the cut")

    # Carry the model id + framing systems from the run's existing vectors manifest if present.
    model_id, creative_system, neutral_system = "unknown", "", ""
    for p in all_positions:
        man = run / "vectors" / p / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            model_id = m.get("model_id", model_id)
            fr = m.get("framings", {})
            creative_system, neutral_system = fr.get("creative", ""), fr.get("neutral", "")
            break

    subset = VerifiedSubset(positives=positives, negatives=negatives, n_dropped=0)
    out.mkdir(parents=True, exist_ok=True)
    for p in positions:
        pos_stack, neg_stack = subset.stacks(p)
        result = DiffMeansResult.from_stacks(pos_stack, neg_stack)
        vs = SteeringVectorSet.from_diffmeans(
            result, model_id=model_id, capture_position=p,
            creative_system=creative_system, neutral_system=neutral_system,
        )
        vs.save(out / "vectors" / p)
        print(f"  [{p}] vectors saved -> {out / 'vectors' / p}")

    meta = {
        "n_positive": subset.n_positive, "n_negative": subset.n_negative,
        "creative_quantile": creative_quantile, "neutral_quantile": neutral_quantile,
        "length_match": length_match, "length_tolerance": length_tolerance,
        "target": target, "positions": positions, "source_pool": str(run),
    }
    (out / "recut_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[recut] done -> {out}  ({subset.n_positive} pos / {subset.n_negative} neg)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="Derive run dir with scored_pool.jsonl")
    ap.add_argument("--out", type=Path, help="Output dir (default: <run>, overwriting vectors/)")
    ap.add_argument("--creative-quantile", type=float, default=0.9)
    ap.add_argument("--neutral-quantile", type=float, default=0.1)
    lm = ap.add_mutually_exclusive_group()
    lm.add_argument("--length-match", dest="length_match", action="store_true")
    lm.add_argument("--no-length-match", dest="length_match", action="store_false")
    ap.set_defaults(length_match=False)
    ap.add_argument("--length-tolerance", type=float, default=0.5)
    ap.add_argument("--target", type=int, help="Cap to this many per pole (most/least novel).")
    ap.add_argument("--positions", nargs="*")
    args = ap.parse_args()
    recut(
        args.run, args.out or args.run,
        creative_quantile=args.creative_quantile, neutral_quantile=args.neutral_quantile,
        length_match=args.length_match, length_tolerance=args.length_tolerance,
        target=args.target, positions=args.positions,
    )


if __name__ == "__main__":
    main()
