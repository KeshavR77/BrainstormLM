#!/usr/bin/env python3
"""Evaluate the sweep-winning L17 hyper-purified creative vector on the NEW creativity_evaluators.

The L17 vector (``data/steering_14b/hyper_creative_purified_vector_L17.pt``) was selected by
``sweep_14b_steering.py`` (2026-05-18, Vendi 3.32 vs L19's 2.26). Its steered generations live in
``artifacts/eval_2026-05-18/conceptial-14b/`` alongside the unsteered ``baseline-14b/`` arm on the
same 10 creative briefs (5 samples each).

This re-scores both arms through the new two-stage ``creativity_evaluators`` package so we can read
the vector's effect on the new metric. Stage 2 (LLM judge ensemble) needs API keys; by default we
run Stage 1 only (model-free: DSI / length-normalised PPL / lexical-slop redundancy). Pass
``--stage2`` to add the judge ensemble.

Note: the final composite ``creativity_score`` only carries signal once the QUALITY pole exists
(Stage-2 judges or the reward model). In Stage-1-only mode the headline metric is **novelty**
(``DSI_z + PPL_z - lexical_z``), the axis a creativity vector is supposed to move.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

from creativity_evaluators import EvalConfig, PromptCategory, evaluate_batch

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "eval_2026-05-18"
BRIEFS = ROOT / "config" / "benchmark_creative_briefs.json"
ARMS = {"baseline-14b": "baseline (unsteered)", "conceptial-14b": "L17 steered"}


def load_records(truncate_words: int | None = None) -> list[dict]:
    briefs = {p["id"]: p["brief"] for p in json.loads(BRIEFS.read_text())["prompts"]}
    records: list[dict] = []
    for arm in ARMS:
        for f in sorted((EVAL_DIR / arm).glob("*.json")):
            doc = json.loads(f.read_text())
            pid = doc["prompt_id"]
            for out in doc["outputs"]:
                text = out["text"]
                if truncate_words:
                    text = " ".join(text.split()[:truncate_words])
                records.append(
                    {
                        "prompt": briefs[pid],
                        "response": text,
                        "category": PromptCategory.CREATIVE,
                        "_arm": arm,
                        "_pid": pid,
                        "_k": out.get("k"),
                    }
                )
    return records


def summarise(records: list[dict], results: list) -> None:
    # attach raw signals back to records
    rows = []
    for rec, res in zip(records, results):
        rows.append(
            {
                "arm": rec["_arm"],
                "pid": rec["_pid"],
                "dsi": res.dsi,
                "ppl": res.ppl,
                "lex_redundancy": res.lexical.combined,
                "novelty": res.components.get("novelty"),
                "novelty_z": res.components.get("novelty_z"),
                "creativity_score": res.creativity_score,
                "n_tokens": res.n_tokens,
            }
        )

    def m(arm: str, key: str) -> float:
        vals = [r[key] for r in rows if r["arm"] == arm and r[key] is not None]
        return stats.mean(vals) if vals else float("nan")

    print("\n" + "=" * 74)
    print("L17 hyper-purified creative vector — new creativity_evaluators (Stage 1)")
    print("=" * 74)
    hdr = f"{'metric':<22}{'baseline':>14}{'L17 steered':>16}{'Δ (steer-base)':>18}"
    print(hdr)
    print("-" * 74)
    for key, label in [
        ("dsi", "DSI (↑ novel)"),
        ("ppl", "PPL norm (↑ surpris.)"),
        ("lex_redundancy", "lexical redund (↓)"),
        ("novelty", "NOVELTY (↑) [headline]"),
        ("novelty_z", "novelty_z (batch-rel)"),
        ("n_tokens", "mean tokens"),
    ]:
        b, c = m("baseline-14b", key), m("conceptial-14b", key)
        print(f"{label:<22}{b:>14.4f}{c:>16.4f}{c - b:>+18.4f}")

    print("\nPer-prompt NOVELTY (mean of 5 samples):")
    print(f"{'prompt_id':<32}{'baseline':>12}{'L17':>12}{'Δ':>12}")
    print("-" * 68)
    pids = sorted({r["pid"] for r in rows})
    wins = 0
    for pid in pids:
        bv = stats.mean([r["novelty"] for r in rows if r["pid"] == pid and r["arm"] == "baseline-14b"])
        cv = stats.mean([r["novelty"] for r in rows if r["pid"] == pid and r["arm"] == "conceptial-14b"])
        flag = "  ✓" if cv > bv else ""
        wins += cv > bv
        print(f"{pid:<32}{bv:>12.4f}{cv:>12.4f}{cv - bv:>+12.4f}{flag}")
    print("-" * 68)
    print(f"L17 wins novelty on {wins}/{len(pids)} prompts")

    out = ROOT / "artifacts" / "L17_new_evaluator_scores.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nPer-sample scores -> {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", action="store_true", help="also run the LLM judge ensemble (needs API keys)")
    ap.add_argument(
        "--truncate-words",
        type=int,
        default=None,
        help="length-control: truncate every response to this many words before scoring",
    )
    args = ap.parse_args()

    # PPL is disabled: gpt2-medium emits nan logits under transformers 5.6.2 / torch 2.11.0
    # in this env, so the surprise signal would silently z-impute to 0 anyway. Novelty here is
    # DSI - lexical-slop. Reward + Stage-2 quality pole are off unless --stage2.
    config = EvalConfig(enable_stage2=args.stage2, run_ppl=False)

    records = load_records(truncate_words=args.truncate_words)
    label = f" (truncated to {args.truncate_words} words)" if args.truncate_words else " (full length)"
    print(f"Loaded {len(records)} responses across {len(ARMS)} arms{label}.")
    results = evaluate_batch(records, config)
    summarise(records, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
