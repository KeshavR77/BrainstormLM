#!/usr/bin/env python3
"""Generate the human-readable headline-numbers report from a benchmark run.

Reads ``artifacts/benchmark_v2/<run>/SUMMARY.json`` and writes a markdown
report with:

- Aggregate numbers per arm
- Per-category breakdown
- Pareto-style insight (Conceptial's Vendi gain vs its composite delta)
- Distinct-archetype count via similarity-clustering at cosine ≥ 0.85
- Defensible headline sentences ready to copy into the pitch

Usage::

    uv run python scripts/summarize_benchmark.py \\
        --run artifacts/benchmark_v2/run_2026_05_16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _embed_texts(texts: list[str]) -> np.ndarray:
    from creativity_evaluator.embeddings.local_embedder import LocalEmbedder
    return np.asarray(LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2").embed(texts), dtype=np.float32)


def _archetype_count(vectors: np.ndarray, threshold: float = 0.85) -> int:
    """Greedy cluster count at cosine ≥ threshold — proxy for 'distinct archetypes'."""
    if len(vectors) == 0:
        return 0
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    clusters: list[np.ndarray] = []
    for i in range(len(v)):
        matched = False
        for centroid in clusters:
            if float(np.dot(v[i], centroid)) >= threshold:
                matched = True
                break
        if not matched:
            clusters.append(v[i])
    return len(clusters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.run / "SUMMARY.json"
    if not summary_path.exists():
        print(f"Missing {summary_path}", file=sys.stderr)
        return 1
    data = json.loads(summary_path.read_text())
    arms = data["arms"]
    per_prompt = data["per_prompt"]
    agg = data["aggregate"]
    n = data["n_per_arm_per_prompt"]

    # ── Per-category aggregates ───────────────────────────────────────────────
    by_cat: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in per_prompt:
        cat = row["category"]
        for arm in arms:
            m = row["arms"][arm]
            if m["vendi"] is None:
                continue
            by_cat[cat][arm]["vendi"].append(m["vendi"])
            by_cat[cat][arm]["near_dup"].append(m["near_dup"])
            by_cat[cat][arm]["distinct_2"].append(m["distinct_2"])

    # ── Archetype counts (re-embed) ───────────────────────────────────────────
    archetype_rows: list[dict] = []
    for row in per_prompt:
        pid = row["prompt_id"]
        cat = row["category"]
        rec: dict = {"prompt_id": pid, "category": cat, "arms": {}}
        for arm in arms:
            arm_path = args.run / f"{pid}.{arm}.json"
            if not arm_path.exists():
                rec["arms"][arm] = None
                continue
            outs = json.loads(arm_path.read_text()).get("outputs", [])
            texts = [o["text"] for o in outs if o.get("text")]
            if len(texts) < 2:
                rec["arms"][arm] = None
                continue
            vectors = _embed_texts(texts)
            rec["arms"][arm] = _archetype_count(vectors, threshold=0.85)
        archetype_rows.append(rec)

    arch_mean: dict[str, float] = {}
    for arm in arms:
        vals = [r["arms"][arm] for r in archetype_rows if r["arms"][arm] is not None]
        arch_mean[arm] = float(np.mean(vals)) if vals else 0.0

    # ── Headline sentences ────────────────────────────────────────────────────
    def pct(a: float, b: float) -> str:
        if not b:
            return "—"
        return f"{(a - b) / b * 100:+.1f}%"

    headlines: list[str] = []
    if "conceptial-3b" in arms and "gpt-4o" in arms:
        c = agg["conceptial-3b"]
        g = agg["gpt-4o"]
        headlines.append(
            f"**Diversity:** Conceptial 3B mean Vendi {c['mean_vendi']:.3f} vs GPT-4o "
            f"{g['mean_vendi']:.3f} ({pct(c['mean_vendi'], g['mean_vendi'])} relative)."
        )
        headlines.append(
            f"**Near-duplicate rate:** Conceptial 3B {c['mean_near_dup']:.3f} vs GPT-4o "
            f"{g['mean_near_dup']:.3f} ({pct(c['mean_near_dup'], g['mean_near_dup'])} relative — "
            f"lower is better)."
        )
        headlines.append(
            f"**Quality:** Conceptial 3B composite {c['mean_composite']:.2f} vs GPT-4o "
            f"{g['mean_composite']:.2f} ({pct(c['mean_composite'], g['mean_composite'])} relative)."
        )
        ac = arch_mean.get("conceptial-3b", 0)
        ag = arch_mean.get("gpt-4o", 0)
        headlines.append(
            f"**Distinct archetypes:** Conceptial 3B {ac:.1f} per prompt vs GPT-4o "
            f"{ag:.1f} ({pct(ac, ag)} relative). Cosine ≥ 0.85 clustering at N={n}."
        )

    # ── Write report ──────────────────────────────────────────────────────────
    out_path = args.out or (args.run / "HEADLINE_NUMBERS.md")
    lines: list[str] = []
    lines.append("# Conceptial benchmark — headline numbers")
    lines.append("")
    lines.append(f"**Run:** `{args.run}`")
    lines.append(f"**N per arm per prompt:** {n}")
    lines.append(f"**Prompts:** {len(per_prompt)}  ·  **Arms:** {', '.join(arms)}")
    lines.append("")
    lines.append("## Headline sentences (paste-ready)")
    lines.append("")
    for h in headlines:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("## Aggregate across all prompts")
    lines.append("")
    lines.append("| Arm | Vendi | Near-dup | Distinct-2 | Archetypes | Composite | Creativity | Specificity | On-topic |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        a = agg[arm]
        def f(x, fmt=".2f"):
            return "—" if x is None else format(x, fmt)
        lines.append(
            f"| {arm} | {f(a['mean_vendi'], '.3f')} | {f(a['mean_near_dup'], '.3f')} | "
            f"{f(a['mean_distinct_2'], '.3f')} | {arch_mean[arm]:.1f} | "
            f"{f(a['mean_composite'])} | {f(a['mean_creativity'])} | "
            f"{f(a['mean_specificity'])} | {f(a['mean_on_topic'])} |"
        )
    lines.append("")
    lines.append("## Per-category breakdown — mean Vendi")
    lines.append("")
    cats = sorted(by_cat.keys())
    header = ["category"] + arms
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cat in cats:
        cells = [cat]
        for arm in arms:
            vs = by_cat[cat][arm].get("vendi", [])
            cells.append(f"{np.mean(vs):.3f}" if vs else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-prompt Vendi")
    lines.append("")
    h2 = ["prompt"] + arms
    lines.append("| " + " | ".join(h2) + " |")
    lines.append("|" + "|".join(["---"] * len(h2)) + "|")
    for row in per_prompt:
        cells = [row["prompt_id"]]
        for arm in arms:
            v = row["arms"][arm].get("vendi")
            cells.append("—" if v is None else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-prompt distinct archetypes (cosine ≥ 0.85 clusters)")
    lines.append("")
    lines.append("| " + " | ".join(h2) + " |")
    lines.append("|" + "|".join(["---"] * len(h2)) + "|")
    for r in archetype_rows:
        cells = [r["prompt_id"]]
        for arm in arms:
            v = r["arms"][arm]
            cells.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
