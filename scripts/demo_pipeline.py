#!/usr/bin/env python3
"""End-to-end Conceptial demo pipeline: generate → score → filter → final output.

The production-ready flow for the GPT-5.5-vs-Conceptial side-by-side demo. Combines:

- v3 anti-mode config (no exemplars) for **maximum cross-session diversity** (Vendi ≈ 4.0 on logo)
- Best-of-N over-generation (defaults: generate ``--n-target × 1.5``, keep top ``--n-target`` by composite)
- GPT-4o-mini quality scoring (composite of creativity / coherence / specificity / on_topic)
- Final ranked output saved as a single JSON ready for downstream image-gen pipelines

For multi-shot persistence (e.g. 3 sessions × 5 outputs = 15 demo concepts), call this
script 3 times with the same ``--project-id`` and ``--n-target 5``; each call generates
``5×1.5 ≈ 8`` candidates, persists+excludes from the project memory, and writes top-5 by
quality.

Run::

    python3 scripts/demo_pipeline.py \\
        --prompt "Give me a creative logo for an engineering club at Stanford." \\
        --n-target 5 --over-generate-factor 1.5 \\
        --output-mode image-prompt \\
        --project-id "demo-final-logo" --reset-project \\
        --out-json artifacts/conceptial/demo_final_logo_S1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_cognitive_anti_mode.py"
SCORE_SCRIPT = ROOT / "scripts" / "score_outputs.py"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--n-target", type=int, default=5, help="Final number of concepts to keep (default 5).")
    p.add_argument(
        "--over-generate-factor",
        type=float,
        default=1.5,
        help="Generate n_target * factor candidates, keep top n_target by composite (default 1.5 → 8 from 5).",
    )
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--base-temp", type=float, default=0.8)
    p.add_argument("--temp-jitter", type=float, default=0.05)
    p.add_argument("--min-p", type=float, default=0.01)
    p.add_argument(
        "--output-mode",
        choices=("image-prompt", "strategy", "generic"),
        default="image-prompt",
    )
    p.add_argument(
        "--project-id",
        default=None,
        help="Persist concepts under this project (text-level anti-mode across sessions).",
    )
    p.add_argument("--reset-project", action="store_true", default=False)
    p.add_argument(
        "--exemplars-json",
        type=Path,
        default=None,
        help=(
            "Optional few-shot exemplars (in-context demos). NOTE: at 15+ output scale this "
            "tends to *narrow* stylistic diversity by anchoring naming patterns — see "
            "src/brainstormlm/conceptial/README.md Pareto table. Recommended OFF for 3-shot demo flows."
        ),
    )
    p.add_argument(
        "--scoring-model",
        default=os.environ.get("CONCEPTIAL_SCORE_MODEL", "gpt-4o-mini"),
        help="OpenAI model for quality scoring (default gpt-4o-mini).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Where to write the final ranked top-N concepts (JSON).",
    )
    args = p.parse_args()

    n_over = max(args.n_target + 1, int(round(args.n_target * args.over_generate_factor)))
    print(f"[pipeline] target={args.n_target} over-generate={n_over}", flush=True)

    # ---- Step 1: over-generate ----
    smoke_md = args.out_json.with_suffix(".smoke.md")
    smoke_cmd = [
        sys.executable,
        str(SMOKE_SCRIPT),
        "--prompt",
        args.prompt,
        "--n",
        str(n_over),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--base-temp",
        str(args.base_temp),
        "--temp-jitter",
        str(args.temp_jitter),
        "--min-p",
        str(args.min_p),
        "--no-pristine",
        "--output-mode",
        args.output_mode,
        "--out-md",
        str(smoke_md),
    ]
    if args.project_id:
        smoke_cmd.extend(["--project-id", args.project_id])
    if args.reset_project:
        smoke_cmd.append("--reset-project")
    if args.exemplars_json:
        smoke_cmd.extend(["--exemplars-json", str(args.exemplars_json)])

    print(f"[pipeline] step 1/3 — over-generate {n_over} candidates", flush=True)
    t0 = time.time()
    r = subprocess.run(smoke_cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[pipeline] generation failed (exit {r.returncode})", file=sys.stderr)
        return r.returncode
    print(f"[pipeline] generation elapsed={time.time() - t0:.1f}s", flush=True)

    # ---- Step 2: score ----
    print(f"[pipeline] step 2/3 — score with {args.scoring_model}", flush=True)
    score_cmd = [
        sys.executable,
        str(SCORE_SCRIPT),
        "--md",
        str(smoke_md),
        "--model",
        args.scoring_model,
    ]
    r = subprocess.run(score_cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[pipeline] scoring failed (exit {r.returncode})", file=sys.stderr)
        return r.returncode

    # ---- Step 3: filter top-N by composite ----
    scores_json = smoke_md.with_suffix(".scores.json")
    if not scores_json.is_file():
        print(f"[pipeline] expected scores file not found: {scores_json}", file=sys.stderr)
        return 1
    sd = json.loads(scores_json.read_text(encoding="utf-8"))
    per_idea = []
    for arm, rows in (sd.get("per_idea") or {}).items():
        per_idea.extend(rows)
    if not per_idea:
        print("[pipeline] no scored ideas to rank", file=sys.stderr)
        return 1

    # Re-extract per-idea body text from the smoke markdown so we have full content paired
    # with score. (score_outputs has the parsing already; reuse via subprocess result.)
    import re as _re

    IDEA_RX = _re.compile(r"^###\s+Idea\s+(\d+)\s*\(T=([\d.]+)\)\s*$", _re.MULTILINE)
    FENCE_RX = _re.compile(r"```(?:[a-z]*\n)?(.*?)```", _re.DOTALL)
    md_text = smoke_md.read_text(encoding="utf-8")
    body_by_i: dict[int, str] = {}
    matches = list(IDEA_RX.finditer(md_text))
    for j, m in enumerate(matches):
        end = matches[j + 1].start() if j + 1 < len(matches) else len(md_text)
        fences = FENCE_RX.findall(md_text[m.end() : end])
        if fences:
            body_by_i[int(m.group(1))] = fences[-1].strip()

    ranked = sorted(
        per_idea,
        key=lambda r: (
            -(r.get("composite") or 0.0),
            -(r.get("creativity") or 0),
            -(r.get("specificity") or 0),
            -(r.get("on_topic") or 0),
        ),
    )
    top = ranked[: args.n_target]

    final = {
        "prompt": args.prompt,
        "output_mode": args.output_mode,
        "project_id": args.project_id,
        "n_target": args.n_target,
        "n_generated": len(ranked),
        "scoring_model": args.scoring_model,
        "config": {
            "base_temp": args.base_temp,
            "temp_jitter": args.temp_jitter,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "concepts": [
            {
                "rank": idx + 1,
                "composite": r.get("composite"),
                "creativity": r.get("creativity"),
                "coherence": r.get("coherence"),
                "specificity": r.get("specificity"),
                "on_topic": r.get("on_topic"),
                "temp": r.get("temp"),
                "rationale": r.get("one_line_rationale"),
                "body": body_by_i.get(int(r["i"])) or "",
            }
            for idx, r in enumerate(top)
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[pipeline] step 3/3 — wrote top-{args.n_target}/{len(ranked)} to {args.out_json}", flush=True)
    print("\n=== Top concepts ===")
    for c in final["concepts"]:
        first_line = (c["body"].splitlines() or [""])[0]
        print(f"  rank={c['rank']} composite={c['composite']} :: {first_line[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
