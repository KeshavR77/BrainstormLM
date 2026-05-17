#!/usr/bin/env python3
"""Build the 15-vs-15 Stanford engineering logo face-off pack.

Pulls 15 Conceptial 14B outputs from artifacts/conceptial_14b/CLOUD_14B_S{1,2,3}.md
and 15 GPT-4o outputs (the existing 8 from the benchmark run plus 7 more
generated here to match), then writes both a JSON and a side-by-side markdown
pack to artifacts/logo_faceoff/.

Usage::

    set -a; source .env; set +a
    uv run python scripts/build_logo_faceoff.py
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BRIEF = (
    "Give me a creative logo for an engineering club at Stanford. I want a "
    "memorable concept name and a one-paragraph visual description suitable "
    "for an image-generation prompt."
)

CONCEPTIAL_FILES = [
    ROOT / "artifacts" / "conceptial_14b" / "CLOUD_14B_S1.md",
    ROOT / "artifacts" / "conceptial_14b" / "CLOUD_14B_S2.md",
    ROOT / "artifacts" / "conceptial_14b" / "CLOUD_14B_S3.md",
]
GPT_CACHED = ROOT / "artifacts" / "benchmark_v2" / "run_14b_pending" / "logo-stanford-engineering.gpt-4o.json"
OUT_DIR = ROOT / "artifacts" / "logo_faceoff"


def parse_conceptial(md_path: Path) -> list[dict]:
    """Pull each idea body from a CLOUD_14B_S*.md file."""
    text = md_path.read_text()
    pattern = re.compile(
        r"### Idea (\d+) \(T=([\d.]+)\).*?_Output:_\n```\n(.*?)\n```",
        re.DOTALL,
    )
    out: list[dict] = []
    for m in pattern.finditer(text):
        body = m.group(3).strip()
        name = body.split("\n", 1)[0].strip().lstrip("*# ").rstrip("*").strip()
        out.append(
            {
                "k": int(m.group(1)),
                "temperature": float(m.group(2)),
                "name": name,
                "body": body,
                "source_file": md_path.name,
            }
        )
    return out


def generate_extra_gpt(n_new: int, k_offset: int) -> list[dict]:
    """Generate n_new additional GPT-4o outputs for the same brief."""
    from brainstormlm.clients.openai import complete as openai_complete

    system = (
        "You are a brand creative director. When given a brief, respond with ONE "
        "concept direction: a memorable 2-6 word name as the first line, then a "
        "one-paragraph body describing the concrete details. No preamble, no "
        "alternatives — just the single direction."
    )
    out: list[dict] = []
    for i in range(n_new):
        k = k_offset + i
        text, _usage = openai_complete(
            system=system,
            messages=[{"role": "user", "content": BRIEF + f" (sample {k+1})"}],
            model="gpt-4o",
            max_tokens=350,
            temperature=1.0,
            use_cache=False,
        )
        out.append(
            {
                "model": "gpt-4o",
                "k": k,
                "text": text.strip(),
                "trajectory_id": uuid.uuid4().hex,
            }
        )
        print(f"  + GPT-4o k={k}: {text.strip().splitlines()[0][:60]}")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Conceptial side ───────────────────────────────────────────────────────
    conceptial: list[dict] = []
    for f in CONCEPTIAL_FILES:
        for idx, item in enumerate(parse_conceptial(f)):
            item["batch"] = f.stem  # CLOUD_14B_S1 etc.
            conceptial.append(item)
    # Renumber k globally 0..14 in order
    for i, item in enumerate(conceptial):
        item["k_global"] = i
    print(f"Conceptial 14B: parsed {len(conceptial)} bodies from 3 batches.")

    # ── GPT side: load cached 8 + generate 7 more ─────────────────────────────
    cached = json.loads(GPT_CACHED.read_text())
    gpt_existing = cached["outputs"]
    print(f"GPT-4o cached: {len(gpt_existing)} bodies.")
    need = 15 - len(gpt_existing)
    if need > 0:
        print(f"Generating {need} more GPT-4o bodies …")
        extra = generate_extra_gpt(need, k_offset=len(gpt_existing))
        gpt_all = gpt_existing + extra
    else:
        gpt_all = gpt_existing[:15]

    # Normalize GPT entries: extract name (first line) and body
    gpt_normalized: list[dict] = []
    for o in gpt_all:
        body = o["text"].strip()
        name = body.split("\n", 1)[0].strip().lstrip("*# ").rstrip("*").strip()
        gpt_normalized.append(
            {
                "k": o["k"],
                "name": name,
                "body": body,
                "trajectory_id": o.get("trajectory_id"),
            }
        )

    # ── Write JSON ────────────────────────────────────────────────────────────
    payload = {
        "brief": BRIEF,
        "prompt_id": "logo-stanford-engineering",
        "n_per_arm": 15,
        "arms": {
            "conceptial-14b": {
                "model": "Qwen2.5-14B-Instruct + CAA + anti-mode + project memory",
                "n": len(conceptial),
                "outputs": conceptial,
            },
            "gpt-4o": {
                "model": "gpt-4o (temperature=1.0)",
                "n": len(gpt_normalized),
                "outputs": gpt_normalized,
            },
        },
    }
    json_path = OUT_DIR / "logo_faceoff.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path.relative_to(ROOT)}")

    # ── Write side-by-side markdown ───────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Logo face-off — Stanford engineering club (15 vs 15)")
    lines.append("")
    lines.append(f"**Brief:** {BRIEF}")
    lines.append("")
    lines.append("**Left arm:** Conceptial 14B (Qwen2.5-14B-Instruct + CAA L14/L18 + anti-mode + project memory)")
    lines.append("**Right arm:** GPT-4o (temperature=1.0, single-shot, no memory)")
    lines.append("")
    lines.append("## Names at a glance")
    lines.append("")
    lines.append("| # | Conceptial 14B | GPT-4o |")
    lines.append("|---:|---|---|")
    for i in range(15):
        c_name = conceptial[i]["name"] if i < len(conceptial) else "—"
        g_name = gpt_normalized[i]["name"] if i < len(gpt_normalized) else "—"
        lines.append(f"| {i+1} | {c_name} | {g_name} |")
    lines.append("")
    lines.append("## Full bodies — Conceptial 14B")
    lines.append("")
    for i, item in enumerate(conceptial):
        lines.append(f"### C{i+1}. {item['name']}")
        lines.append("")
        lines.append(item["body"])
        lines.append("")
    lines.append("## Full bodies — GPT-4o")
    lines.append("")
    for i, item in enumerate(gpt_normalized):
        lines.append(f"### G{i+1}. {item['name']}")
        lines.append("")
        lines.append(item["body"])
        lines.append("")

    md_path = OUT_DIR / "LOGO_FACEOFF.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path.relative_to(ROOT)}")

    # ── Quick duplicate-name flag ────────────────────────────────────────────
    from collections import Counter
    cnames = Counter(o["name"] for o in conceptial)
    gnames = Counter(o["name"] for o in gpt_normalized)
    c_dups = {k: v for k, v in cnames.items() if v > 1}
    g_dups = {k: v for k, v in gnames.items() if v > 1}
    print(f"\nDuplicate names — Conceptial: {c_dups or 'none'}")
    print(f"Duplicate names — GPT-4o:     {g_dups or 'none'}")
    print(f"Unique names — Conceptial: {len(cnames)}/15 · GPT-4o: {len(gnames)}/15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
