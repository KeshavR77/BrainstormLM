#!/usr/bin/env python3
"""Auto-extract top-scoring concepts from prior scored smoke runs as few-shot exemplars.

The "manual ReasoningBank" the user explicitly asked for: feed the model its own best
outputs back as in-context demonstrations. Reads one or more ``*.scores.json`` files
produced by ``score_outputs.py``, picks the top-K by composite, deduplicates near-identical
ones, and writes a fresh ``exemplars.json`` ready to feed to ``smoke_cognitive_anti_mode.py``.

Composite tie-breaks favor: higher creativity > higher specificity > more on-topic.

Run::

    python3 scripts/extract_exemplars.py \\
        --scores artifacts/V17_ANTIMODE_3SHOT_V3_S2.scores.json \\
        --scores artifacts/V17_ANTIMODE_3SHOT_V3_S3.scores.json \\
        --top-k 2 \\
        --min-composite 4.0 \\
        --out artifacts/exemplars_auto_logo.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


_IDEA_HEADER_RX = re.compile(r"^###\s+Idea\s+(\d+)\s*\(T=([\d.]+)\)\s*$", re.MULTILINE)
_FENCE_RX = re.compile(r"```(?:[a-z]*\n)?(.*?)```", re.DOTALL)


def _load_completion_for_idea(smoke_md_path: Path, arm: str, i: int) -> str | None:
    """Pull the i-th idea's body text from a smoke MD; arm header is informational only since
    most smokes have a single arm at this stage."""
    text = smoke_md_path.read_text(encoding="utf-8")
    matches = list(_IDEA_HEADER_RX.finditer(text))
    for j, m in enumerate(matches):
        if int(m.group(1)) != i:
            continue
        end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
        fences = _FENCE_RX.findall(text[m.end() : end])
        if fences:
            return fences[-1].strip()
    return None


def _shingle_set(text: str, n: int = 5) -> frozenset[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return frozenset(" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))


def _name_tokens(text: str) -> frozenset[str]:
    """First-line tokens (concept name), lowercased, alpha only — used for stylistic dedup."""
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", first_line) if len(t) > 1]
    return frozenset(toks)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores",
        action="append",
        type=Path,
        required=True,
        help="One or more ``*.scores.json`` files. Can be repeated.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Final number of exemplars to write (default: 3).",
    )
    p.add_argument(
        "--min-composite",
        type=float,
        default=4.0,
        help="Skip any concept with composite < this (default: 4.0).",
    )
    p.add_argument(
        "--dedup-jaccard",
        type=float,
        default=0.30,
        help="Reject a candidate if its 5-shingle Jaccard with any already-picked exemplar exceeds this (default: 0.30).",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--note",
        default="Auto-extracted by scripts/extract_exemplars.py from scored smoke runs.",
    )
    args = p.parse_args()

    # Collect every per-idea entry across all input score files, with body text resolved.
    candidates: list[dict] = []
    briefs_seen: list[str] = []
    for scores_path in args.scores:
        if not scores_path.is_file():
            print(f"missing scores file: {scores_path}", file=sys.stderr)
            return 1
        sd = json.loads(scores_path.read_text(encoding="utf-8"))
        brief = sd.get("brief", "")
        if brief and brief not in briefs_seen:
            briefs_seen.append(brief)
        smoke_md_path = Path(sd.get("source_md", "")) if sd.get("source_md") else scores_path.with_suffix(".md").with_name(scores_path.name.replace(".scores.json", ".md"))
        if not smoke_md_path.is_file():
            print(f"could not locate source smoke MD for {scores_path}: tried {smoke_md_path}", file=sys.stderr)
            continue
        for arm, rows in (sd.get("per_idea") or {}).items():
            for r in rows:
                if not r.get("ok"):
                    continue
                composite = float(r.get("composite") or 0.0)
                if composite < args.min_composite:
                    continue
                body = _load_completion_for_idea(smoke_md_path, arm, int(r["i"]))
                if not body:
                    continue
                candidates.append(
                    {
                        "arm": arm,
                        "i": int(r["i"]),
                        "temp": float(r.get("temp", 0.0)),
                        "composite": composite,
                        "creativity": int(r.get("creativity") or 0),
                        "specificity": int(r.get("specificity") or 0),
                        "on_topic": int(r.get("on_topic") or 0),
                        "coherence": int(r.get("coherence") or 0),
                        "rationale": str(r.get("one_line_rationale") or ""),
                        "source_md": str(smoke_md_path),
                        "body": body,
                        "shingle": _shingle_set(body),
                        "name_tokens": _name_tokens(body),
                    }
                )

    if not candidates:
        print(
            f"no candidates with composite >= {args.min_composite} found across {len(args.scores)} score files",
            file=sys.stderr,
        )
        return 1
    print(f"[exemplars] {len(candidates)} candidates above composite={args.min_composite}", flush=True)

    # Rank: composite desc, then creativity desc, then specificity desc, then on_topic desc.
    candidates.sort(
        key=lambda r: (
            -r["composite"],
            -r["creativity"],
            -r["specificity"],
            -r["on_topic"],
        )
    )

    picked: list[dict] = []
    for cand in candidates:
        if len(picked) >= args.top_k:
            break
        # Dedup against already-picked exemplars: BOTH body Jaccard AND first-line-token overlap.
        # The 8-bit-Gearheads vs 8-bit-Eureka-Spark case proved we need stylistic dedup beyond body.
        skip = False
        for p_ in picked:
            jacc = _jaccard(cand["shingle"], p_["shingle"])
            name_jacc = _jaccard(cand["name_tokens"], p_["name_tokens"])
            if jacc > args.dedup_jaccard:
                print(
                    f"  skip i={cand['i']}@{Path(cand['source_md']).name} composite={cand['composite']} "
                    f"(body-jacc={jacc:.2f} vs picked i={p_['i']})",
                    flush=True,
                )
                skip = True
                break
            if name_jacc >= 0.20:  # share >=1/5 of name tokens => same stylistic family
                print(
                    f"  skip i={cand['i']}@{Path(cand['source_md']).name} composite={cand['composite']} "
                    f"(name-jacc={name_jacc:.2f} vs picked i={p_['i']}: shared name tokens)",
                    flush=True,
                )
                skip = True
                break
        if skip:
            continue
        picked.append(cand)
        print(
            f"  pick i={cand['i']}@{Path(cand['source_md']).name} composite={cand['composite']} "
            f"(cr={cand['creativity']} sp={cand['specificity']} ot={cand['on_topic']}): "
            f"{cand['body'].splitlines()[0][:80]}",
            flush=True,
        )

    if len(picked) < args.top_k:
        print(
            f"[exemplars] only found {len(picked)} of {args.top_k} requested after dedup; lowering --dedup-jaccard or --min-composite may help",
            flush=True,
        )

    out = {
        "note": args.note,
        "source_brief": briefs_seen[0] if briefs_seen else "",
        "extracted_from": [str(p) for p in args.scores],
        "selection_criteria": {
            "min_composite": args.min_composite,
            "dedup_jaccard": args.dedup_jaccard,
            "rank": "composite, creativity, specificity, on_topic",
        },
        "exemplars": [c["body"] for c in picked],
        "exemplar_metadata": [
            {
                "i": c["i"],
                "temp": c["temp"],
                "composite": c["composite"],
                "creativity": c["creativity"],
                "specificity": c["specificity"],
                "on_topic": c["on_topic"],
                "source_md": c["source_md"],
                "rationale": c["rationale"],
            }
            for c in picked
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out} ({len(picked)} exemplars)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
