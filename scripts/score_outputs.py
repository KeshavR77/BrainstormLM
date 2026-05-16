#!/usr/bin/env python3
"""Quality scoring pass for smoke outputs.

Reads a smoke artifact ``.md`` file produced by ``smoke_classic_v17.py`` or
``smoke_cognitive_anti_mode.py``, extracts the per-arm/per-idea completions, and
calls GPT-4o-mini with a 4-axis rubric:

- creativity   (1-5): novel, surprising, distinctive
- coherence    (1-5): the concept hangs together
- specificity  (1-5): concrete & actionable rather than generic
- on_topic     (1-5): actually responds to the brief

Writes a per-arm aggregate table and per-idea scores back as a sibling
``.scores.json`` + appends a summary section to a sibling ``.scores.md``.

Cost: ~$0.001 per concept (~$0.05 for a typical 5+5+5 smoke). Uses the
shrooms-llm repo's ``.env`` ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_env() -> None:
    """Load OPENAI_API_KEY from any of the usual local locations.

    Looks first in this repo's .env, then falls back to the sibling shrooms-llm/.env
    (the editable model dep) so the BrainstormLM port doesn't require duplicating
    secrets across repos.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (ROOT / ".env", ROOT.parent / "shrooms-llm" / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)


_ARM_HEADER_RX = re.compile(r"^##\s+(.+?)\s+outputs.*$", re.MULTILINE)
_IDEA_HEADER_RX = re.compile(r"^###\s+Idea\s+(\d+)\s*\(T=([\d.]+)\)\s*$", re.MULTILINE)
_FENCE_RX = re.compile(r"```(?:[a-z]*\n)?(.*?)```", re.DOTALL)


def _arm_name_from_header(header: str) -> str:
    """Map a markdown section header to a short arm key."""
    s = header.strip().lower()
    if "pristine" in s and "v17" in s:
        return "v17-pristine"
    if "anti-mode" in s or "anti mode" in s:
        return "v17-antimode"
    if "v17 classic" in s:
        return "v17-classic"
    if "v17 +" in s and "iter" in s:
        return "v17-iter"
    if s.startswith("baseline"):
        return "baseline"
    if "v17" in s:
        return "v17"
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _parse_smoke_md(path: Path) -> dict[str, list[dict]]:
    """Return {arm_name: [{i, temp, text}, ...]} from a smoke markdown file."""
    text = path.read_text(encoding="utf-8")
    arms: dict[str, list[dict]] = {}
    arm_headers = list(_ARM_HEADER_RX.finditer(text))
    if not arm_headers:
        raise ValueError(f"No '## <arm> outputs' section headers found in {path}")
    boundaries = [h.start() for h in arm_headers] + [len(text)]
    for idx, header_match in enumerate(arm_headers):
        arm = _arm_name_from_header(header_match.group(1))
        section = text[boundaries[idx] : boundaries[idx + 1]]
        # Find every "### Idea N (T=X.XX)" followed by a fenced block.
        idea_matches = list(_IDEA_HEADER_RX.finditer(section))
        for j, im in enumerate(idea_matches):
            i = int(im.group(1))
            temp = float(im.group(2))
            # The completion fence is the LAST fenced block before the next idea (or end of section).
            end = idea_matches[j + 1].start() if j + 1 < len(idea_matches) else len(section)
            chunk = section[im.end() : end]
            fences = _FENCE_RX.findall(chunk)
            if not fences:
                continue
            # Prefer the last fence under each idea (anti-mode emits exclusions then output).
            completion = fences[-1].strip()
            if not completion:
                continue
            arms.setdefault(arm, []).append({"i": i, "temp": temp, "text": completion})
    return arms


_SYSTEM_PROMPT = (
    "You are a brutally fair design / strategy reviewer. You score creative concepts on "
    "four 1-5 axes. Be willing to give 1s and 2s for generic or sloppy work; reserve 5s "
    "for genuinely surprising, well-formed concepts that a senior designer or strategist "
    "would actually want to use. Score the concept itself, not the writing style."
)


def _user_prompt(prompt: str, idea: str) -> str:
    return (
        f"BRIEF:\n{prompt.strip()}\n\n"
        f"CONCEPT TO SCORE:\n```\n{idea.strip()[:6000]}\n```\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "creativity": <1-5>,\n'
        '  "coherence": <1-5>,\n'
        '  "specificity": <1-5>,\n'
        '  "on_topic": <1-5>,\n'
        '  "one_line_rationale": "<one sentence, < 200 chars>"\n'
        "}\n"
        "Definitions:\n"
        " - creativity: novelty + surprise. 1 = obvious / generic. 5 = a senior designer goes 'oh, that's new'.\n"
        " - coherence: does the concept hang together as a single idea, not a list of loose elements.\n"
        " - specificity: concrete enough to act on (visual details / mechanic / numbers). 1 = handwave. 5 = could be sent to a designer or VC.\n"
        " - on_topic: does it answer the brief. 5 = direct fit. 1 = drifted."
    )


def _score_one(client, model: str, prompt: str, idea: str) -> dict:
    user = _user_prompt(prompt, idea)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        for k in ("creativity", "coherence", "specificity", "on_topic"):
            data[k] = max(1, min(5, int(round(float(data.get(k, 0))))))
        data["one_line_rationale"] = str(data.get("one_line_rationale", ""))[:300]
        data["composite"] = round(
            (data["creativity"] + data["coherence"] + data["specificity"] + data["on_topic"]) / 4.0, 3
        )
        data["ok"] = True
        return data
    except Exception as exc:
        return {
            "creativity": None,
            "coherence": None,
            "specificity": None,
            "on_topic": None,
            "composite": None,
            "one_line_rationale": f"<scoring failed: {type(exc).__name__}: {exc}>",
            "ok": False,
        }


def _extract_brief(md_text: str) -> str:
    m = re.search(r"^\*\*Prompt:\*\*\s*(.+?)$", md_text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _summarize(scores_by_arm: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for arm, rows in scores_by_arm.items():
        valid = [r for r in rows if r.get("ok") and r.get("composite") is not None]
        if not valid:
            continue
        keys = ("creativity", "coherence", "specificity", "on_topic", "composite")
        out[arm] = {k: round(sum(r[k] for r in valid) / len(valid), 3) for k in keys}
        out[arm]["n"] = len(valid)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md", type=Path, required=True, help="Smoke markdown to score.")
    p.add_argument(
        "--model",
        default=os.environ.get("CONCEPTIAL_SCORE_MODEL", "gpt-4o-mini"),
        help="OpenAI model (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override scores JSON output path (default: alongside .md).",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Override scores summary markdown path (default: alongside .md).",
    )
    args = p.parse_args()

    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Aborting.", file=sys.stderr)
        return 1

    if not args.md.is_file():
        print(f"Smoke MD not found: {args.md}", file=sys.stderr)
        return 1

    md_text = args.md.read_text(encoding="utf-8")
    brief = _extract_brief(md_text)
    if not brief:
        print(f"Could not extract brief (**Prompt:** ...) from {args.md}", file=sys.stderr)
        return 1

    arms = _parse_smoke_md(args.md)
    if not arms:
        print(f"No arm outputs parsed from {args.md}", file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI()

    scores_by_arm: dict[str, list[dict]] = {}
    t0 = time.time()
    for arm, ideas in arms.items():
        print(f"[score] arm={arm} n={len(ideas)}", flush=True)
        scored: list[dict] = []
        for o in ideas:
            score = _score_one(client, args.model, brief, o["text"])
            scored.append({**score, "i": o["i"], "temp": o["temp"]})
            print(
                f"  idea {o['i']} (T={o['temp']:.2f}): "
                f"cr={score.get('creativity')} co={score.get('coherence')} "
                f"sp={score.get('specificity')} ot={score.get('on_topic')} "
                f"composite={score.get('composite')}",
                flush=True,
            )
        scores_by_arm[arm] = scored
    elapsed = time.time() - t0

    summary = _summarize(scores_by_arm)
    out_json = args.out_json or args.md.with_suffix(".scores.json")
    out_md = args.out_md or args.md.with_suffix(".scores.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "model": args.model,
                "source_md": str(args.md),
                "brief": brief,
                "elapsed_s": round(elapsed, 1),
                "per_arm_summary": summary,
                "per_idea": scores_by_arm,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Compact markdown summary
    lines = [
        f"# Quality scores — {args.md.name}",
        "",
        f"**Brief:** {brief}",
        f"**Judge:** {args.model}",
        f"**Elapsed:** {elapsed:.1f}s",
        "",
        "## Per-arm aggregates (mean across N samples)",
        "",
        "| Arm | N | creativity | coherence | specificity | on_topic | composite |",
        "|-----|--:|-----------:|----------:|------------:|---------:|----------:|",
    ]
    for arm, agg in summary.items():
        lines.append(
            f"| {arm} | {agg['n']} | {agg['creativity']:.2f} | {agg['coherence']:.2f} | "
            f"{agg['specificity']:.2f} | {agg['on_topic']:.2f} | **{agg['composite']:.2f}** |"
        )
    lines.extend(["", "## Per-idea scores", ""])
    for arm, rows in scores_by_arm.items():
        lines.extend([f"### {arm}", "", "| i | T | cr | co | sp | ot | comp | rationale |", "|--:|--:|---:|---:|---:|---:|-----:|-----------|"])
        for r in rows:
            rat = (r.get("one_line_rationale") or "").replace("|", "/")
            lines.append(
                f"| {r['i']} | {r['temp']:.2f} | {r.get('creativity')} | {r.get('coherence')} | "
                f"{r.get('specificity')} | {r.get('on_topic')} | {r.get('composite')} | {rat} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}\nWrote {out_md}", flush=True)
    print("\n[SUMMARY]")
    for arm, agg in summary.items():
        print(
            f"  {arm:>18s}  composite={agg['composite']:.2f} "
            f"(cr={agg['creativity']:.2f} co={agg['coherence']:.2f} "
            f"sp={agg['specificity']:.2f} ot={agg['on_topic']:.2f}, n={agg['n']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
