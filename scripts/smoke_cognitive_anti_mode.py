#!/usr/bin/env python3
"""Conceptial / Corpius 1 cognitive-anti-mode smoke runner.

The CLI front-end for the locked Conceptial demo configuration. The prompt-engineering
logic and concept-name extraction live in :mod:`brainstormlm.conceptial`; this script
just wires CLI args → model → reporting.

**Locked stack:**

- ``Qwen/Qwen2.5-3B-Instruct``
- v17 purified L14 + L18 steering (no V3.5–V5 IOR / ghost / cognitive-route)
- ``min_p=0.01`` for wide-tail sampling
- Cognitive anti-mode at PROMPT level
- Optional cross-session persistence via :class:`IdeaMemoryStore`
- Optional few-shot exemplars (in-context demos with auto-anti-copy)
- Brief-aware ``--output-mode`` ({image-prompt, strategy, generic})

Run from the BrainstormLM repo (uses its venv + editable shrooms-llm install)::

    python3 scripts/smoke_cognitive_anti_mode.py \\
        --prompt "Give me a creative logo for an engineering club at Stanford." \\
        --n 5 --base-temp 0.8 --temp-jitter 0.05 --min-p 0.01 \\
        --output-mode image-prompt \\
        --project-id my-demo-project \\
        --exemplars-json artifacts/conceptial/exemplars_auto_logo.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# shrooms_llm is editable-installed in BrainstormLM via [tool.uv.sources]; idea_store
# + anti_mode logic live in BrainstormLM/src/brainstormlm/conceptial/
from brainstormlm.conceptial import (
    CAPTURE_LAYER,
    DEFAULT_PROMPT,
    OUTPUT_MODE_INSTRUCTIONS,
    IdeaMemoryStore,
    SOURCE_GENERATED,
    extract_summary,
    format_anti_mode,
    format_pristine,
)
from brainstormlm.conceptial.engine.config import SteeringConfig
from brainstormlm.conceptial.engine.core.hidden_capture import last_token_hidden_at_layer, normalize_hidden
from brainstormlm.conceptial.engine.corpius import apply_corpius_1, load_corpius_spec
from brainstormlm.conceptial.engine.pipeline import SteeringEngine
from brainstormlm.conceptial.engine.steering_presets import apply_local_unsteered

ROOT = Path(__file__).resolve().parents[1]


def _heuristics(text: str) -> dict:
    text = text or ""
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "commas_per_word": (text.count(",") / max(1, len(words))),
        "non_ascii_frac": (sum(1 for c in text if ord(c) > 127) / max(1, len(text))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--base-temp", type=float, default=0.8)
    p.add_argument("--temp-jitter", type=float, default=0.05)
    p.add_argument("--min-p", type=float, default=0.01)
    p.add_argument(
        "--also-run-pristine",
        action="store_true",
        default=False,
        help="Also run v17 without anti-mode (comparison arm). Default off (saves wall-clock).",
    )
    p.add_argument("--no-pristine", dest="also_run_pristine", action="store_false")
    p.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "artifacts" / "conceptial" / "ANTIMODE_SMOKE.md",
    )
    p.add_argument(
        "--project-id",
        default=None,
        help=(
            "Persist generated concepts to the IdeaMemoryStore under this project_id and "
            "load prior persisted summaries as initial exclusions."
        ),
    )
    p.add_argument(
        "--reset-project",
        action="store_true",
        default=False,
        help="Delete every prior row for --project-id before this session.",
    )
    p.add_argument(
        "--exemplars-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with shape {'exemplars': [text1, text2, ...]}. Each is injected "
            "as a prior user→assistant turn (in-context demo). Their concept names are auto-added "
            "to the anti-mode exclusion list to prevent verbatim copying."
        ),
    )
    p.add_argument(
        "--output-mode",
        choices=tuple(OUTPUT_MODE_INSTRUCTIONS.keys()),
        default="image-prompt",
        help=(
            "Brief-aware output-format instruction. 'image-prompt' (default): visual concept "
            "for image-gen pipelines — for logo/design briefs. 'strategy': structured "
            "strategic content — for GTM/R&D/business briefs. 'generic': minimal stay-on-brief."
        ),
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override SQLite path for the idea store (default: env CONCEPTIAL_IDEA_DB_PATH or ./data/idea_memory.db).",
    )
    args = p.parse_args()

    exemplars: list[str] = []
    if args.exemplars_json:
        if not args.exemplars_json.is_file():
            print(f"--exemplars-json not found: {args.exemplars_json}", file=sys.stderr)
            return 1
        data = json.loads(args.exemplars_json.read_text(encoding="utf-8"))
        exemplars = [str(x) for x in data.get("exemplars", []) if str(x).strip()]
        print(f"[exemplars] loaded {len(exemplars)} from {args.exemplars_json}", flush=True)

    print("[load] Qwen2.5-3B-Instruct + v17 purified vectors", flush=True)
    cfg = SteeringConfig(model_id="Qwen/Qwen2.5-3B-Instruct")
    engine = SteeringEngine(cfg)
    engine.load()
    spec = load_corpius_spec()
    apply_corpius_1(engine, spec)
    engine.config.min_p = float(args.min_p)
    print(
        f"[cfg] v17 schedule loaded · min_p={engine.config.min_p} · base_temp={args.base_temp} "
        f"jitter={args.temp_jitter} · output_mode={args.output_mode}",
        flush=True,
    )

    # --- Arm A: pristine v17 (optional comparison) ---
    pristine_outs: list[dict] = []
    if args.also_run_pristine:
        apply_local_unsteered(engine)
        t0 = time.time()
        for i in range(args.n):
            temp = float(args.base_temp + args.temp_jitter * i)
            text = engine.complete(
                format_pristine(args.prompt, exemplars=exemplars, output_mode=args.output_mode),
                max_new_tokens=int(args.max_new_tokens),
                temperature=temp,
                do_sample=True,
                do_steering=False,
                use_ghost_rejection=False,
            )
            pristine_outs.append({"i": i + 1, "temp": temp, "text": text})
            print(f"[prist {i+1}/{args.n}] T={temp:.2f} {len(text)} chars", flush=True)
        print(f"[prist] elapsed={time.time() - t0:.1f}s", flush=True)
        # restore v17 for the next arm
        apply_corpius_1(engine, spec)

    # --- Arm B: v17 + cognitive anti-mode (with optional cross-session persistence + few-shot) ---
    anti_outs: list[dict] = []
    prior_summaries: list[str] = []
    store: IdeaMemoryStore | None = None
    project_id = (args.project_id or "").strip() or None
    if project_id:
        store = IdeaMemoryStore(args.db_path)
        if args.reset_project:
            n_removed = store.delete_project(project_id)
            print(f"[anti ] --reset-project: cleared {n_removed} prior rows for {project_id!r}", flush=True)
        prior_rows = store.list_recent(project_id, limit=64)
        prior_summaries = [r.text for r in prior_rows if r.text][::-1]  # oldest → newest
        print(
            f"[anti ] persistent project={project_id!r} hydrated {len(prior_summaries)} prior summaries from {store.path}",
            flush=True,
        )
    n_seeded = len(prior_summaries)

    t1 = time.time()
    for i in range(args.n):
        temp = float(args.base_temp + args.temp_jitter * i)
        rewritten = format_anti_mode(args.prompt, prior_summaries, exemplars=exemplars, output_mode=args.output_mode)
        text = engine.complete(
            rewritten,
            max_new_tokens=int(args.max_new_tokens),
            temperature=temp,
            do_sample=True,
            do_steering=True,
            use_ghost_rejection=False,
        )
        summary = extract_summary(text)
        anti_outs.append(
            {
                "i": i + 1,
                "temp": temp,
                "text": text,
                "summary": summary,
                "exclusions": list(prior_summaries),
            }
        )
        if summary:
            prior_summaries.append(summary)
            if store and project_id:
                try:
                    h_l14 = last_token_hidden_at_layer(
                        engine.model, engine.tokenizer, rewritten + text, CAPTURE_LAYER
                    )
                    h_unit = normalize_hidden(h_l14)
                    idea_id = f"antimode-{int(time.time())}-{i+1:03d}"
                    store.add(project_id, idea_id, h_unit.detach().cpu(), summary, source=SOURCE_GENERATED)
                except Exception as exc:
                    print(f"[anti  {i+1}/{args.n}] store.add failed (non-fatal): {exc!r}", flush=True)
        n_chars = len(text)
        print(
            f"[anti  {i+1}/{args.n}] T={temp:.2f} {n_chars} chars · summary='{(summary or '(no-summary)')[:60]}' · "
            f"now_excluding={len(prior_summaries)} (seeded={n_seeded})",
            flush=True,
        )
    print(f"[anti ] elapsed={time.time() - t1:.1f}s", flush=True)

    # --- Metrics ---
    from creativity_evaluator import CreativityEvaluator
    from creativity_evaluator.metrics.structural import vendi_score

    ce = CreativityEvaluator()
    vendi_anti = float(vendi_score(ce.embed([o["text"] for o in anti_outs]))) if len(anti_outs) >= 2 else float("nan")
    vendi_prist = (
        float(vendi_score(ce.embed([o["text"] for o in pristine_outs])))
        if args.also_run_pristine and len(pristine_outs) >= 2
        else float("nan")
    )

    arms: list[tuple[str, list[dict]]] = []
    if args.also_run_pristine:
        arms.append(("v17-pristine", pristine_outs))
    arms.append(("v17-antimode", anti_outs))
    for _arm, outs in arms:
        for o in outs:
            o["heuristics"] = _heuristics(o["text"])

    vendi_summary = ""
    if args.also_run_pristine:
        vendi_summary += f"v17-pristine={vendi_prist:.4f} · "
    vendi_summary += f"v17-antimode={vendi_anti:.4f}"

    lines = [
        "# Conceptial cognitive-anti-mode smoke",
        "",
        f"**Prompt:** {args.prompt}",
        f"**Samples per arm:** {args.n}",
        f"**Aggregate Vendi (MiniLM):** {vendi_summary}",
        f"**Config:** v17 purified L14+L18 · min_p={args.min_p} · output-mode={args.output_mode}",
    ]
    if exemplars:
        lines.append(f"**Few-shot exemplars:** {len(exemplars)} loaded from {args.exemplars_json}")
    if project_id:
        lines.append(f"**Project:** {project_id} (seeded={n_seeded} prior summaries hydrated from disk)")
    lines.extend(
        [
            "",
            "## Per-idea heuristics",
            "",
            "| Arm | i | T | chars | words | commas/word | non-ascii frac |",
            "|-----|--:|--:|------:|------:|------------:|---------------:|",
        ]
    )
    for arm_name, outs in arms:
        for o in outs:
            h = o["heuristics"]
            lines.append(
                f"| {arm_name} | {o['i']} | {o['temp']:.2f} | {h['chars']} | {h['words']} | "
                f"{h['commas_per_word']:.3f} | {h['non_ascii_frac']:.3f} |"
            )

    if args.also_run_pristine:
        lines.extend(["", "---", "", "## v17 pristine outputs", ""])
        for o in pristine_outs:
            lines.extend([f"### Idea {o['i']} (T={o['temp']:.2f})", "", "```", o["text"].strip(), "```", ""])
    lines.extend(["", "---", "", "## v17 + cognitive anti-mode outputs", ""])
    for o in anti_outs:
        excl = o.get("exclusions") or []
        excl_str = "\n".join(f"  - {e}" for e in excl) if excl else "  (none — first sample)"
        lines.extend(
            [
                f"### Idea {o['i']} (T={o['temp']:.2f})",
                "",
                "_Exclusions presented in prompt:_",
                f"```\n{excl_str}\n```",
                "_Output:_",
                "```",
                o["text"].strip(),
                "```",
                "",
            ]
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_md}", flush=True)
    print(vendi_summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
