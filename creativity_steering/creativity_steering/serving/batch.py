"""Batch steer -> refine over a set of briefs, with per-sample checkpointing + resume.

For each brief, generate K steered samples (varying the seed), optionally an unsteered baseline
(for A/B), and optionally a concept-frozen refine of each steered sample. Every sample is appended
to ``<out_dir>/<brief_id>.jsonl`` the instant it completes, so a crash or SSH disconnect never
loses progress and a re-run resumes exactly where it stopped (the RunPod-Lessons checkpoint
discipline — the run's own checkpoints are the source of truth, not a fragile watcher).

Fault isolation for long unattended runs: a failure on one sample never aborts the whole run.
A generation failure records an error row (so resume skips the poison sample instead of looping);
a refine/baseline failure degrades gracefully, keeping the steered draft. ``run_briefs`` returns
per-brief counts including ``errors`` so the caller can surface how many samples were lost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_prompt(brief: dict, lens: str | None = None) -> str:
    """Assemble the model prompt: the brief text, an optional archetype *lens* (rotated across
    samples to stop the set mode-collapsing), and an explicit cliché-exclusion line so the
    diversity edge is visible (the model is forced off the obvious defaults)."""
    text = brief["brief"]
    if lens:
        text += ("\n\nExplore it specifically through this lens (do not name the lens literally): "
                 f"{lens}.")
    excludes = brief.get("motif_excludes") or []
    if excludes:
        text += "\n\nAvoid these over-used treatments entirely: " + ", ".join(excludes) + "."
    return text


def _done_samples(path: Path) -> set[int]:
    """Sample indices already written to a brief's jsonl (for resume). A trailing partial line
    (process killed mid-write) is unparseable and skipped, so that sample simply re-runs; we warn
    so the operator knows a resume re-did work rather than silently dropping it."""
    done: set[int] = set()
    if not path.exists():
        return done
    bad = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(int(json.loads(line)["sample"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            bad += 1
    if bad:
        print(f"[resume] {path.name}: skipped {bad} unparseable line(s); those samples re-run",
              file=sys.stderr)
    return done


def run_briefs(
    service,
    briefs,
    *,
    variant: str = "L7",
    samples: int = 8,
    out_dir: str = "results/designer_run",
    refine: bool = False,
    baseline: bool = False,
    rotate_seeds: bool = True,
    max_new_tokens: int | None = None,
    base_seed: int = 0,
    progress=print,
) -> list[dict]:
    """Run every brief x sample through the service, checkpointing each completed sample.

    When a brief carries ``archetype_seeds`` and ``rotate_seeds`` is set, sample i is anchored to
    lens ``seeds[i % len(seeds)]`` so the set spans distinct conceptual territories instead of
    collapsing. Re-running with the same ``out_dir`` skips samples already on disk, so an
    interrupted run resumes simply by launching it again."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for brief in briefs:
        bid = brief["id"]
        seeds = brief.get("archetype_seeds") or []
        jpath = out / f"{bid}.jsonl"
        done = _done_samples(jpath)
        made = 0
        errors = 0
        with jpath.open("a") as f:
            for i in range(samples):
                if i in done:
                    continue
                lens = seeds[i % len(seeds)] if (rotate_seeds and seeds) else None
                prompt = build_prompt(brief, lens)
                seed = base_seed + i
                row = {"brief_id": bid, "sample": i, "seed": seed, "variant": variant, "lens": lens}

                # Generation is the expensive GPU work; a failure here (OOM, kernel hiccup) records
                # an error row so resume skips it rather than looping on a poison sample.
                try:
                    row["steered"] = service.generate(
                        prompt, variant=variant, steered=True,
                        max_new_tokens=max_new_tokens, seed=seed,
                    )
                except Exception as exc:
                    row["steered"] = None
                    row["error"] = f"generate: {exc}"
                    _append(f, row)
                    errors += 1
                    progress(f"[{bid}] sample {i + 1}/{samples} GENERATION FAILED: {exc} "
                             f"(recorded, continuing)")
                    continue

                # Refine and baseline are best-effort: a failure keeps the steered draft rather
                # than losing the sample. The refiner retries transient API errors internally.
                if refine:
                    try:
                        row["refined"] = service.refine_text(prompt, row["steered"])
                    except Exception as exc:
                        row["refined"] = None
                        row["refine_error"] = str(exc)
                        progress(f"[{bid}] sample {i + 1}/{samples} refine failed: {exc} "
                                 f"(kept steered)")
                if baseline:
                    try:
                        row["baseline"] = service.generate(
                            prompt, steered=False, max_new_tokens=max_new_tokens, seed=seed,
                        )
                    except Exception as exc:
                        row["baseline"] = None
                        row["baseline_error"] = str(exc)

                _append(f, row)
                made += 1
                progress(f"[{bid}] sample {i + 1}/{samples} (seed {seed}, "
                         f"lens={lens[:18] if lens else 'none'}) done")
        summary.append({"brief_id": bid, "made": made, "errors": errors, "total": samples,
                        "path": str(jpath)})
    _write_markdown(out, briefs, refine=refine)
    return summary


def _append(f, row: dict) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def _write_markdown(out: Path, briefs, *, refine: bool) -> None:
    """Render the per-brief jsonl into one human-readable IMAGE_PROMPTS.md (the prompts to paste
    into an image-gen platform — refined when available, else the raw steered text)."""
    lines = ["# Designer run — image prompts\n",
             "_Paste a direction into your image-gen platform. Refined = concept-frozen polish "
             "over the steered draft._\n"]
    for brief in briefs:
        bid = brief["id"]
        jpath = out / f"{bid}.jsonl"
        if not jpath.exists():
            continue
        rows = [json.loads(line) for line in jpath.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: r.get("sample", 0))
        lines.append(f"\n## {bid}  ({brief.get('category', '')})\n")
        lines.append(f"_{brief.get('brief', '')}_\n")
        for r in rows:
            n = r.get("sample", 0) + 1
            text = (r.get("refined") if refine and r.get("refined") else r.get("steered")) \
                or "_(generation failed — see the jsonl error field)_"
            lens = r.get("lens")
            tag = f"  ·  lens: {lens.split(' — ')[0]}" if lens else ""
            lines.append(f"\n**Direction {n}**  ·  seed {r.get('seed')}  ·  {r.get('variant')}{tag}\n")
            lines.append(f"{text}\n")
    (out / "IMAGE_PROMPTS.md").write_text("\n".join(lines))
