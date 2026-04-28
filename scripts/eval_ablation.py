"""Run the 4-arm ablation table: base / base+mem / trained / trained+mem.

Held-out prompts MUST NOT enter the training pool. This script writes to
`data/eval/ablation_<run_id>.jsonl` (and a sibling `_summary.json`) which is
disjoint from `data/trajectories.jsonl` and `data/preferences.jsonl`.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from brainstormlm.config import get_settings
from brainstormlm.eval.fixtures import HELD_OUT
from brainstormlm.eval.harness import format_summary_table, row_to_dict, run_ablation, summarize
from brainstormlm.feedback import evaluator
from brainstormlm.memory import LessonStore
from brainstormlm.model import HFGenerator, TogetherGenerator, make_generator

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _score_with_evaluator(prompt: str, response: str):
    result, _ = evaluator.score(prompt, response)
    return result


@app.command()
def main(
    adapter: Annotated[Path, typer.Option(help="PEFT adapter directory for the trained arms.")],
    run_id: Annotated[str | None, typer.Option(help="Identifier for this eval run (default: random hex).")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Eval output dir (default: data/eval).")] = None,
    k_success: Annotated[int, typer.Option(help="Success lessons per memory-arm prompt.")] = 3,
    k_failure: Annotated[int, typer.Option(help="Failure lessons per memory-arm prompt.")] = 2,
    temperature: Annotated[float, typer.Option(help="Generator sampling temperature.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed.")] = 42,
    bootstrap_iterations: Annotated[int, typer.Option(help="Paired-bootstrap resamples.")] = 10000,
    skip_memory: Annotated[bool, typer.Option("--skip-memory", help="Run only base + trained arms (no LessonStore needed).")] = False,
    steering_path: Annotated[Path | None, typer.Option(help="Apply steering vector to both base and trained generators (rename arms to steered_*).")] = None,
    steering_strength: Annotated[float | None, typer.Option(help="Override the steering strength multiplier.")] = None,
) -> None:
    """Score every HELD_OUT prompt under each arm; print + persist the table."""
    settings = get_settings()
    rid = run_id or uuid.uuid4().hex[:12]
    out = out_dir or (settings.data_dir / "eval")
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"ablation_{rid}.jsonl"
    summary_path = out / f"ablation_{rid}_summary.json"

    if not adapter.exists():
        typer.echo(f"Adapter directory not found: {adapter}", err=True)
        raise typer.Exit(code=2)

    store: LessonStore | None = None
    if not skip_memory:
        store = LessonStore(settings.lessons_path)
        store.load()
        if not store.lessons:
            typer.echo(
                f"Memory store is empty ({settings.lessons_path}). "
                "Run scripts/rebuild_memory.py first, or pass --skip-memory.",
                err=True,
            )
            raise typer.Exit(code=1)

    effective_strength = (
        steering_strength
        if steering_strength is not None
        else (settings.default_steering_strength if steering_path is not None else None)
    )
    if steering_path is not None:
        base_generator = make_generator(
            None, steering_path=steering_path, steering_strength=effective_strength
        )
        trained_generator = make_generator(
            adapter, steering_path=steering_path, steering_strength=effective_strength
        )
        arm_label_prefix = "steered_"
    else:
        base_generator = TogetherGenerator()
        trained_generator = HFGenerator(adapter_path=adapter)
        arm_label_prefix = ""

    typer.echo(f"Run ID: {rid}")
    typer.echo(f"Held-out prompts: {len(HELD_OUT)}")
    typer.echo(f"Adapter: {adapter}")
    if steering_path is not None:
        typer.echo(f"Steering: {steering_path} (strength={effective_strength})")
    arms_label = "base, trained" if skip_memory else "base, base+mem, trained, trained+mem"
    typer.echo(f"Arms: {arm_label_prefix}{{{arms_label}}}")
    typer.echo("")

    rows = []
    for arm_rows in _arm_iterator(
        HELD_OUT,
        base_generator=base_generator,
        trained_generator=trained_generator,
        store=store,
        adapter_path=adapter,
        k_success=k_success,
        k_failure=k_failure,
        temperature=temperature,
        arm_label_prefix=arm_label_prefix,
    ):
        rows.extend(arm_rows)

    with rows_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(row_to_dict(r), sort_keys=True) + "\n")

    summaries = summarize(rows, n_iterations=bootstrap_iterations, seed=seed)
    typer.echo(format_summary_table(summaries))

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": rid,
                "adapter": str(adapter),
                "n_prompts": len(HELD_OUT),
                "summaries": [
                    {
                        "arm": s.arm,
                        "n": s.n,
                        "mean_creativity": s.mean_creativity,
                        "mean_logic": s.mean_logic,
                        "mean_total": s.mean_total,
                        "mean_delta_vs_base": s.mean_delta_vs_base,
                        "p_value": s.p_value,
                    }
                    for s in summaries
                ],
            },
            f,
            indent=2,
        )

    typer.echo("")
    typer.echo(f"Rows written to: {rows_path}")
    typer.echo(f"Summary written to: {summary_path}")


def _arm_iterator(
    held_out,
    *,
    base_generator,
    trained_generator,
    store,
    adapter_path,
    k_success,
    k_failure,
    temperature,
    arm_label_prefix,
):
    """Wrap run_ablation with a tqdm bar — just for the user's progress signal."""
    with tqdm(total=len(held_out) * (4 if store is not None else 2), desc="ablation") as bar:
        rows = run_ablation(
            held_out,
            base_generator=_TqdmGen(base_generator, bar),
            trained_generator=_TqdmGen(trained_generator, bar),
            score_fn=_score_with_evaluator,
            store=store,
            adapter_path=adapter_path,
            k_success=k_success,
            k_failure=k_failure,
            temperature=temperature,
            arm_label_prefix=arm_label_prefix,
        )
    yield rows


class _TqdmGen:
    """Wraps a generator to bump the tqdm bar on every call."""

    def __init__(self, inner, bar) -> None:
        self._inner = inner
        self._bar = bar

    def generate_with_meta(self, prompt: str, *, system, temperature, max_tokens):
        result = self._inner.generate_with_meta(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self._bar.update(1)
        return result


if __name__ == "__main__":
    sys.exit(app() or 0)
