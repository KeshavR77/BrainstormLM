"""Re-extract ReasoningBank lessons from data/trajectories.jsonl."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from brainstormlm.config import get_settings
from brainstormlm.feedback import pricing
from brainstormlm.feedback.trajectories import read_all
from brainstormlm.memory import LessonStore
from brainstormlm.memory.lessons import extract_lessons

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    trajectories: Annotated[Path | None, typer.Option(help="Trajectory JSONL path.")] = None,
    out: Annotated[Path | None, typer.Option(help="Lessons JSONL path.")] = None,
    max_trajectories: Annotated[int, typer.Option(help="Cap on trajectories processed.")] = 1000,
    max_spend_usd: Annotated[float, typer.Option(help="Abort if cumulative cost exceeds this.")] = 0.50,
    reset: Annotated[bool, typer.Option("--reset", help="Truncate lessons.jsonl before extracting.")] = False,
) -> None:
    settings = get_settings()
    trajectories_path = trajectories or settings.trajectories_path
    out_path = out or settings.lessons_path

    if not trajectories_path.exists():
        typer.echo(f"No trajectories at {trajectories_path}", err=True)
        raise typer.Exit(code=1)

    all_trajectories = list(read_all(trajectories_path))
    cap = min(max_trajectories, len(all_trajectories))
    if cap == 0:
        typer.echo("Trajectory file is empty.", err=True)
        raise typer.Exit(code=1)

    estimate = pricing.estimate_extraction_cost(cap)
    typer.echo(f"Pre-flight estimate for {cap} trajectories: ${estimate:.4f}")
    if estimate > max_spend_usd:
        typer.echo(
            f"Aborting: estimate ${estimate:.4f} exceeds --max-spend-usd ${max_spend_usd:.4f}.",
            err=True,
        )
        raise typer.Exit(code=2)

    if reset and out_path.exists():
        out_path.unlink()

    store = LessonStore(out_path)
    store.load()

    processed = 0
    new_lessons = 0
    total_cost = 0.0

    try:
        for t in tqdm(all_trajectories[:cap], desc="extract"):
            lessons = extract_lessons(t)
            added = store.add_many(lessons)
            new_lessons += added
            processed += 1
            # Rough running tally — embed cost is dominated by the extractor call,
            # which the cache may have served. We use the budget estimate per call
            # as an upper bound to be conservative.
            total_cost += pricing.estimate_extraction_cost(1)
            if total_cost > max_spend_usd:
                typer.echo(
                    f"\nAborting after {processed}: cumulative estimate "
                    f"${total_cost:.4f} exceeded --max-spend-usd ${max_spend_usd:.4f}.",
                    err=True,
                )
                break
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user.", err=True)

    typer.echo("")
    typer.echo(f"Processed: {processed} / {cap} trajectories")
    typer.echo(f"New lessons added: {new_lessons}")
    typer.echo(f"Total lessons in store: {len(store.lessons)}")
    typer.echo(f"Estimated cost upper bound: ${total_cost:.4f}")
    typer.echo(f"Lessons written to: {out_path}")


if __name__ == "__main__":
    sys.exit(app() or 0)
