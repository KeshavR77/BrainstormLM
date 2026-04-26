"""Held-out paired eval: --use-memory vs baseline on the 50 frozen prompts."""

from __future__ import annotations

import datetime as dt
import random
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from brainstormlm.config import get_settings
from brainstormlm.eval.fixtures import HELD_OUT
from brainstormlm.feedback import evaluator
from brainstormlm.feedback.trajectories import Trajectory, append
from brainstormlm.memory import LessonStore, augmented_system, retrieve
from brainstormlm.model.base import TogetherGenerator

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _generate_and_score(
    domain: str,
    prompt: str,
    *,
    generator: TogetherGenerator,
    settings,
    temperature: float,
    system: str | None,
    memory_context_ids: list[str],
) -> Trajectory:
    response, _ = generator.generate_with_meta(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=600,
    )
    result, _ = evaluator.score(prompt, response)
    return Trajectory(
        id=str(uuid.uuid4()),
        timestamp=_utc_now_iso(),
        domain=domain,
        prompt=prompt,
        response=response,
        generator_model=settings.together_base_model,
        generator_temperature=temperature,
        prompter_model=settings.prompter_model,
        evaluator_model=settings.evaluator_model,
        rubric_version=evaluator.RUBRIC_VERSION,
        creativity=result.creativity,
        logic=result.logic,
        rationale=result.rationale,
        failure_modes=result.failure_modes,
        memory_context_ids=memory_context_ids,
    )


def _bootstrap_p_value(deltas: list[float], *, n_iterations: int, rng: random.Random) -> float:
    """One-sided paired bootstrap: H1 = mean(deltas) > 0."""
    n = len(deltas)
    if n == 0:
        return 1.0
    mean_obs = sum(deltas) / n
    centered = [d - mean_obs for d in deltas]  # center under H0
    geq = 0
    for _ in range(n_iterations):
        total = 0.0
        for _ in range(n):
            total += centered[rng.randrange(n)]
        if total / n >= mean_obs:
            geq += 1
    return geq / n_iterations


@app.command()
def main(
    out: Annotated[Path | None, typer.Option(help="Eval JSONL path (default: data/eval/held_out_eval.jsonl).")] = None,
    temperature: Annotated[float, typer.Option(help="Generator sampling temperature.")] = 1.0,
    k_success: Annotated[int, typer.Option(help="Success lessons per prompt for the memory arm.")] = 3,
    k_failure: Annotated[int, typer.Option(help="Failure lessons per prompt for the memory arm.")] = 2,
    bootstrap_iterations: Annotated[int, typer.Option(help="Paired-bootstrap resamples.")] = 10000,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed.")] = 42,
    reset: Annotated[bool, typer.Option("--reset", help="Truncate the eval JSONL before writing.")] = False,
) -> None:
    """For each held-out prompt, score baseline vs memory arms; report paired bootstrap."""
    settings = get_settings()
    out_path = out or (Path("./data/eval") / "held_out_eval.jsonl")
    if reset and out_path.exists():
        out_path.unlink()

    store = LessonStore(settings.lessons_path)
    store.load()
    if not store.lessons:
        typer.echo(
            f"Memory store is empty ({settings.lessons_path}). "
            "Run scripts/rebuild_memory.py first.",
            err=True,
        )
        raise typer.Exit(code=1)

    generator = TogetherGenerator()

    baseline_scores: list[int] = []
    memory_scores: list[int] = []
    deltas: list[float] = []

    for domain, prompt in tqdm(HELD_OUT, desc="held-out"):
        baseline = _generate_and_score(
            domain,
            prompt,
            generator=generator,
            settings=settings,
            temperature=temperature,
            system=None,
            memory_context_ids=[],
        )
        append(baseline, out_path)

        lessons = retrieve(store, prompt, k_success=k_success, k_failure=k_failure)
        memory_system = augmented_system(None, lessons)
        memory = _generate_and_score(
            domain,
            prompt,
            generator=generator,
            settings=settings,
            temperature=temperature,
            system=memory_system,
            memory_context_ids=[lesson.id for lesson in lessons],
        )
        append(memory, out_path)

        b_total = baseline.creativity + baseline.logic
        m_total = memory.creativity + memory.logic
        baseline_scores.append(b_total)
        memory_scores.append(m_total)
        deltas.append(float(m_total - b_total))

    n = len(deltas)
    mean_baseline = sum(baseline_scores) / n
    mean_memory = sum(memory_scores) / n
    mean_delta = sum(deltas) / n
    rng = random.Random(seed)
    p_value = _bootstrap_p_value(deltas, n_iterations=bootstrap_iterations, rng=rng)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = n - wins - losses

    typer.echo("")
    typer.echo(f"Held-out prompts: {n}")
    typer.echo(f"Mean creativity+logic — baseline: {mean_baseline:.3f}    memory: {mean_memory:.3f}")
    typer.echo(f"Mean per-prompt delta (memory - baseline): {mean_delta:+.3f}")
    typer.echo(f"Wins/Losses/Ties: {wins}/{losses}/{ties}")
    typer.echo(f"Paired bootstrap one-sided p-value (H1: memory > baseline): {p_value:.4f}")
    typer.echo(f"Eval trajectories appended to: {out_path}")


if __name__ == "__main__":
    sys.exit(app() or 0)
