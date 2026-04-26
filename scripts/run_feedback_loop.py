"""Drive N (prompter → generator → evaluator) trials and append to JSONL."""

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
from brainstormlm.feedback import evaluator, pricing, prompter
from brainstormlm.feedback.trajectories import Trajectory, append
from brainstormlm.memory import LessonStore, augmented_system, retrieve
from brainstormlm.model.base import TogetherGenerator

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@app.command()
def main(
    n: Annotated[int, typer.Option(help="Number of trajectories to produce.")] = 200,
    temperature: Annotated[float, typer.Option(help="Generator sampling temperature.")] = 1.0,
    max_calls: Annotated[int, typer.Option(help="Hard cap on completed trajectories.")] = 1000,
    max_spend_usd: Annotated[float, typer.Option(help="Abort if cumulative cost exceeds this.")] = 1.00,
    out: Annotated[Path | None, typer.Option(help="Trajectory JSONL path.")] = None,
    seed: Annotated[int, typer.Option(help="Seed for taxonomy shuffle.")] = 42,
    use_memory: Annotated[bool, typer.Option("--use-memory", help="Inject ReasoningBank lessons into each generator call.")] = False,
    k_success: Annotated[int, typer.Option(help="Success lessons retrieved when --use-memory is on.")] = 3,
    k_failure: Annotated[int, typer.Option(help="Failure lessons retrieved when --use-memory is on.")] = 2,
) -> None:
    settings = get_settings()
    out_path = out or settings.trajectories_path

    estimate = pricing.estimate_run_cost(n)
    typer.echo(f"Pre-flight estimate for n={n}: ${estimate:.4f}")
    if estimate > max_spend_usd:
        typer.echo(
            f"Aborting: estimate ${estimate:.4f} exceeds --max-spend-usd ${max_spend_usd:.4f}.",
            err=True,
        )
        raise typer.Exit(code=2)

    held_out_prompts = {p for _, p in HELD_OUT}

    rng = random.Random(seed)
    domains = list(prompter.DOMAINS)
    rng.shuffle(domains)

    generator = TogetherGenerator()
    cap = min(n, max_calls)

    store: LessonStore | None = None
    if use_memory:
        store = LessonStore(settings.lessons_path)
        store.load()

    completed = 0
    total_cost = 0.0
    creativity_sum = 0
    logic_sum = 0
    cost_breakdown = {"prompter": 0.0, "generator": 0.0, "evaluator": 0.0}

    try:
        for i in tqdm(range(cap), desc="trials"):
            domain = domains[i % len(domains)]
            salt = i // len(domains)

            prompt_text, p_usage = prompter.generate_prompt(domain, salt=salt)
            assert prompt_text not in held_out_prompts, (
                f"Prompter produced a held-out prompt: {prompt_text!r}"
            )
            p_cost = (
                0.0
                if p_usage.cached
                else pricing.cost_of(settings.prompter_model, p_usage.prompt_tokens, p_usage.completion_tokens)
            )

            memory_context_ids: list[str] = []
            trial_system: str | None = None
            if store is not None:
                lessons = retrieve(
                    store, prompt_text, k_success=k_success, k_failure=k_failure
                )
                trial_system = augmented_system(None, lessons)
                memory_context_ids = [lesson.id for lesson in lessons]

            response, g_cached = generator.generate_with_meta(
                prompt_text,
                system=trial_system,
                temperature=temperature,
                max_tokens=600,
            )
            g_cost = (
                0.0
                if g_cached
                else pricing.cost_of(
                    settings.together_base_model,
                    len(prompt_text) // 4 + 50,
                    len(response) // 4 + 1,
                )
            )

            result, e_usage = evaluator.score(prompt_text, response)
            e_cost = (
                0.0
                if e_usage.cached
                else pricing.cost_of(settings.evaluator_model, e_usage.prompt_tokens, e_usage.completion_tokens)
            )

            trial_cost = p_cost + g_cost + e_cost
            total_cost += trial_cost
            cost_breakdown["prompter"] += p_cost
            cost_breakdown["generator"] += g_cost
            cost_breakdown["evaluator"] += e_cost

            t = Trajectory(
                id=str(uuid.uuid4()),
                timestamp=_utc_now_iso(),
                domain=domain,
                prompt=prompt_text,
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
            append(t, out_path)

            completed += 1
            creativity_sum += result.creativity
            logic_sum += result.logic

            if total_cost > max_spend_usd:
                typer.echo(
                    f"\nAborting after {completed} trials: cumulative cost "
                    f"${total_cost:.4f} exceeded --max-spend-usd ${max_spend_usd:.4f}.",
                    err=True,
                )
                break
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user.", err=True)

    if completed == 0:
        typer.echo("No trajectories produced.", err=True)
        raise typer.Exit(code=1)

    mean_c = creativity_sum / completed
    mean_l = logic_sum / completed
    typer.echo("")
    typer.echo(f"Completed: {completed} / {cap}")
    typer.echo(f"Total cost: ${total_cost:.4f}")
    typer.echo(
        "Cost breakdown: "
        f"prompter ${cost_breakdown['prompter']:.4f}, "
        f"generator ${cost_breakdown['generator']:.4f}, "
        f"evaluator ${cost_breakdown['evaluator']:.4f}"
    )
    typer.echo(f"Mean creativity: {mean_c:.2f}    Mean logic: {mean_l:.2f}")
    typer.echo(f"Trajectories appended to: {out_path}")


if __name__ == "__main__":
    sys.exit(app() or 0)
