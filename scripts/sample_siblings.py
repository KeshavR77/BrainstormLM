"""Generate N siblings per prompt, score them, and emit preference pairs.

Mirrors `scripts/run_feedback_loop.py` for cost guardrails, tqdm progress,
typer CLI, and per-component cost tracking. One pair per prompt at most
(best vs worst with a minimum score gap).
"""

from __future__ import annotations

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
from brainstormlm.model import make_generator
from brainstormlm.training.preferences import (
    append,
    mine_pairs,
    read_all,
    sample_and_score_siblings,
)

app = typer.Typer(add_completion=False, no_args_is_help=False)


DEFAULT_TEMPERATURES = "0.7,0.9,1.0,1.1,1.2,1.4"


def _parse_temperatures(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--temperatures must contain at least one value")
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--temperatures must be comma-separated floats: {e}") from e


@app.command()
def main(
    n_prompts: Annotated[int, typer.Option(help="Number of distinct prompts to sample.")] = 300,
    siblings: Annotated[int, typer.Option(help="Sibling responses per prompt.")] = 6,
    min_gap: Annotated[float, typer.Option(help="Minimum chosen-minus-rejected score gap.")] = 2.0,
    max_spend_usd: Annotated[float, typer.Option(help="Abort if cumulative cost exceeds this.")] = 4.0,
    max_calls: Annotated[int, typer.Option(help="Per-component hard cap (prompter/generator/evaluator).")] = 5000,
    seed: Annotated[int, typer.Option(help="Seed for domain shuffle.")] = 42,
    out: Annotated[Path | None, typer.Option(help="PreferencePair JSONL path.")] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume", help="Skip prompts already represented in the output.")] = True,
    temperatures: Annotated[str, typer.Option(help="Comma-separated sibling temperature schedule (cycles if siblings>len).")] = DEFAULT_TEMPERATURES,
    max_tokens: Annotated[int, typer.Option(help="Generator max_tokens per sibling.")] = 600,
    adapter: Annotated[Path | None, typer.Option(help="Route generation through HFGenerator with this PEFT adapter.")] = None,
    steering_path: Annotated[Path | None, typer.Option(help="Steering-vector directory; routes through SteeredGenerator.")] = None,
    steering_strength: Annotated[float | None, typer.Option(help="Override the steering strength multiplier.")] = None,
) -> None:
    settings = get_settings()
    out_path = out or settings.preferences_path
    temps = _parse_temperatures(temperatures)

    estimate = pricing.estimate_sibling_run_cost(n_prompts, siblings=siblings)
    typer.echo(
        f"Pre-flight estimate for n_prompts={n_prompts}, siblings={siblings}: ${estimate:.4f}"
    )
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

    seen_prompts: set[str] = set()
    if resume and out_path.exists():
        for existing in read_all(out_path):
            seen_prompts.add(existing.prompt)
        if seen_prompts:
            typer.echo(f"Resume: {len(seen_prompts)} prompt(s) already in {out_path}; skipping.")

    effective_strength = (
        steering_strength
        if steering_strength is not None
        else (settings.default_steering_strength if steering_path is not None else None)
    )
    generator = make_generator(
        adapter,
        steering_path=steering_path,
        steering_strength=effective_strength,
    )
    sibling_run_id = str(uuid.uuid4())

    cap = min(n_prompts, max_calls)

    prompts_attempted = 0
    pairs_emitted = 0
    gap_sum = 0.0
    total_cost = 0.0
    cost_breakdown = {"prompter": 0.0, "generator": 0.0, "evaluator": 0.0}
    call_counts = {"prompter": 0, "generator": 0, "evaluator": 0}

    aborted = False

    try:
        for i in tqdm(range(cap), desc="prompts"):
            domain = domains[i % len(domains)]
            salt = i // len(domains)

            prompt_text, p_usage = prompter.generate_prompt(domain, salt=salt)
            call_counts["prompter"] += 1
            assert prompt_text not in held_out_prompts, (
                f"Prompter produced a held-out prompt: {prompt_text!r}"
            )
            p_cost = (
                0.0
                if p_usage.cached
                else pricing.cost_of(
                    settings.prompter_model,
                    p_usage.prompt_tokens,
                    p_usage.completion_tokens,
                )
            )
            total_cost += p_cost
            cost_breakdown["prompter"] += p_cost

            if resume and prompt_text in seen_prompts:
                continue

            prompts_attempted += 1

            if call_counts["prompter"] > max_calls:
                typer.echo(
                    f"\nAborting: prompter calls exceeded --max-calls {max_calls}.",
                    err=True,
                )
                aborted = True
                break

            sibs = sample_and_score_siblings(
                prompt_text,
                n=siblings,
                temperatures=temps,
                generator=generator,
                max_tokens=max_tokens,
            )

            # Per-component cost accounting based on cached flags from each sample.
            for s in sibs:
                call_counts["generator"] += 1
                call_counts["evaluator"] += 1
                local_generator = adapter is not None or steering_path is not None
                if not s.generator_cached and not local_generator:
                    g_cost = pricing.cost_of(
                        settings.together_base_model,
                        len(prompt_text) // 4 + 50,
                        len(s.response) // 4 + 1,
                    )
                    total_cost += g_cost
                    cost_breakdown["generator"] += g_cost
                if not s.evaluator_cached:
                    # Use token budget heuristic; we don't get back a Usage object
                    # for cached evaluator results so the conservative path uses
                    # the budgeted token counts from pricing._TOKEN_BUDGET.
                    e_in, e_out = pricing._TOKEN_BUDGET["evaluator"]
                    e_cost = pricing.cost_of(settings.evaluator_model, e_in, e_out)
                    total_cost += e_cost
                    cost_breakdown["evaluator"] += e_cost

            if any(count > max_calls for count in call_counts.values()):
                typer.echo(
                    f"\nAborting: per-component call cap exceeded "
                    f"(--max-calls {max_calls}); counts={call_counts}.",
                    err=True,
                )
                aborted = True
                break

            new_pairs = mine_pairs(
                prompt_text,
                domain=domain,
                siblings=sibs,
                sibling_run_id=sibling_run_id,
                generator_model=settings.together_base_model,
                evaluator_model=settings.evaluator_model,
                rubric_version=evaluator.RUBRIC_VERSION,
                min_gap=min_gap,
            )
            for pair in new_pairs:
                append(pair, out_path)
                pairs_emitted += 1
                gap_sum += pair.score_gap
                seen_prompts.add(prompt_text)

            if total_cost > max_spend_usd:
                typer.echo(
                    f"\nAborting after {prompts_attempted} prompts: cumulative cost "
                    f"${total_cost:.4f} exceeded --max-spend-usd ${max_spend_usd:.4f}.",
                    err=True,
                )
                aborted = True
                break
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user.", err=True)
        aborted = True

    mean_gap = (gap_sum / pairs_emitted) if pairs_emitted else 0.0
    yield_rate = (pairs_emitted / prompts_attempted) if prompts_attempted else 0.0
    typer.echo("")
    typer.echo(f"Prompts attempted: {prompts_attempted} / {cap}")
    typer.echo(f"Pairs emitted: {pairs_emitted}  (yield {yield_rate:.0%})")
    typer.echo(f"Mean score gap: {mean_gap:.2f}")
    typer.echo(f"Total cost: ${total_cost:.4f}")
    typer.echo(
        "Cost breakdown: "
        f"prompter ${cost_breakdown['prompter']:.4f}, "
        f"generator ${cost_breakdown['generator']:.4f}, "
        f"evaluator ${cost_breakdown['evaluator']:.4f}"
    )
    typer.echo(f"Sibling run id: {sibling_run_id}")
    typer.echo(f"Preferences appended to: {out_path}")

    if aborted and pairs_emitted == 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    sys.exit(app() or 0)
