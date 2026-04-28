"""`brainstorm` CLI entrypoint."""

from pathlib import Path
from typing import Annotated

import typer

from brainstormlm.config import get_settings
from brainstormlm.memory import LessonStore, augmented_system, retrieve
from brainstormlm.model import make_generator

app = typer.Typer(add_completion=False, help="BrainstormLM CLI.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """BrainstormLM CLI."""


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="The brainstorming prompt.")],
    temperature: Annotated[float | None, typer.Option(help="Sampling temperature.")] = None,
    max_tokens: Annotated[int | None, typer.Option(help="Max tokens to generate.")] = None,
    system: Annotated[str | None, typer.Option(help="Optional system prompt.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the response cache.")] = False,
    use_memory: Annotated[bool, typer.Option("--use-memory", help="Inject ReasoningBank lessons into the system prompt.")] = False,
    k_success: Annotated[int, typer.Option(help="Number of success lessons to retrieve.")] = 3,
    k_failure: Annotated[int, typer.Option(help="Number of failure lessons to retrieve.")] = 2,
    show_lessons: Annotated[bool, typer.Option("--show-lessons", help="Print retrieved lessons to stderr.")] = False,
    adapter: Annotated[Path | None, typer.Option(help="Path to a PEFT LoRA adapter directory; routes generation through HFGenerator.")] = None,
    no_adapter: Annotated[bool, typer.Option("--no-adapter", help="Force TogetherGenerator even if a default adapter is configured.")] = False,
    steered: Annotated[bool, typer.Option("--steered", help="Apply the steering vector at settings.default_steering_path (or --steering-path).")] = False,
    no_steered: Annotated[bool, typer.Option("--no-steered", help="Disable steering even if BRAINSTORMLM_STEERING_PATH is set.")] = False,
    steering_path: Annotated[Path | None, typer.Option(help="Path to a steering-vector directory (implies --steered).")] = None,
    steering_strength: Annotated[float | None, typer.Option(help="Override the steering strength multiplier.")] = None,
) -> None:
    """Send PROMPT to the base model and print the response."""
    settings = get_settings()
    effective_system = system

    if use_memory:
        store = LessonStore(settings.lessons_path)
        store.load()
        lessons = retrieve(store, prompt, k_success=k_success, k_failure=k_failure)
        effective_system = augmented_system(system, lessons)
        if show_lessons:
            if not lessons:
                typer.echo("(no lessons retrieved)", err=True)
            else:
                for lesson in lessons:
                    typer.echo(f"[{lesson.kind}] {lesson.lesson}", err=True)

    if no_adapter:
        adapter_path: Path | None = None
    elif adapter is not None:
        adapter_path = adapter
    else:
        adapter_path = settings.default_adapter_path

    if no_steered:
        effective_steering: Path | None = None
    elif steering_path is not None:
        effective_steering = steering_path
    elif steered:
        if settings.default_steering_path is None:
            typer.echo(
                "--steered set but no --steering-path and no BRAINSTORMLM_STEERING_PATH configured.",
                err=True,
            )
            raise typer.Exit(code=2)
        effective_steering = settings.default_steering_path
    else:
        effective_steering = None

    generator = make_generator(
        adapter_path,
        steering_path=effective_steering,
        steering_strength=steering_strength,
    )
    response = generator.generate(
        prompt,
        system=effective_system,
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.default_max_tokens,
        use_cache=not no_cache,
    )
    typer.echo(response)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind host (default: settings.api_host).")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port (default: settings.api_port).")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="uvicorn auto-reload (dev only).")] = False,
) -> None:
    """Run the BrainstormLM FastAPI server (plan-06)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "brainstormlm.api.server:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
