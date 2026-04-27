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

    generator = make_generator(adapter_path)
    response = generator.generate(
        prompt,
        system=effective_system,
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.default_max_tokens,
        use_cache=not no_cache,
    )
    typer.echo(response)


if __name__ == "__main__":
    app()
