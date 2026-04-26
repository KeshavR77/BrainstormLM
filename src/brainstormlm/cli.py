"""`brainstorm` CLI entrypoint."""

from typing import Annotated

import typer

from brainstormlm.config import get_settings
from brainstormlm.memory import LessonStore, augmented_system, retrieve
from brainstormlm.model.base import TogetherGenerator

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

    generator = TogetherGenerator()
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
