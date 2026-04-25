"""`brainstorm` CLI entrypoint."""

from typing import Annotated

import typer

from brainstormlm.config import get_settings
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
) -> None:
    """Send PROMPT to the base model and print the response."""
    settings = get_settings()
    generator = TogetherGenerator()
    response = generator.generate(
        prompt,
        system=system,
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.default_max_tokens,
        use_cache=not no_cache,
    )
    typer.echo(response)


if __name__ == "__main__":
    app()
