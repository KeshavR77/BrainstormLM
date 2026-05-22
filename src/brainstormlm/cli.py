"""`brainstorm` CLI entrypoint."""

from typing import Annotated

import typer

from brainstormlm.config import get_settings

app = typer.Typer(add_completion=False, help="BrainstormLM CLI.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """BrainstormLM CLI."""


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind host (default: settings.api_host).")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port (default: settings.api_port).")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="uvicorn auto-reload (dev only).")] = False,
) -> None:
    """Run the BrainstormLM FastAPI server."""
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
