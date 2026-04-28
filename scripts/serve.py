"""Convenience uvicorn launcher for the FastAPI server (plan-06).

Equivalent to ``brainstorm serve``. Useful for ``python scripts/serve.py``
in environments where the entrypoint script isn't on PATH.
"""

from __future__ import annotations

import sys

import uvicorn

from brainstormlm.config import get_settings


def main() -> int:
    settings = get_settings()
    uvicorn.run(
        "brainstormlm.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
