"""Generator protocol — the swap point for base vs steered models."""

from typing import Protocol


class Generator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str: ...
