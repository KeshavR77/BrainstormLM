"""Adapter: wraps SteeringEngine to conform to the Generator protocol."""

from __future__ import annotations

from pathlib import Path

from brainstormlm.conceptial.engine.config import SteeringConfig
from brainstormlm.conceptial.engine.pipeline import SteeringEngine


class EngineGenerator:
    """Generator-protocol adapter around SteeringEngine.

    Lazily loads the model on first call to .generate() or .generate_with_meta().
    """

    def __init__(self, config: SteeringConfig, *, do_steering: bool = True):
        self._engine = SteeringEngine(config)
        self._do_steering = do_steering

    @classmethod
    def from_config_path(cls, path: Path, *, do_steering: bool = True) -> "EngineGenerator":
        import json

        raw = json.loads(path.read_text())
        config = SteeringConfig(**raw)
        return cls(config, do_steering=do_steering)

    @property
    def model_id(self) -> str:
        return self._engine.config.model_id

    def _format_prompt(self, prompt: str, system: str | None) -> str:
        """Format as Qwen2.5 ChatML if system is provided."""
        if system is None:
            return (
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        formatted = self._format_prompt(prompt, system)
        return self._engine.complete(
            formatted,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            do_steering=self._do_steering,
        )

    def generate_with_meta(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> tuple[str, bool]:
        """Returns (text, cached). Local engine never caches."""
        text = self.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        return text, False
