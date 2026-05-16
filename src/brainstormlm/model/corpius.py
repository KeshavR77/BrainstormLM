"""CorpiusGenerator — Corpius 1 (Qwen2.5-3B + RepE steering) behind the Generator protocol.

Heavy deps (torch / steering engine) are imported lazily so cache hits stay fast
and ``from brainstormlm.model import CorpiusGenerator`` remains cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brainstormlm.clients.cache import cache_key, read_cache, write_cache

# Default config path relative to repo root — populated at install time.
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[4] / "data" / "corpius" / "corpius_1.json"
)


class CorpiusGenerator:
    """Wraps the ShroomsLLM SteeringEngine with the Corpius 1 frozen v17 preset.

    Implements the `model.interface.Generator` protocol.  Cache hits short-
    circuit before any model load — stays fast on warm cache even without GPU.
    """

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
        self._device_override = device
        self._engine: Any = None
        self._spec: Any = None

    # ------------------------------------------------------------------
    # Generator protocol

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        text, _ = self.generate_with_meta(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        return text

    def generate_with_meta(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> tuple[str, bool]:
        key = cache_key(
            "corpius",
            self._base_model_id(),
            system or "",
            prompt,
            f"{temperature:.4f}",
            str(max_tokens),
        )
        if use_cache:
            cached = read_cache(key)
            if cached is not None:
                return cached, True

        self._ensure_loaded()
        text = self._run_generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        write_cache(key, text)
        return text, False

    # ------------------------------------------------------------------
    # Internals

    def _base_model_id(self) -> str:
        if self._spec is not None:
            return self._spec.base_model
        # Read just enough from config to build the cache key without full load.
        try:
            import json
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
            return str(raw.get("base_model", "Qwen/Qwen2.5-3B-Instruct"))
        except Exception:
            return "Qwen/Qwen2.5-3B-Instruct"

    def _ensure_loaded(self) -> None:
        """Lazy-load the Corpius SteeringEngine. Idempotent."""
        if self._engine is not None and self._spec is not None:
            return

        from brainstormlm.conceptial.engine.corpius import apply_corpius_1, load_corpius_spec
        from brainstormlm.conceptial.engine.pipeline import SteeringEngine
        from brainstormlm.conceptial.engine.config import SteeringConfig

        spec = load_corpius_spec(self._config_path)
        cfg = SteeringConfig(base_model=spec.base_model)
        if self._device_override:
            cfg.device = self._device_override

        engine = SteeringEngine(cfg)
        apply_corpius_1(engine, spec)

        self._spec = spec
        self._engine = engine

    def _run_generate(
        self,
        *,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return self._engine.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
