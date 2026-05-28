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
        self._vectors_loaded = False

    @classmethod
    def from_config_path(cls, path: Path, *, do_steering: bool = True) -> "EngineGenerator":
        """Build from a Conceptial JSON config (e.g. ``config/conceptial_14b.json``).

        The JSON has metadata keys (``id``, ``display_name``, ``steering_vectors``,
        ``schedule``, etc.) that don't map to ``SteeringConfig`` fields.  This method
        extracts what ``SteeringConfig`` needs and builds ``layer_injections`` from the
        ``steering_vectors`` / ``layers`` / ``schedule`` sections.
        """
        import json

        raw = json.loads(path.read_text())
        model_id = raw.get("base_model") or raw.get("model_id", SteeringConfig.model_id)

        # Build layer_injections from steering_vectors + layers + schedule
        layer_injections: list[dict] = []
        sv = raw.get("steering_vectors", {})
        layers = raw.get("layers", {})
        schedule = raw.get("schedule", {})

        if sv.get("creative") and layers.get("creative_layer") is not None:
            exp_zero = schedule.get("exp_zero_by_step", 0)
            creative_inj: dict = {
                "layer": layers["creative_layer"],
                "vector_path": sv["creative"],
                "start_alpha": schedule.get("creative_start_alpha", 1.0),
                "decay_tokens": exp_zero,
                "alpha_schedule": "exp_decay" if exp_zero > 0 else "constant",
                "steering_math": schedule.get("steering_math", "additive"),
            }
            # exp_decay reads exp_zero_by_step from the injection dict at runtime
            if exp_zero > 0:
                creative_inj["exp_zero_by_step"] = exp_zero
            layer_injections.append(creative_inj)

        if sv.get("technical") and layers.get("technical_layer") is not None:
            has_sigmoid = (
                schedule.get("sigmoid_mid_step") is not None
                and schedule.get("sigmoid_steepness") is not None
            )
            technical_inj: dict = {
                "layer": layers["technical_layer"],
                "vector_path": sv["technical"],
                "start_alpha": schedule.get("technical_start_alpha", 1.0),
                "decay_tokens": 0,
                "steering_math": schedule.get("steering_math", "additive"),
            }
            if has_sigmoid:
                technical_inj["alpha_schedule"] = "sigmoid_ramp"
                technical_inj["sigmoid_mid_step"] = schedule["sigmoid_mid_step"]
                technical_inj["sigmoid_steepness"] = schedule["sigmoid_steepness"]
            layer_injections.append(technical_inj)

        # Extract only SteeringConfig-compatible fields from raw, if present
        sc_fields = {f.name for f in SteeringConfig.__dataclass_fields__.values()}
        sc_kwargs = {k: v for k, v in raw.items() if k in sc_fields}
        sc_kwargs["model_id"] = model_id
        if layer_injections:
            sc_kwargs["layer_injections"] = layer_injections

        config = SteeringConfig(**sc_kwargs)
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

    def _ensure_vectors(self) -> None:
        """Load steering vectors from disk on first use (lazy init)."""
        if self._vectors_loaded or not self._do_steering:
            return
        if self._engine.config.layer_injections:
            from brainstormlm.conceptial.engine.pipeline import apply_layer_injections

            apply_layer_injections(self._engine)
        self._vectors_loaded = True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        self._ensure_vectors()
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
