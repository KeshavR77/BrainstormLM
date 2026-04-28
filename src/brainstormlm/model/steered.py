"""SteeredGenerator — `HFGenerator` plus residual-stream activation steering.

Implements the `Generator` protocol. Cache hits short-circuit before any model
load or hook registration so warm runs stay fast. On cache miss, forward hooks
are installed on the requested decoder blocks for the duration of the
generation call and removed in `finally` so non-steered calls on the same
instance are unaffected.

Heavy deps (`torch`, `transformers`, `peft`) are imported lazily; this module
is import-cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.model.hf import HFGenerator
from brainstormlm.model.steering import SteeringVector


class SteeredGenerator(HFGenerator):
    """HF generator that adds a steering vector to the residual stream."""

    def __init__(
        self,
        *,
        steering_path: Path,
        steering_strength: float = 1.0,
        steering_layers: Sequence[int] | None = None,
        rescale: bool = True,
        base_model_id: str | None = None,
        adapter_path: Path | None = None,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> None:
        super().__init__(
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            device=device,
            dtype=dtype,
        )
        self.steering_path = Path(steering_path)
        self.steering_strength = float(steering_strength)
        self._steering_layers_override = (
            tuple(int(layer) for layer in steering_layers)
            if steering_layers is not None
            else None
        )
        self.rescale = rescale
        self._vector: SteeringVector | None = None

    # ------------------------------------------------------------------
    # Generator protocol — re-implemented to extend the cache key.

    def generate_with_meta(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> tuple[str, bool]:
        # The cache key includes steering provenance so steered and unsteered
        # calls can never collide.
        layers_key = (
            ",".join(str(layer) for layer in sorted(self._steering_layers_override))
            if self._steering_layers_override is not None
            else "manifest"
        )
        key = cache_key(
            "hf-steered",
            self.base_model_id,
            str(self.adapter_path) if self.adapter_path is not None else "no-adapter",
            str(self.steering_path),
            f"{self.steering_strength:.4f}",
            layers_key,
            "rescale" if self.rescale else "no-rescale",
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
        self._ensure_vector_loaded()
        handles = self._install_hooks()
        try:
            text = self._run_generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
        write_cache(key, text)
        return text, False

    # ------------------------------------------------------------------
    # Internals

    def _ensure_vector_loaded(self) -> None:
        if self._vector is not None:
            return
        self._vector = SteeringVector.load(
            self.steering_path, expected_base_model_id=self.base_model_id
        )

    def _resolve_layers(self) -> tuple[int, ...]:
        assert self._vector is not None, "vector must be loaded before resolving layers"
        if self._steering_layers_override is None:
            return self._vector.layers
        # Validate every requested layer is present in the artifact.
        missing = [
            layer for layer in self._steering_layers_override if layer not in self._vector.vectors
        ]
        if missing:
            raise ValueError(
                f"Steering layers {missing} not present in {self.steering_path}; "
                f"available layers: {self._vector.layers}"
            )
        return self._steering_layers_override

    def _residual_layers(self) -> list[Any]:
        """Return the list of decoder blocks to hook (LLaMA/Qwen vs GPT-2)."""
        model = self._model
        # LLaMA / Qwen / Mistral all expose `model.model.layers`.
        inner = getattr(model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return list(layers)
        # GPT-2 / Falcon style.
        transformer = getattr(model, "transformer", None)
        if transformer is not None:
            layers = getattr(transformer, "h", None)
            if layers is not None:
                return list(layers)
        raise RuntimeError(
            f"Cannot locate decoder blocks on model {type(model).__name__!r}; "
            "expected `.model.layers` or `.transformer.h`."
        )

    def _install_hooks(self) -> list[Any]:
        """Register forward hooks on the requested layers; return handles."""
        layers = self._resolve_layers()
        blocks = self._residual_layers()
        assert self._vector is not None
        device = getattr(self._model, "device", None) or self._device
        strength = self.steering_strength
        rescale = self.rescale

        handles: list[Any] = []
        for layer_idx in layers:
            block = blocks[layer_idx]
            v = self._vector.vectors[layer_idx]
            v_on_device = self._move_vector(v, device)
            hook = _make_residual_hook(v_on_device, strength=strength, rescale=rescale)
            handles.append(block.register_forward_hook(hook))
        return handles

    @staticmethod
    def _move_vector(v: Any, device: Any) -> Any:
        try:
            return v.to(device) if device is not None else v
        except Exception:
            return v


def _make_residual_hook(vector: Any, *, strength: float, rescale: bool):
    """Return a forward hook that adds `strength * vector` to the residual.

    Decoder blocks in LLaMA/Qwen return a tuple whose first element is the
    residual-stream output; GPT-2 blocks return the same. Hooking the *output*
    keeps the math identical to "intervene before the next block reads."

    `rescale=True` matches the additive intervention to the layer's natural
    activation norm at each token, so the same `strength` flag transfers
    across model sizes.
    """

    def hook(_module: Any, _inputs: Any, output: Any):  # noqa: ANN001 — torch hook signature
        # transformers blocks return either a Tensor or a tuple-like
        # `(hidden_states, ...)`. Handle both.
        if isinstance(output, tuple):
            hidden = output[0]
            tail = output[1:]
        else:
            hidden = output
            tail = ()

        try:
            v_cast = vector.to(dtype=hidden.dtype, device=hidden.device)
        except Exception:
            v_cast = vector

        if rescale:
            v_norm = v_cast.norm()
            if float(v_norm) <= 0.0:
                return output
            # Per-token activation norm (over hidden dim), shape (batch, seq, 1).
            h_norm = hidden.norm(dim=-1, keepdim=True)
            scaled = (v_cast / v_norm) * h_norm * strength
        else:
            scaled = v_cast * strength

        new_hidden = hidden + scaled
        if tail:
            return (new_hidden, *tail)
        return new_hidden

    return hook
