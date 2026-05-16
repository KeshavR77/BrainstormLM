from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch

from brainstormlm.conceptial.engine.device_utils import PreferredDevice

# Required keys on each ``layer_injections`` dict (V3 multi-checkpoint path).
LAYER_INJECTION_KEYS: frozenset[str] = frozenset({"layer", "vector_path", "start_alpha", "decay_tokens"})

# Optional keys (see docs on :class:`SteeringConfig`).
LAYER_INJECTION_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "injection_site",
        "steering_math",
        "alpha_schedule",
        "exp_zero_by_step",
        "sigmoid_mid_step",
        "sigmoid_steepness",
    }
)

INJECTION_SITES: tuple[str, ...] = ("residual", "attn_o_proj", "mlp_down_proj")
ALPHA_SCHEDULES: tuple[str, ...] = ("linear", "constant", "exp_decay", "sigmoid_ramp")
STEERING_MATHS: tuple[str, ...] = ("additive", "spherical")


def validate_layer_injections(injections: list[dict[str, Any]]) -> None:
    """Raise ``KeyError`` / ``TypeError`` / ``ValueError`` if any injection dict is malformed."""
    seen: set[int] = set()
    for i, raw in enumerate(injections):
        if not isinstance(raw, dict):
            raise TypeError(f"layer_injections[{i}] must be dict, got {type(raw).__name__}")
        missing = LAYER_INJECTION_KEYS - set(raw.keys())
        if missing:
            raise KeyError(f"layer_injections[{i}] missing keys: {sorted(missing)}")
        extra = set(raw.keys()) - LAYER_INJECTION_KEYS - LAYER_INJECTION_OPTIONAL_KEYS
        if extra:
            raise KeyError(f"layer_injections[{i}] unknown keys: {sorted(extra)}")
        layer = int(raw["layer"])
        if layer in seen:
            raise ValueError(f"layer_injections duplicates layer {layer}")
        seen.add(layer)
        path = raw["vector_path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"layer_injections[{i}] vector_path must be non-empty str")
        float(raw["start_alpha"])
        int(raw["decay_tokens"])
        site = raw.get("injection_site", "residual")
        if site not in INJECTION_SITES:
            raise ValueError(
                f"layer_injections[{i}] injection_site must be one of {INJECTION_SITES}, got {site!r}"
            )
        steering_math = raw.get("steering_math", "additive")
        if steering_math not in STEERING_MATHS:
            raise ValueError(
                f"layer_injections[{i}] steering_math must be one of {STEERING_MATHS}, got {steering_math!r}"
            )
        sched = raw.get("alpha_schedule")
        if sched is not None and sched not in ALPHA_SCHEDULES:
            raise ValueError(
                f"layer_injections[{i}] alpha_schedule must be one of {ALPHA_SCHEDULES}, got {sched!r}"
            )


class Pooling(str, Enum):
    """How to aggregate prompt-token hidden states for contrastive vectors."""

    LAST = "last"
    MEAN = "mean"


class Normalization(str, Enum):
    """Optional post-processing of the steering vector before scaling by alpha."""

    NONE = "none"
    L2 = "l2"


class SteeringStrategy(str, Enum):
    """High-level benchmark / injection mode."""

    CONTROL = "control"
    PROMPT_ONLY = "prompt_only"
    SINGLE = "single"
    MULTI = "multi"
    FADING = "fading"


@dataclass
class SteeringConfig:
    """Model load settings and per-layer steering strengths."""

    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    strategy: SteeringStrategy = SteeringStrategy.SINGLE
    #: V3: one dict per injected layer. Each entry:
    #: ``{"layer": int, "vector_path": str, "start_alpha": float, "decay_tokens": int}``.
    #: If ``decay_tokens > 0``, linear fading is applied for that layer; if ``0``, static ``start_alpha``.
    #: When non-empty, hooks and checkpoint loading use this list; ``layer_alphas`` / ``decay_tokens``
    #: are ignored for injection (load vectors via :func:`shrooms_llm.pipeline.apply_layer_injections`).
    layer_injections: list[dict[str, Any]] = field(default_factory=list)
    #: **Deprecated** (V2): use ``layer_injections`` when loading vectors from disk per layer.
    #: When ``layer_injections`` is empty, hooks still read this map (benchmarks / CLI).
    layer_alphas: dict[int, float] = field(default_factory=lambda: {16: 1.0})
    #: **Deprecated** for per-layer use when ``layer_injections`` is set. If ``layer_injections`` is
    #: empty and ``strategy`` is ``FADING`` with ``decay_tokens > 0``, all ``layer_alphas`` entries
    #: share this decay horizon (V2 behavior).
    decay_tokens: int = 128
    pooling: Pooling = Pooling.LAST
    normalization: Normalization = Normalization.L2
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    preferred_device: PreferredDevice = "auto"
    device_map: str | dict[str, Any] | None = None
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] | None = None
    l2_eps: float = 1e-8
    inject_last_token_only: bool = False
    #: Optional in-memory steering tensors keyed by decoder layer index. When set for a layer listed in
    #: ``layer_injections``, :func:`~shrooms_llm.pipeline.load_steering_vectors_from_layer_injections`
    #: uses the tensor instead of loading ``vector_path`` from disk (benchmark iterative anti-mode).
    steering_vector_overrides: dict[int, Any] = field(default_factory=dict)
    #: Decoder layers where ghost (unsteered) hiddens are captured and projected out of the main path.
    ghost_layers: list[int] = field(default_factory=lambda: [28])
    ghost_projection_eps: float = 1e-8
    #: Scale ghost subtraction: ``h - ghost_strength * proj(h,g)`` (``1.0`` = full removal).
    ghost_strength: float = 0.6
    ghost_injection_site: str = "residual"
    #: When True, ghost strength follows top-k logit entropy each decode step (see entropy fields below).
    use_entropy_gated_ghost: bool = False
    ghost_entropy_top_k: int = 50
    ghost_entropy_h_low: float = 1.5
    ghost_entropy_h_high: float = 4.0
    #: Ghost strength when entropy is low / high (maps to cliché-like vs uncertain).
    ghost_entropy_gamma_high: float = 0.9
    ghost_entropy_gamma_low: float = 0.1
    #: Minimum softmax prob of draft token on main verify pass (speculative PoC).
    speculative_min_accept_prob: float = 0.05
    #: Min-P (V8.1): after temperature softmax, drop tokens with p < rho * p_max before multinomial (ghost path only).
    min_p: float = 0.05
    #: Sliding-window max PPL (exp of mean NLL in window) triggers manifold circuit breaker (ghost completions).
    manifold_ppl_max: float = 50.0
    #: Token NLL window length for sliding PPL (must be >= 1).
    ppl_sliding_window_tokens: int = 32
    #: Retries per solution-set step after IOR flush + base-temperature reset when sliding PPL exceeds threshold.
    manifold_max_retries: int = 3
    #: Semantic latent mixing: tag -> weight (e.g. ``{"Minimalist": 0.8, "Corporate": -0.2}``).
    #: Resolved against :attr:`latent_registry_path` (or runtime registry on :class:`SteeringEngine`).
    latent_weights: dict[str, float] = field(default_factory=dict)
    latent_registry_path: str | None = None

    def resolve_dtype(self) -> torch.dtype | str:
        if self.torch_dtype == "auto":
            return "auto"
        return getattr(torch, self.torch_dtype)
