from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import PreTrainedModel

from brainstormlm.conceptial.engine.config import SteeringConfig, SteeringStrategy, validate_layer_injections
from brainstormlm.conceptial.engine.fading_strategy import current_alpha_for_injection
from brainstormlm.conceptial.engine.model_utils import decoder_layers

_LOG = logging.getLogger(__name__)
_SHAPE_WARNED: set[str] = set()


def _apply_steering(
    h: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    last_token_only: bool,
    steering_math: str = "additive",
    eps: float = 1e-8,
) -> torch.Tensor:
    vv = v.to(device=h.device, dtype=h.dtype)
    math_mode = (steering_math or "additive").strip().lower()
    if math_mode not in ("additive", "spherical"):
        raise ValueError(f"unknown steering_math: {steering_math!r}")
    delta = alpha * vv

    def _spherical_project(base: torch.Tensor, steered: torch.Tensor) -> torch.Tensor:
        # Compute norms in fp32 for numerical stability on fp16/bf16 activations.
        base_fp32 = base.float()
        steered_fp32 = steered.float()
        base_norm = torch.linalg.vector_norm(base_fp32, ord=2, dim=-1, keepdim=True)
        steered_norm = torch.linalg.vector_norm(steered_fp32, ord=2, dim=-1, keepdim=True).clamp_min(float(eps))
        out_fp32 = steered_fp32 * (base_norm / steered_norm)
        return out_fp32.to(dtype=base.dtype)

    if last_token_only:
        out = h.clone()
        base = out[:, -1, :]
        steered = base + delta
        out[:, -1, :] = steered if math_mode == "additive" else _spherical_project(base, steered)
        return out
    steered = h + delta
    if math_mode == "additive":
        return steered
    return _spherical_project(h, steered)


def _hidden_tensor(out: object) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out  # type: ignore[return-value]


def _legacy_shared_decay_tokens(config: SteeringConfig) -> int:
    """V2: one decay horizon for every hooked layer when strategy is FADING."""
    if config.strategy is SteeringStrategy.FADING and config.decay_tokens > 0:
        return int(config.decay_tokens)
    return 0


def resolve_steering_module(model: PreTrainedModel, layer_index: int, injection_site: str) -> torch.nn.Module:
    """
    Return the submodule to hook for steering at ``layer_index``.

    ``injection_site``: ``residual`` (decoder block output), ``attn_o_proj``, ``mlp_down_proj``.
    """
    layers = decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer index must be in [0, {len(layers) - 1}], got {layer_index}")
    block = layers[layer_index]
    site = (injection_site or "residual").strip()
    if site == "residual":
        return block
    attn = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    if site == "attn_o_proj":
        if attn is None or not hasattr(attn, "o_proj"):
            raise ValueError(
                f"Layer {layer_index} has no self_attn.o_proj; split-brain injection_site not supported."
            )
        return attn.o_proj  # type: ignore[no-any-return]
    if site == "mlp_down_proj":
        if mlp is None or not hasattr(mlp, "down_proj"):
            raise ValueError(
                f"Layer {layer_index} has no mlp.down_proj; split-brain injection_site not supported."
            )
        return mlp.down_proj  # type: ignore[no-any-return]
    raise ValueError(f"unknown injection_site: {injection_site!r}")


def _injection_dicts_for_config(config: SteeringConfig) -> list[dict[str, Any]]:
    """Sorted list of injection dicts (including legacy ``layer_alphas`` synthesis).

    Returns an empty list when there are no ``layer_injections`` and no ``layer_alphas``
    (unsteered / no-hook generation via :func:`steering_context`).
    """
    if config.layer_injections:
        validate_layer_injections(config.layer_injections)
        rows = [dict(x) for x in config.layer_injections]
        for r in rows:
            r.setdefault("injection_site", "residual")
            r.setdefault("steering_math", "additive")
        return sorted(rows, key=lambda x: int(x["layer"]))
    if not config.layer_alphas:
        return []
    dec = _legacy_shared_decay_tokens(config)
    rows = []
    for L, a in sorted(config.layer_alphas.items()):
        sched = "linear" if dec > 0 else "constant"
        rows.append(
            {
                "layer": int(L),
                "vector_path": "",
                "start_alpha": float(a),
                "decay_tokens": int(dec),
                "injection_site": "residual",
                "steering_math": "additive",
                "alpha_schedule": sched,
            }
        )
    return rows


def _steering_hook_factory(
    v: torch.Tensor,
    inj: dict[str, Any],
    config: SteeringConfig,
    *,
    token_step: list[int],
    owner_layer_index: int,
    layer_index: int,
    last_token_only: bool,
    shape_tag: str,
    capture_target_layer: int | None = None,
    capture_sink: list[torch.Tensor] | None = None,
):
    def hook(_module, _inp, out):
        h = _hidden_tensor(out)
        s = token_step[0]
        if (
            capture_sink is not None
            and capture_target_layer is not None
            and layer_index == capture_target_layer
            and h.shape[1] == 1
            and s == 0
        ):
            capture_sink.append(h[:, -1, :].detach().float().cpu().reshape(-1).clone())
        alpha = current_alpha_for_injection(s, inj)

        if layer_index == owner_layer_index and h.shape[1] == 1:
            token_step[0] = s + 1

        if v.shape[-1] != h.shape[-1]:
            key = shape_tag
            if key not in _SHAPE_WARNED:
                _LOG.error(
                    "Steering vector dim %s != hidden dim %s (%s); skipping steering for this forward.",
                    tuple(v.shape),
                    tuple(h.shape),
                    shape_tag,
                )
                _SHAPE_WARNED.add(key)
            raise RuntimeError(
                f"steering vector hidden dim {v.shape[-1]} != activation hidden dim {h.shape[-1]} ({shape_tag})"
            )

        new_h = _apply_steering(
            h,
            v,
            alpha,
            last_token_only,
            steering_math=str(inj.get("steering_math", "additive")),
            eps=float(getattr(config, "l2_eps", 1e-8)),
        )
        if isinstance(out, tuple):
            return (new_h,) + tuple(out[1:])
        return new_h

    return hook


@contextmanager
def steering_context(
    model: PreTrainedModel,
    vectors: dict[int, torch.Tensor],
    config: SteeringConfig,
    *,
    initial_token_step: int = 0,
    capture_first_decode_residual: tuple[int, list[torch.Tensor]] | None = None,
) -> Iterator[None]:
    """
    Register forward hooks for each injection in the active steering spec.

    ``vectors`` must contain exactly one tensor per layer in the spec.

    Fading / schedules use :func:`shrooms_llm.fading_strategy.current_alpha_for_injection`.
    The step counter advances on single-token decode forwards (seq_len==1) on the
    highest hooked **decoder layer** index only (once per model forward).

    ``initial_token_step`` seeds the internal counter so schedules (e.g. exp decay) continue
    from a non-zero index (Tree-of-Thoughts / resumed generation).

    When the active config resolves to zero injections (unsteered), this context manager
    is a no-op: no hooks are registered.
    """
    injections = _injection_dicts_for_config(config)
    if not injections:
        yield
        return

    layers = decoder_layers(model)
    owner_layer_index = max(int(x["layer"]) for x in injections)
    token_step: list[int] = [max(0, int(initial_token_step))]
    handles: list[torch.utils.hooks.RemovableHandle] = []
    cap_layer: int | None = None
    cap_sink: list[torch.Tensor] | None = None
    if capture_first_decode_residual is not None:
        cap_layer, cap_sink = capture_first_decode_residual
    try:
        for inj in injections:
            L = int(inj["layer"])
            if L not in vectors:
                raise KeyError(f"vectors missing key for layer {L}")
            if not 0 <= L < len(layers):
                raise ValueError(f"layer index must be in [0, {len(layers) - 1}], got {L}")
            site = str(inj.get("injection_site", "residual"))
            mod = resolve_steering_module(model, L, site)
            tag = f"L{L}:{site}"
            hook_fn = _steering_hook_factory(
                vectors[L],
                inj,
                config,
                token_step=token_step,
                owner_layer_index=owner_layer_index,
                layer_index=L,
                last_token_only=config.inject_last_token_only,
                shape_tag=tag,
                capture_target_layer=cap_layer,
                capture_sink=cap_sink,
            )
            handles.append(mod.register_forward_hook(hook_fn))
        yield
    finally:
        for h in handles:
            h.remove()


def register_steering_hook(
    model: PreTrainedModel,
    vector: torch.Tensor,
    layer_index: int,
    alpha: float,
    *,
    inject_last_token_only: bool = False,
):
    """Single-layer static steering; returns one RemovableHandle. Prefer :func:`steering_context`."""
    cfg = SteeringConfig(
        layer_alphas={layer_index: alpha},
        strategy=SteeringStrategy.SINGLE,
        inject_last_token_only=inject_last_token_only,
    )
    layers = decoder_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"layer_index must be in [0, {len(layers) - 1}], got {layer_index}")
    inj = {
        "layer": layer_index,
        "vector_path": "",
        "start_alpha": float(alpha),
        "decay_tokens": 0,
        "injection_site": "residual",
        "steering_math": "additive",
        "alpha_schedule": "constant",
    }
    token_step = [0]
    hook_fn = _steering_hook_factory(
        vector,
        inj,
        cfg,
        token_step=token_step,
        owner_layer_index=layer_index,
        layer_index=layer_index,
        last_token_only=inject_last_token_only,
        shape_tag=f"L{layer_index}:residual",
    )
    mod = resolve_steering_module(model, layer_index, "residual")
    return mod.register_forward_hook(hook_fn)
