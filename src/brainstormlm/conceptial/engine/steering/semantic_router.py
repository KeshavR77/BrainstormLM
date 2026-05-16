"""
V4.1 semantic routing: inject deterministic mock style directions into decoder hiddens
when ``accumulate_ior_memory`` anti-cliche runs, keyed by IOR ``explored_len`` (0..4 → style slot).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import PreTrainedModel

from brainstormlm.conceptial.engine.hooks import _hidden_tensor, resolve_steering_module

STYLE_ORDER: tuple[str, ...] = ("Brutalist", "Whimsical", "Corporate", "Surreal", "Cyberpunk")
MOCK_VECTOR_DIM: int = 2048
DEFAULT_ROUTE_LAYER: int = 20
DEFAULT_ROUTE_ALPHA: float = 0.15

_vectors_by_dim: dict[int, dict[str, torch.Tensor]] = {}


def _style_vectors_for_dim(hidden_size: int) -> dict[str, torch.Tensor]:
    """Deterministic unit-ish mock vectors; one dict entry per ``STYLE_ORDER`` name."""
    if hidden_size in _vectors_by_dim:
        return _vectors_by_dim[hidden_size]
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(STYLE_ORDER):
        g = torch.Generator().manual_seed(424200 + i * 97)
        v = torch.randn(int(hidden_size), generator=g, dtype=torch.float32)
        v = v / (v.norm() + 1e-8)
        out[name] = v.cpu()
    _vectors_by_dim[hidden_size] = out
    return out


def style_name_for_explored_len(explored_len: int) -> str:
    """Map IOR history length at request start to a style label (0→Brutalist, …)."""
    return STYLE_ORDER[int(explored_len) % len(STYLE_ORDER)]


def _route_hook_factory(v_row: torch.Tensor, alpha: float):
    """Add ``alpha * v`` to residual stream (broadcast over batch and seq)."""

    def hook(_module: torch.nn.Module, _inp: Any, out: object) -> object:
        h = _hidden_tensor(out)
        if h.numel() == 0:
            return out
        vv = v_row.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
        h_new = h + float(alpha) * vv
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new

    return hook


@contextmanager
def semantic_route_inject_context(
    model: PreTrainedModel,
    *,
    explored_len: int,
    layer_index: int = DEFAULT_ROUTE_LAYER,
    alpha: float = DEFAULT_ROUTE_ALPHA,
    injection_site: str = "residual",
) -> Iterator[None]:
    """
    Register a forward hook on the decoder submodule at ``layer_index`` that adds a small
    style vector to hidden activations. ``explored_len`` selects the style slot (0..4 cyclically).

    Hooks are always removed on exit (success or exception).
    """
    hs = int(getattr(model.config, "hidden_size", MOCK_VECTOR_DIM))
    vecs = _style_vectors_for_dim(hs)
    name = style_name_for_explored_len(explored_len)
    v_cpu = vecs[name]
    mod = resolve_steering_module(model, int(layer_index), injection_site)
    hook_fn = _route_hook_factory(v_cpu, float(alpha))
    h = mod.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        h.remove()


def mock_style_vectors_2048() -> dict[str, torch.Tensor]:
    """Return the 2048-D mock style dictionary (useful for tests / docs)."""
    return _style_vectors_for_dim(MOCK_VECTOR_DIM)
