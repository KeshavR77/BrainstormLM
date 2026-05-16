"""
Mock Sparse Autoencoder (SAE) feature ablation hooks for Conceptial Corpius 1 (V3.5 demos).

**Hook order:** register SAE hooks *before* ghost rejection hooks on the same submodule so the
ghost path sees ablated activations (PyTorch runs forward hooks in registration order).
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import PreTrainedModel

from brainstormlm.conceptial.engine.hooks import _hidden_tensor, resolve_steering_module
from brainstormlm.conceptial.engine.model_utils import decoder_layers


def _stable_seed(seed_key: str, hidden_size: int) -> int:
    h = hashlib.sha256(f"{seed_key}:{hidden_size}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


def mock_cliche_mask(hidden_size: int, *, seed_key: str = "conceptial_mock_sae_v1") -> torch.Tensor:
    """
    Boolean mask of length ``hidden_size`` with exactly ``floor(0.05 * hidden_size)`` True entries.
    """
    hs = int(hidden_size)
    n = max(1, int(hs * 0.05))
    g = torch.Generator()
    g.manual_seed(_stable_seed(seed_key, hs))
    idx = torch.randperm(hs, generator=g)[:n]
    m = torch.zeros(hs, dtype=torch.bool)
    m[idx] = True
    return m


def sae_demo_layer_indices(model: PreTrainedModel) -> list[int]:
    """Layers ``20, 28, 35`` clamped to valid decoder indices (Qwen2.5-3B style)."""
    n = len(decoder_layers(model))
    raw = [20, 28, 35]
    out: list[int] = []
    for L in raw:
        li = min(max(0, int(L)), n - 1)
        if li not in out:
            out.append(li)
    return out


class SAEInterceptor:
    """Holds a mock SAE feature mask and builds a forward hook closure."""

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def hook_fn(self, _module: torch.nn.Module, _inp: Any, out: object) -> object:
        h = _hidden_tensor(out)
        if h.shape[-1] != self.mask.shape[0]:
            return out
        m = self.mask.to(device=h.device, dtype=torch.bool)
        h2 = h.clone()
        h2[..., m] = 0.0
        if isinstance(out, tuple):
            return (h2,) + tuple(out[1:])
        return h2


@contextmanager
def sae_ablation_context(
    model: PreTrainedModel,
    layer_indices: list[int],
    mask: torch.Tensor,
    *,
    injection_site: str = "residual",
) -> Iterator[None]:
    """Register mock SAE ablation hooks; always removed in ``finally``."""
    handles: list[torch.utils.hooks.RemovableHandle] = []
    interceptor = SAEInterceptor(mask)
    try:
        for L in layer_indices:
            mod = resolve_steering_module(model, int(L), injection_site)
            handles.append(mod.register_forward_hook(interceptor.hook_fn))
        yield
    finally:
        for h in handles:
            h.remove()
