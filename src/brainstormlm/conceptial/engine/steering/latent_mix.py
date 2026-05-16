"""Load latent style registries and build weighted steering tensors per layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _load_steering_like(path: Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    missing = {"vector", "layer_index"} - set(bundle.keys())
    if missing:
        raise KeyError(f"checkpoint missing keys: {sorted(missing)}")
    return bundle


def load_latent_registry_from_manifest(manifest_path: str | Path) -> dict[str, dict[int, torch.Tensor]]:
    """
    Load a JSON manifest mapping tag -> layer (str) -> checkpoint path (relative to manifest dir).

    Each checkpoint is a ``.pt`` bundle with ``vector`` and ``layer_index`` (same layout as steering).
    """
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise TypeError("manifest root must be a JSON object")
    base = p.parent
    out: dict[str, dict[int, torch.Tensor]] = {}
    for tag, layer_map in raw.items():
        if not isinstance(layer_map, dict):
            raise TypeError(f"manifest[{tag!r}] must be an object mapping layer -> path")
        out[str(tag)] = {}
        for layer_s, rel in layer_map.items():
            ck = base / str(rel)
            bundle = _load_steering_like(ck)
            L = int(bundle["layer_index"])
            if int(layer_s) != L:
                raise ValueError(
                    f"manifest key layer {layer_s!r} != checkpoint layer_index {L} for {ck}"
                )
            out[str(tag)][L] = bundle["vector"].detach().cpu().float().contiguous()
    return out


def resolve_weighted_latent_vectors(
    registry: dict[str, dict[int, torch.Tensor]],
    weights: dict[str, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """
    ``v_mix(L) = sum_tag w_tag * v_{tag,L}`` for every layer appearing in any weighted tag.

    Unknown tags in ``weights`` raise ``KeyError``. Missing layer for a non-zero weight raises ``KeyError``.
    """
    if not weights:
        return {}
    layers: set[int] = set()
    for tag, w in weights.items():
        if abs(float(w)) < 1e-12:
            continue
        if tag not in registry:
            raise KeyError(f"unknown latent tag {tag!r} (not in registry)")
        layers.update(registry[tag].keys())
    acc: dict[int, torch.Tensor] = {}
    for L in sorted(layers):
        mix = None
        for tag, w in weights.items():
            wf = float(w)
            if abs(wf) < 1e-12:
                continue
            per = registry[tag]
            if L not in per:
                raise KeyError(f"layer {L} missing for latent tag {tag!r}")
            t = per[L].to(device=device, dtype=dtype) * wf
            mix = t if mix is None else mix + t
        if mix is not None:
            acc[int(L)] = mix
    return acc
