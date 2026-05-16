"""
Pick a single inference device and dtype for HF ``from_pretrained``.

``device_map="auto"`` from Accelerate can offload layers to **disk** on memory-constrained
machines (e.g. 8B on 16GB unified memory), which makes generation extremely slow. The
default path maps the **entire** model to one accelerator: **CUDA** if available, else
**MPS** on Apple Silicon, else **CPU**.

Override with env ``SHROOMS_DEVICE`` ∈ {mps, cuda, cpu}.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch

PreferredDevice = Literal["auto", "mps", "cuda", "cpu"]


def resolve_accelerator(preferred: PreferredDevice = "auto") -> Literal["mps", "cuda", "cpu"]:
    env = (os.environ.get("SHROOMS_DEVICE") or "").strip().lower()
    if env in ("mps", "cuda", "cpu"):
        return env  # type: ignore[return-value]
    if preferred in ("mps", "cuda", "cpu"):
        return preferred  # type: ignore[return-value]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_single_device_map(preferred: PreferredDevice = "auto") -> dict[str, Any]:
    """Accelerate-style map placing all weights on one device (no disk offload)."""
    dev = resolve_accelerator(preferred)
    if dev == "cuda":
        return {"": 0}
    return {"": dev}


def resolve_load_torch_dtype(
    torch_dtype_setting: Literal["auto", "float16", "bfloat16", "float32"],
    accelerator: Literal["mps", "cuda", "cpu"],
) -> torch.dtype:
    if torch_dtype_setting != "auto":
        return getattr(torch, torch_dtype_setting)
    if accelerator == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if accelerator in ("cuda", "mps"):
        return torch.float16
    return torch.float32
