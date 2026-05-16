"""
Local embedding-based semantic distance for investor / benchmark tooling.

Uses ``sentence-transformers`` on **CPU** by default (or **MPS** when explicitly requested) so the
judge does not occupy CUDA VRAM alongside the main LM. On Apple Silicon unified memory, CPU for
the small MiniLM model keeps pressure off the MPS working set used by generation.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F

_JudgeDevice = Literal["cpu", "mps", "cuda", "auto"]


def _resolve_device(pref: _JudgeDevice) -> str:
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if pref == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None or not mps.is_available():
            return "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return pref


class SemanticJudge:
    """
    Lightweight local embedder (default: ``all-MiniLM-L6-v2``) for cosine-based novelty signals.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: _JudgeDevice | None = None,
    ) -> None:
        raw = device or os.environ.get("SEMANTIC_JUDGE_DEVICE", "cpu")
        if raw not in ("cpu", "mps", "cuda", "auto"):
            raw = "cpu"
        self._device_str = _resolve_device(raw)  # type: ignore[arg-type]
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=self._device_str)

    def calculate_escape_score(self, text_baseline: str, text_steered: str) -> float:
        """
        Semantic escape score in ``[0, 200]`` scale (typically ``[0, 100]`` for valid cosine):

        ``S_C = cos(v1, v2)`` with unit-normalized embeddings; return ``(1 - S_C) * 100``.
        Higher = farther from the baseline embedding (more "escape" from that latent direction).
        """
        t1 = (text_baseline or "").strip()
        t2 = (text_steered or "").strip()
        if not t1 or not t2:
            return 0.0
        v1 = self._model.encode(t1, convert_to_tensor=True)
        v2 = self._model.encode(t2, convert_to_tensor=True)
        v1 = F.normalize(v1.float(), dim=-1)
        v2 = F.normalize(v2.float(), dim=-1)
        sc = float((v1 * v2).sum().item())
        sc = max(-1.0, min(1.0, sc))
        return float((1.0 - sc) * 100.0)
