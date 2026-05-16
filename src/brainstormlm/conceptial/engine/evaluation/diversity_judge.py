"""
Intra-set semantic diversity for Conceptial Corpius 1 Creative Ops benchmarks (edge stack).

Uses the same local ``sentence-transformers`` stack as :class:`~shrooms_llm.evaluation.semantic_judge.SemanticJudge`.
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


def diversity_score_from_unit_embeddings(embeddings: torch.Tensor) -> float:
    """
    Given L2-normalized rows ``(n, d)``, return ``(1 - mean_{i<j} cos_sim_ij) * 100`` in ``[0, 100]``.
    """
    e = embeddings.float()
    if e.ndim != 2 or e.shape[0] < 2:
        return 0.0
    g = e @ e.t()
    n = int(g.shape[0])
    tri = torch.triu(torch.ones((n, n), device=g.device, dtype=torch.bool), diagonal=1)
    sims = g[tri]
    if sims.numel() == 0:
        return 0.0
    avg_sim = float(sims.mean().item())
    avg_sim = max(-1.0, min(1.0, avg_sim))
    return float(max(0.0, min(100.0, (1.0 - avg_sim) * 100.0)))


class DiversityJudge:
    """Local embedder for set-level diversity (default: ``all-MiniLM-L6-v2``)."""

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

    def calculate_set_diversity(self, texts: list[str]) -> float:
        """
        Embed non-empty strings; mean pairwise cosine similarity over all pairs ``i < j``;
        return ``(1 - mean_sim) * 100``. Empty or single-text sets return ``0.0``.
        """
        cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if len(cleaned) < 2:
            return 0.0
        emb = self._model.encode(cleaned, convert_to_tensor=True)
        emb = F.normalize(emb.float(), dim=-1)
        return diversity_score_from_unit_embeddings(emb)


_default_judge: DiversityJudge | None = None


def calculate_set_diversity(texts: list[str]) -> float:
    """Module-level helper using a process-wide default :class:`DiversityJudge` instance."""
    global _default_judge
    if _default_judge is None:
        _default_judge = DiversityJudge()
    return _default_judge.calculate_set_diversity(texts)
