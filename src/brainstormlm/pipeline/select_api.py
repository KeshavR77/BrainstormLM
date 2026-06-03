"""API-shaped selection adapter — pipeline step ④ ("Select → n diverse ideas").

Thin layer over the pure :func:`brainstormlm.pipeline.select.select_by_repulsion`
primitive that returns exactly the response contract the public API expects
(``docs/brainstorm_pipeline_architecture.md``): a list of ``IdeaV1`` ({id, title,
body}) plus a ``DiversityScoreV1`` Vendi score on the chosen set.

The arch-doc step ④ is "quality-gate AND coverage MMR · Vendi score":
  1. quality gate — drop degenerate ("salad") candidates;
  2. coverage     — greedy embedding-repulsion over the survivors;
  3. measure      — Vendi score on the selected embeddings (single embed pass).

This module owns the API-schema coupling (``IdeaV1``/``DiversityScoreV1``) so the
core ``select.py`` primitive stays schema-free. Nothing calls this yet — it is
the drop-in for ``api/public.py`` step ④ once this branch merges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from brainstormlm.api.public_schemas import DiversityScoreV1, IdeaV1
from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD
from brainstormlm.eval.salad import salad_detector
from brainstormlm.pipeline.select import _embed, select_by_repulsion

_VENDI_METHOD = "vendi"


@dataclass(frozen=True)
class SelectedIdeas:
    """Step-④ output: the assembled ideas plus the diversity score."""

    ideas: list[IdeaV1]
    diversity: DiversityScoreV1


def _vendi_from_embeddings(emb: np.ndarray) -> float:
    """Vendi score from an embedding matrix (rows L2-normalised internally)."""
    n = emb.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    from vendi_score import vendi

    v = np.asarray(emb, dtype=np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return float(vendi.score_K(v @ v.T))


def _split_title_body(text: str) -> tuple[str, str]:
    """First non-empty line → title; everything after it → body."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            return line.strip(), "\n".join(lines[i + 1 :]).strip()
    return "", ""


def _mint_id(title: str, body: str, prefix: str) -> str:
    """Stable, deterministic id from the idea's content."""
    h = hashlib.sha256()
    h.update(title.encode("utf-8"))
    h.update(b"\x00")
    h.update(body.encode("utf-8"))
    return f"{prefix}_{h.hexdigest()[:12]}"


def select_response_ideas(
    candidates: Sequence[str],
    *,
    n: int,
    epsilon: float = DEFAULT_CLUSTER_THRESHOLD,
    quality: Sequence[float] | None = None,
    prior_embeddings: np.ndarray | None = None,
    drop_salad: bool = True,
    id_prefix: str = "idea",
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> SelectedIdeas:
    """Select ``n`` diverse, coherent ideas and shape them for the API response.

    Drops salad candidates (when ``drop_salad``), runs embedding-repulsion
    coverage selection over the survivors, parses each kept text into
    ``{id, title, body}``, and scores the chosen set with Vendi. Returns the
    ``IdeaV1`` list + ``DiversityScoreV1`` the public endpoint assembles into its
    response.
    """
    texts = list(candidates)
    if drop_salad:
        keep = [i for i, t in enumerate(texts) if not salad_detector(t)["is_salad"]]
    else:
        keep = list(range(len(texts)))

    gated = [texts[i] for i in keep]
    gated_quality = [quality[i] for i in keep] if quality is not None else None

    if not gated:
        return SelectedIdeas(ideas=[], diversity=DiversityScoreV1(score=0.0, method=_VENDI_METHOD))

    emb = np.asarray(embed_fn(gated) if embed_fn is not None else _embed(gated))
    selection = select_by_repulsion(
        gated,
        n=n,
        epsilon=epsilon,
        quality=gated_quality,
        prior_embeddings=prior_embeddings,
        candidate_embeddings=emb,
    )

    score = _vendi_from_embeddings(emb[selection.kept_indices])
    ideas: list[IdeaV1] = []
    for i in selection.kept_indices:
        title, body = _split_title_body(gated[i])
        ideas.append(IdeaV1(id=_mint_id(title, body, id_prefix), title=title, body=body))

    return SelectedIdeas(ideas=ideas, diversity=DiversityScoreV1(score=score, method=_VENDI_METHOD))
