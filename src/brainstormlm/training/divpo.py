"""DivPO data-logging flywheel — STUB (no trainer).

Diverse Preference Optimization (Lanchantin et al. 2025, arXiv:2501.18101) trains
on pairs that reward rare-but-good outputs over common-but-weak ones. The LoRA
training itself is DEFERRED in the Build-Plan; this module only mines and LOGS
the pairs so the flywheel starts priming now.

  - "rare" = small semantic-cluster membership within a prompt's candidate batch
    (a singleton idea is rarest), via the shared
    :func:`brainstormlm.eval.mode_collapse.cluster_labels` primitive.
  - "good" = higher quality score.

A DivPO pair (chosen = rare-and-good, rejected = common-and-weak) is accepted
when the chosen candidate is strictly rarer than the rejected one AND clears a
quality gap. Pairs are persisted as :class:`PreferencePair` JSONL via
``training.preferences`` so existing readers work unchanged. Because
``PreferencePair.source`` is locked to ``"sibling_sampling"`` (``extra="forbid"``),
DivPO origin is encoded in ``sibling_run_id`` as ``"divpo:<run_id>"``; extending
the ``source`` Literal to ``"divpo"`` is the follow-up when training lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD, cluster_labels
from brainstormlm.training import preferences as _preferences
from brainstormlm.training.preferences import PreferencePair, pair_id

DEFAULT_MIN_QUALITY_GAP = 2.0
STUB_MODEL = "stub"


@dataclass(frozen=True)
class DivpoCandidate:
    """One scored generation for a single prompt."""

    text: str
    quality_score: float
    temperature: float = 0.0


@dataclass(frozen=True)
class DivpoConfig:
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    min_quality_gap: float = DEFAULT_MIN_QUALITY_GAP
    domain: str = "divpo"
    generator_model: str = STUB_MODEL
    evaluator_model: str = STUB_MODEL
    rubric_version: int = 0


def _embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised OpenAI embeddings via the cached client."""
    from brainstormlm.clients.openai import embed

    matrix, _ = embed(list(texts))
    return matrix


def _rarity(
    texts: Sequence[str],
    *,
    threshold: float,
    embeddings: np.ndarray | None = None,
) -> list[float]:
    """Per-candidate rarity = 1 / cluster size (a singleton scores 1.0)."""
    emb = embeddings if embeddings is not None else _embed(texts)
    labels = cluster_labels(emb, threshold=threshold)
    sizes: dict[int, int] = {}
    for lab in labels:
        sizes[lab] = sizes.get(lab, 0) + 1
    return [1.0 / sizes[lab] for lab in labels]


def mine_divpo_pairs(
    prompt: str,
    candidates: Sequence[DivpoCandidate],
    *,
    run_id: str,
    config: DivpoConfig | None = None,
    embeddings: np.ndarray | None = None,
) -> list[PreferencePair]:
    """Mine (rare-and-good vs common-and-weak) preference pairs for one prompt.

    A pair is emitted for every ordered (chosen, rejected) where chosen is
    strictly rarer than rejected, the quality gap clears ``min_quality_gap``,
    and the texts differ. Fewer than two candidates → no pairs.
    """
    cfg = config or DivpoConfig()
    cand = list(candidates)
    if len(cand) < 2:
        return []

    rarity = _rarity(
        [c.text for c in cand],
        threshold=cfg.cluster_threshold,
        embeddings=embeddings,
    )
    timestamp = _preferences._utc_now_iso()

    pairs: list[PreferencePair] = []
    for ci, chosen in enumerate(cand):
        for ri, rejected in enumerate(cand):
            if ci == ri or chosen.text == rejected.text:
                continue
            if rarity[ci] <= rarity[ri]:
                continue
            gap = chosen.quality_score - rejected.quality_score
            if gap < cfg.min_quality_gap:
                continue
            pairs.append(
                PreferencePair(
                    id=pair_id(prompt, chosen.text, rejected.text),
                    timestamp=timestamp,
                    domain=cfg.domain,
                    prompt=prompt,
                    chosen=chosen.text,
                    rejected=rejected.text,
                    chosen_score=float(chosen.quality_score),
                    rejected_score=float(rejected.quality_score),
                    score_gap=float(gap),
                    chosen_temperature=float(chosen.temperature),
                    rejected_temperature=float(rejected.temperature),
                    generator_model=cfg.generator_model,
                    evaluator_model=cfg.evaluator_model,
                    rubric_version=cfg.rubric_version,
                    source="sibling_sampling",
                    sibling_run_id=f"divpo:{run_id}",
                )
            )
    return pairs


def log_divpo_round(
    path: Path | str,
    prompt: str,
    candidates: Sequence[DivpoCandidate],
    *,
    run_id: str,
    config: DivpoConfig | None = None,
    embeddings: np.ndarray | None = None,
) -> list[PreferencePair]:
    """Mine DivPO pairs and append them to the JSONL log at ``path``.

    Returns the mined pairs. Append-only — re-running adds rows rather than
    overwriting, so the flywheel accumulates across rounds.
    """
    pairs = mine_divpo_pairs(
        prompt, candidates, run_id=run_id, config=config, embeddings=embeddings
    )
    target = Path(path)
    for pair in pairs:
        _preferences.append(pair, target)
    return pairs
