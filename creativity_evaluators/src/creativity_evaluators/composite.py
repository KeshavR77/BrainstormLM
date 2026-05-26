"""Composite scoring: fuse Stage 1 + Stage 2 into one creativity scalar.

Formula (Padmakumar et al., ICLR 2026, "novelty x quality" framing, harmonic-mean spirit)::

    novelty            = z(DSI) + z(PPL) - z(lexical_redundancy)
    novelty_combined   = mean(z(novelty), z(originality_s2))
    quality_combined   = mean(z(reward), z(pragmaticality_s2))
    creativity         = sqrt(max(0, novelty_combined) * max(0, quality_combined))
    final              = 0.85 * creativity + 0.15 * z(elaboration_s2)

Every ``z(.)`` is standardized against the **current batch**, so this is a batch operation
and thresholds are relative within a run. All raw and intermediate components are returned so
the formula can be re-tuned offline without re-running the judges.

Missing inputs are imputed to the batch mean (z = 0), so a too-short DSI or a Stage-1-only
record (no Stage-2 dimensions) contributes neutrally instead of crashing.
"""

from __future__ import annotations

import numpy as np

# Stage-1 + Stage-2 raw inputs expected per record.
COMPONENT_KEYS = (
    "dsi",
    "ppl",
    "lexical_redundancy",
    "reward",
    "originality_s2",
    "pragmaticality_s2",
    "elaboration_s2",
)


def zscores(values: list[float | None]) -> np.ndarray:
    """Batch z-scores (population std). None/NaN are imputed to the mean (z = 0); a constant
    batch (or all-missing) returns all zeros."""
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(len(values))
    mu = float(finite.mean())
    sd = float(finite.std())  # population std (ddof=0)
    if sd == 0.0:
        return np.zeros(len(values))
    return (np.nan_to_num(arr, nan=mu) - mu) / sd


def _column(records: list[dict], key: str) -> list[float | None]:
    return [r.get(key) for r in records]


def stage1_composite_batch(records: list[dict]) -> list[float]:
    """Stage-1-only scalar used to gate Stage 2 (the percentile floor).

    Equal-weight blend of the model-free novelty signal and the reward-model quality signal;
    needs only the Stage-1 keys (``dsi``, ``ppl``, ``lexical_redundancy``, ``reward``).
    """
    dsi_z = zscores(_column(records, "dsi"))
    ppl_z = zscores(_column(records, "ppl"))
    lex_z = zscores(_column(records, "lexical_redundancy"))
    novelty_z = zscores(list(dsi_z + ppl_z - lex_z))
    reward_z = zscores(_column(records, "reward"))
    return list(0.5 * novelty_z + 0.5 * reward_z)


def compute_composite_batch(
    records: list[dict],
    *,
    creativity_weight: float = 0.85,
    elaboration_weight: float = 0.15,
) -> tuple[list[float], list[dict]]:
    """Return ``(final_scores, per_record_components)`` for the batch."""
    dsi_z = zscores(_column(records, "dsi"))
    ppl_z = zscores(_column(records, "ppl"))
    lex_z = zscores(_column(records, "lexical_redundancy"))
    novelty = dsi_z + ppl_z - lex_z
    novelty_z = zscores(list(novelty))

    reward_z = zscores(_column(records, "reward"))
    orig_z = zscores(_column(records, "originality_s2"))
    prag_z = zscores(_column(records, "pragmaticality_s2"))
    elab_z = zscores(_column(records, "elaboration_s2"))

    novelty_combined = (novelty_z + orig_z) / 2.0
    quality_combined = (reward_z + prag_z) / 2.0
    creativity = np.sqrt(np.maximum(0.0, novelty_combined) * np.maximum(0.0, quality_combined))
    final = creativity_weight * creativity + elaboration_weight * elab_z

    components = [
        {
            "dsi_z": float(dsi_z[i]),
            "ppl_z": float(ppl_z[i]),
            "lexical_z": float(lex_z[i]),
            "novelty": float(novelty[i]),
            "novelty_z": float(novelty_z[i]),
            "reward_z": float(reward_z[i]),
            "originality_z": float(orig_z[i]),
            "pragmaticality_z": float(prag_z[i]),
            "elaboration_z": float(elab_z[i]),
            "novelty_combined": float(novelty_combined[i]),
            "quality_combined": float(quality_combined[i]),
            "creativity": float(creativity[i]),
        }
        for i in range(len(records))
    ]
    return [float(x) for x in final], components
