"""Activation oracle — probe a direction for the trait of *creativity*.

The conformity paper feeds each layer's DiffMean to a LatentQA activation oracle and asks
both a free-form ("describe this direction") and a binary ("does this represent pressure
toward an incorrect consensus?") question, using the per-layer answers to localise the
trait. We mirror that interface for creativity.

Running the real oracle needs the LatentQA adapter weights + a GPU, so it lives behind an
interface with two implementations:

  * ``StubOracle`` — offline placeholder. Its trait score is the percentile of the probed
    vector's norm within a reference distribution, so on the planted-axis mock it produces
    a sensible per-layer curve and exercises the downstream report. It is **not** a real
    semantic probe; treat its labels as illustrative only.
  * ``LatentQAOracle`` — the real adapter slot. Construction documents exactly what to wire
    on RunPod; calling it before that raises ``NotImplementedError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

CREATIVITY_QUESTION = "Does this activation direction represent the trait of creativity?"


class ActivationOracle(ABC):
    @abstractmethod
    def describe(self, vector: np.ndarray, layer: int) -> str:
        """Free-form natural-language description of what the direction encodes."""

    @abstractmethod
    def score_trait(self, vector: np.ndarray, layer: int, question: str) -> float:
        """Probability in [0, 1] that the direction encodes the queried trait."""


class StubOracle(ActivationOracle):
    """Offline placeholder; trait score = norm percentile within a reference distribution."""

    def __init__(self, reference_norms: np.ndarray | None = None) -> None:
        self._ref = None if reference_norms is None else np.sort(np.asarray(reference_norms, float))

    def _percentile(self, value: float) -> float:
        if self._ref is None or self._ref.size == 0:
            return 0.5
        return float(np.searchsorted(self._ref, value, side="right") / self._ref.size)

    def describe(self, vector: np.ndarray, layer: int) -> str:
        conf = self.score_trait(vector, layer, CREATIVITY_QUESTION)
        tone = "strongly" if conf > 0.66 else "weakly" if conf < 0.33 else "moderately"
        return f"[stub] Layer {layer} direction {tone} resembles a creativity feature (conf={conf:.2f})."

    def score_trait(self, vector: np.ndarray, layer: int, question: str) -> float:  # noqa: ARG002
        return self._percentile(float(np.linalg.norm(np.asarray(vector, dtype=np.float32))))


class LatentQAOracle(ActivationOracle):
    """Real LatentQA activation oracle (RunPod). Documented slot — not implemented offline.

    To wire this up on a GPU pod, load a LatentQA decoder adapter trained for the target
    model (the conformity paper used
    ``adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B``; use the
    Qwen2.5-14B-compatible analogue), feed each per-layer DiffMean as the steered/"read"
    activation, and decode the free-form / binary answers. Implement ``describe`` and
    ``score_trait`` against that adapter here.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "LatentQAOracle is a documented slot. Wire the LatentQA adapter on a GPU pod "
            "(see class docstring). Use StubOracle for offline scaffolding."
        )

    def describe(self, vector: np.ndarray, layer: int) -> str:  # pragma: no cover - slot
        raise NotImplementedError

    def score_trait(self, vector: np.ndarray, layer: int, question: str) -> float:  # pragma: no cover
        raise NotImplementedError


def oracle_layer_scan(
    oracle: ActivationOracle,
    vectors: np.ndarray,
    *,
    question: str = CREATIVITY_QUESTION,
) -> np.ndarray:
    """Score every layer's vector with the oracle; return ``(n_layers,)`` trait scores."""
    vectors = np.asarray(vectors, dtype=np.float32)
    return np.array(
        [oracle.score_trait(vectors[L], L, question) for L in range(vectors.shape[0])],
        dtype=np.float32,
    )
