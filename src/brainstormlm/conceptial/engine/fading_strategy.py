"""
Alpha schedules over generated-token steps (V2/V3 steering).

Default linear fading: ``current_alpha = start_alpha * max(0, 1 - step / decay_tokens)``.

Also supports constant, exponential decay, and sigmoid ramp (phase-shift experiments).
"""

from __future__ import annotations

import math
from typing import Any


def linear_decay_current_alpha(start_alpha: float, step: int, decay_tokens: int) -> float:
    """
    Return steering scale at ``step`` (0-based generated-token index).

    If ``decay_tokens <= 0``, returns ``start_alpha`` unchanged.
    """
    if decay_tokens <= 0:
        return float(start_alpha)
    return float(start_alpha) * max(0.0, 1.0 - (float(step) / float(decay_tokens)))


def infer_alpha_schedule(inj: dict[str, Any]) -> str:
    """Infer schedule when ``alpha_schedule`` is omitted."""
    explicit = inj.get("alpha_schedule")
    if explicit is not None:
        return str(explicit)
    if int(inj.get("decay_tokens", 0)) > 0:
        return "linear"
    return "constant"


def current_alpha_for_injection(step: int, inj: dict[str, Any]) -> float:
    """
    Effective steering scale at generated-token index ``step`` for one injection dict.

    ``inj`` must include ``start_alpha`` and keys understood by :func:`infer_alpha_schedule`.
    """
    start = float(inj["start_alpha"])
    sched = infer_alpha_schedule(inj)
    if sched == "constant":
        return start
    if sched == "linear":
        return linear_decay_current_alpha(start, step, int(inj["decay_tokens"]))
    if sched == "exp_decay":
        z = max(1, int(inj.get("exp_zero_by_step", 100)))
        floor = 1e-4
        lam = -math.log(floor / max(start, 1e-12)) / float(z)
        return float(start * math.exp(-lam * float(step)))
    if sched == "sigmoid_ramp":
        mid = float(inj.get("sigmoid_mid_step", 50.0))
        k = float(inj.get("sigmoid_steepness", 0.12))
        return float(start * (1.0 / (1.0 + math.exp(-k * (float(step) - mid)))))
    raise ValueError(f"unknown alpha_schedule: {sched!r}")


class FadingScheduler:
    """
    Per-injection scheduler: fixed ``start_alpha`` and horizon ``decay_tokens``.

    Hooks advance an external step counter; this object only maps step → alpha.
    """

    __slots__ = ("_start_alpha", "_decay_tokens")

    def __init__(self, start_alpha: float, decay_tokens: int) -> None:
        self._start_alpha = float(start_alpha)
        self._decay_tokens = int(decay_tokens)

    def alpha_at_step(self, step: int) -> float:
        return linear_decay_current_alpha(self._start_alpha, step, self._decay_tokens)
