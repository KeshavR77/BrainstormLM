"""Top-k logit entropy (bits) for entropy-gated ghost strength (Conceptial Corpius 1 — V3.5 / V3.6)."""

from __future__ import annotations

import math
from collections import deque

import torch


def topk_shannon_entropy_bits(logits_1d: torch.Tensor, k: int = 50) -> float:
    """
    Shannon entropy in **bits** over the top-``k`` logits only (renormalized softmax).

    ``logits_1d`` shape ``[vocab]``.
    """
    x = logits_1d.float().flatten()
    n = int(x.numel())
    if n == 0:
        return 0.0
    kk = max(1, min(int(k), n))
    topv, _ = torch.topk(x, kk)
    p = torch.softmax(topv, dim=-1)
    p = p.clamp_min(1e-12)
    h = -(p * torch.log2(p)).sum()
    return float(h.item())


class RollingEntropyGammaCalibrator:
    """
    V3.6 dynamic entropy clutch: keep the last ``maxlen`` entropy readings, then map current ``H``
    to ghost strength ``γ`` using a z-score band around the rolling mean.

    Let ``μ`` and ``σ`` be the sample mean and **sample** standard deviation of the buffer
    (``σ = 0`` when fewer than two samples). Using thresholds ``μ ± σ``:

    - ``H < μ - σ`` (distribution unusually tight → cliché risk) → ``γ = γ_conf`` (default **0.85**).
    - ``H > μ + σ`` (distribution unusually flat → creative / grammar risk) → ``γ = γ_unc`` (default **0.1**).
    - Otherwise linear interpolation in ``H`` between ``(μ - σ, γ_conf)`` and ``(μ + σ, γ_unc)``.

    The **first** observed entropy uses no statistics: return ``cold_start_gamma`` (typically the
    static ``ghost_strength``), then append ``H`` so later steps are calibrated against the stream.
    """

    def __init__(
        self,
        maxlen: int = 20,
        *,
        gamma_confident: float = 0.85,
        gamma_uncertain: float = 0.1,
    ) -> None:
        self._buf: deque[float] = deque(maxlen=int(maxlen))
        self.gamma_confident = float(gamma_confident)
        self.gamma_uncertain = float(gamma_uncertain)

    def reset(self) -> None:
        self._buf.clear()

    def _mean_std(self) -> tuple[float, float]:
        vals = list(self._buf)
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        mu = sum(vals) / float(n)
        if n < 2:
            return mu, 0.0
        var = sum((x - mu) ** 2 for x in vals) / float(n - 1)
        return mu, math.sqrt(max(0.0, var))

    def next_gamma(self, h: float, *, cold_start_gamma: float = 0.6) -> float:
        """Observe ``h`` after computing ``γ`` from the buffer **excluding** this step."""
        if not self._buf:
            self._buf.append(float(h))
            return float(cold_start_gamma)

        mu, sig = self._mean_std()
        lo = mu - sig
        hi = mu + sig
        g_lo = self.gamma_confident
        g_hi = self.gamma_uncertain

        if sig <= 1e-8:
            if float(h) < mu:
                g = g_lo
            elif float(h) > mu:
                g = g_hi
            else:
                g = 0.5 * (g_lo + g_hi)
        elif float(h) <= lo:
            g = g_lo
        elif float(h) >= hi:
            g = g_hi
        else:
            t = (float(h) - lo) / max(2.0 * sig, 1e-12)
            g = g_lo + t * (g_hi - g_lo)

        self._buf.append(float(h))
        return float(max(0.0, min(1.0, g)))
