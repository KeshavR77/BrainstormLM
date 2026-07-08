from __future__ import annotations

import torch


def orthogonal_reject(
    w: torch.Tensor,
    explored_rows: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return ``w - Proj_span(explored_rows)(w)`` in float32.

    ``explored_rows`` is ``[N, H]`` (rows need not be normalized). Uses QR on ``E.T`` so
    columns of ``Q`` form an orthonormal basis for the row span of ``explored_rows``.
    """
    if explored_rows.ndim != 2:
        raise ValueError(f"explored_rows must be [N, H], got {tuple(explored_rows.shape)}")
    if explored_rows.shape[0] == 0:
        return w.float()

    Et = explored_rows.T.float()  # [H, N]
    q, _ = torch.linalg.qr(Et, mode="reduced")
    wf = w.flatten().float()
    proj = q @ (q.T @ wf)
    return wf - proj
