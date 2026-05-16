"""
Orthogonal purification: remove components of a target direction along negative trait directions.

Uses an SVD basis for ``span(negative_vecs)`` so duplicate / dependent negatives do not inflate
the projector rank, then L2-normalizes to match steering-vector conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HyperPurifyReport:
    """Diagnostics for one hyper-purification pass."""

    retained_source_cosine: float
    max_abs_negative_cosine_before: float
    max_abs_negative_cosine_after: float
    removed_rank: int
    n_negative_vectors: int
    singular_values: list[float]


def _unit(v: torch.Tensor, *, eps: float) -> torch.Tensor:
    n = torch.linalg.norm(v)
    if n < eps:
        raise ValueError("vector is near-zero and cannot be normalized")
    return v / n


def _stack_negative_columns(
    target_vec: torch.Tensor,
    negative_vecs_list: list[torch.Tensor],
) -> torch.Tensor | None:
    v = target_vec.reshape(-1).float()
    if not negative_vecs_list:
        return None
    cols: list[torch.Tensor] = []
    for i, nvec in enumerate(negative_vecs_list):
        x = nvec.reshape(-1).float()
        if x.shape != v.shape:
            raise ValueError(
                f"negative_vecs_list[{i}] shape mismatch: got {tuple(x.shape)}, expected {tuple(v.shape)}"
            )
        cols.append(x)
    if not cols:
        return None
    return torch.stack(cols, dim=1)


def _max_abs_cosine_to_columns(v: torch.Tensor, cols: torch.Tensor | None, *, eps: float) -> float:
    if cols is None or cols.shape[1] == 0:
        return 0.0
    v_unit = _unit(v.reshape(-1).float(), eps=eps)
    col_norms = torch.linalg.norm(cols, dim=0).clamp_min(eps)
    c_unit = cols / col_norms
    return float(torch.abs(c_unit.T @ v_unit).max().item())


@torch.no_grad()
def purify_vector(
    target_vec: torch.Tensor,
    negative_vecs_list: list[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return a unit L2 vector orthogonal to the subspace spanned by ``negative_vecs_list``.

    If ``negative_vecs_list`` is empty, returns ``F.normalize(target_vec, dim=-1)``.
    """
    v = target_vec.reshape(-1).float()
    N = _stack_negative_columns(target_vec, negative_vecs_list)
    if N is None or N.shape[1] == 0:
        return _unit(v, eps=eps).to(dtype=target_vec.dtype, device=target_vec.device)

    # SVD (not QR): duplicate / dependent negatives must not inflate the projector rank.
    _u, s, _vh = torch.linalg.svd(N, full_matrices=False)
    smax = float(s[0].item()) if s.numel() else 0.0
    tol = max(eps * max(smax, 1.0), 1e-10)
    keep = s > tol
    if not keep.any():
        return _unit(v, eps=eps).to(dtype=target_vec.dtype, device=target_vec.device)
    q = _u[:, keep]
    proj = q @ (q.T @ v)
    v_out = v - proj
    norm = torch.linalg.norm(v_out)
    if norm < eps:
        raise ValueError(
            "purified vector is near-zero (target lies in span of negatives); cannot normalize"
        )
    v_unit = v_out / norm
    return v_unit.to(dtype=target_vec.dtype, device=target_vec.device)


@torch.no_grad()
def hyper_purify_vector(
    target_vec: torch.Tensor,
    negative_vecs_list: list[torch.Tensor],
    *,
    min_source_cosine: float = 0.15,
    max_negative_cosine: float = 0.08,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, HyperPurifyReport]:
    """
    Aggressively purify ``target_vec`` and return diagnostics.

    Compared with :func:`purify_vector`, this function exposes guardrails useful for
    artifact generation: it fails if the cleaned vector drifts too far from the source
    direction or remains too aligned to any negative vector.
    """
    v = target_vec.reshape(-1).float()
    N = _stack_negative_columns(target_vec, negative_vecs_list)
    before = _max_abs_cosine_to_columns(v, N, eps=eps)
    singular_values: list[float] = []
    removed_rank = 0

    if N is None or N.shape[1] == 0:
        out = _unit(v, eps=eps)
    else:
        _u, s, _vh = torch.linalg.svd(N, full_matrices=False)
        singular_values = [float(x) for x in s.detach().cpu().tolist()]
        smax = float(s[0].item()) if s.numel() else 0.0
        tol = max(eps * max(smax, 1.0), 1e-10)
        keep = s > tol
        removed_rank = int(keep.sum().item())
        if keep.any():
            q = _u[:, keep]
            v = v - q @ (q.T @ v)
        out = _unit(v, eps=eps)

    source_unit = _unit(target_vec.reshape(-1).float(), eps=eps)
    retained = float(torch.dot(source_unit, out).item())
    after = _max_abs_cosine_to_columns(out, N, eps=eps)
    if retained < min_source_cosine:
        raise ValueError(
            f"hyper purification over-scrubbed source direction: retained_source_cosine={retained:.4f} "
            f"< min_source_cosine={min_source_cosine:.4f}"
        )
    if after > max_negative_cosine:
        raise ValueError(
            f"hyper purification left too much negative alignment: max_abs_negative_cosine_after={after:.4f} "
            f"> max_negative_cosine={max_negative_cosine:.4f}"
        )
    report = HyperPurifyReport(
        retained_source_cosine=retained,
        max_abs_negative_cosine_before=before,
        max_abs_negative_cosine_after=after,
        removed_rank=removed_rank,
        n_negative_vectors=0 if N is None else int(N.shape[1]),
        singular_values=singular_values,
    )
    return out.to(dtype=target_vec.dtype, device=target_vec.device), report
