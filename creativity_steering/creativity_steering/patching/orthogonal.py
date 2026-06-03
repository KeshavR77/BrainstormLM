"""X1: per-sample orthogonal steering (STARS-style) — Stiefel manifold basis + batch injection.

Implements two primitives:

``stiefel_basis(base, k, seed)``
    Returns K mutually orthonormal directions on the Stiefel manifold V_K(R^d).
    Algorithm: form the matrix [base | random_perturbations] and QR-decompose it.
    The first column is guaranteed to align with ``base``; subsequent columns span
    orthogonal directions seeded by random perturbations.  This is a standard
    Gram-Schmidt / QR construction on the Stiefel manifold.

    References: STARS (arXiv:2601.22010), Riemannian steering (arXiv:2511.08305).

``inject_orthogonal_batch(hidden_states, dirs, alpha, rescale)``
    Add a per-sample direction to a batched hidden-state tensor [B, T, H].
    Batch element i receives direction dirs[i], scaled by alpha.  When
    ``rescale=True`` the rescale convention matches the existing hook:

        delta_i = (v_i / |v_i|) * |h_token| * alpha

    Since ``stiefel_basis`` already returns unit-norm rows, this simplifies to:

        delta_i = v_i * |h_token| * alpha

IMPORTANT — non-identifiability caveat (arXiv:2602.06801): orthogonal directions in
the residual stream do NOT guarantee orthogonal (semantically distinct) outputs.
Test geometry; validate semantic spread with a real model run.
"""

from __future__ import annotations

import numpy as np
import torch


def stiefel_basis(
    base: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return K mutually orthonormal directions spanning a k-frame on the Stiefel manifold.

    The first returned direction is collinear with ``base`` (unit-normalised).
    The remaining K-1 directions are orthonormal completions seeded from random
    perturbations of ``base`` via QR decomposition (Gram-Schmidt).

    Parameters
    ----------
    base:
        1-D array of shape ``(hidden_dim,)``.  Need not be unit-norm, but must be
        non-zero.
    k:
        Number of orthonormal directions to return.  Must satisfy ``1 <= k <= hidden_dim``.
    seed:
        NumPy RNG seed for the random perturbations.
    eps:
        Numerical zero threshold for detecting degenerate base vectors.

    Returns
    -------
    np.ndarray of shape ``(k, hidden_dim)``, dtype float32.
        Rows are mutually orthonormal: ``dirs @ dirs.T ≈ I_k``.
    """
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    d = base.shape[0]
    bnorm = float(np.linalg.norm(base))
    if bnorm < eps:
        raise ValueError("base vector is zero (or near-zero); cannot form an orthonormal basis")
    if not (1 <= k <= d):
        raise ValueError(
            f"k={k} is out of range for hidden_dim={d}; need 1 <= k <= hidden_dim"
        )

    # Build a (d, k) matrix whose first column is the normalised base and the
    # remaining k-1 columns are random perturbations.  QR gives an orthonormal basis.
    rng = np.random.default_rng(seed)
    cols = np.empty((d, k), dtype=np.float32)
    cols[:, 0] = base / bnorm
    if k > 1:
        perturb = rng.normal(size=(d, k - 1)).astype(np.float32)
        cols[:, 1:] = perturb

    # Thin QR: Q has shape (d, k) with orthonormal columns.
    Q, _ = np.linalg.qr(cols)  # Q: (d, k)
    dirs = Q.T.astype(np.float32)  # (k, d)

    # Guarantee the first direction aligns with base (QR may flip the sign).
    if np.dot(dirs[0], base) < 0:
        dirs[0] = -dirs[0]

    return dirs


def inject_orthogonal_batch(
    hidden_states: torch.Tensor,
    dirs: np.ndarray,
    alpha: float,
    *,
    rescale: bool = True,
) -> torch.Tensor:
    """Add per-sample steering directions to a batched hidden-state tensor.

    Applies direction ``dirs[i]`` to batch element ``i``, optionally rescaled by the
    per-token activation norm (matching the existing ``_make_hook`` convention in
    ``models/hf.py``).

    Parameters
    ----------
    hidden_states:
        Float tensor of shape ``(batch_size, seq_len, hidden_dim)``.
    dirs:
        NumPy array of shape ``(batch_size, hidden_dim)`` with unit-norm rows
        (e.g. from ``stiefel_basis``).
    alpha:
        Steering strength.  Same semantics as ``Injection.alpha``.
    rescale:
        If True, match each token's activation norm so ``alpha`` transfers across
        model sizes.  The formula applied per token ``t`` of sample ``i`` is:

            delta = (dirs[i] / |dirs[i]|) * |h[i,t]| * alpha

        Since ``dirs`` rows are unit-norm this is simply:

            delta = dirs[i] * |h[i,t]| * alpha

    Returns
    -------
    torch.Tensor of shape ``(batch_size, seq_len, hidden_dim)`` — a new tensor,
    input is not modified.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be 3-D (batch, seq, hidden); got {tuple(hidden_states.shape)}"
        )
    batch_size, seq_len, hidden_dim = hidden_states.shape
    dirs_np = np.asarray(dirs, dtype=np.float32)
    if dirs_np.ndim != 2 or dirs_np.shape[0] != batch_size:
        raise ValueError(
            f"dirs must have shape (batch_size, hidden_dim) = ({batch_size}, {hidden_dim}); "
            f"got {tuple(dirs_np.shape)}"
        )
    if dirs_np.shape[1] != hidden_dim:
        raise ValueError(
            f"dirs hidden_dim {dirs_np.shape[1]} does not match hidden_states hidden_dim {hidden_dim}"
        )

    # Convert dirs to a torch tensor on the same device/dtype as hidden_states.
    v = torch.tensor(dirs_np, dtype=hidden_states.dtype, device=hidden_states.device)
    # v: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim) for broadcasting over seq.
    v = v.unsqueeze(1)

    if rescale:
        # h_norm: (batch_size, seq_len, 1)
        h_norm = hidden_states.norm(dim=-1, keepdim=True)
        # Each direction is already unit-norm, so (v / |v|) == v.
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        delta = (v / v_norm) * h_norm * alpha
    else:
        delta = v * alpha

    return hidden_states + delta


def orthonormalize_rows(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return a mutually-orthonormal set spanning the same row space, row order preserved.

    Gram–Schmidt (via QR of ``vectors.T``) in the given row order: output row ``i`` is input row
    ``i`` with its components along rows ``0..i-1`` removed, then unit-normalised.  Row 0 is the
    normalised input row 0.  Each output row is sign-aligned so ``dot(out[i], vectors[i]) >= 0``
    (generalising the row-0 sign fix in ``stiefel_basis``).

    This is the X1×X2 primitive: feed it the K archetype DiffMeans directions to *decorrelate the
    bank*.  Order matters — Gram–Schmidt distorts the later rows most, so order the inputs
    most-distinct-first (see ``distinctness_order``) to keep the salient modes intact.

    Parameters
    ----------
    vectors:
        ``(k, hidden_dim)`` array with ``k <= hidden_dim`` (can't have more than ``hidden_dim``
        mutually orthonormal directions).

    Returns
    -------
    np.ndarray of shape ``(k, hidden_dim)``, float32, with ``out @ out.T ≈ I_k``.
    """
    M = np.asarray(vectors, dtype=np.float32)
    if M.ndim != 2:
        raise ValueError(f"vectors must be 2-D (k, hidden_dim); got shape {M.shape}")
    k, d = M.shape
    if k > d:
        raise ValueError(
            f"cannot orthonormalise {k} rows in a {d}-dim space; need k <= hidden_dim"
        )
    # QR of M.T (d, k): Q has orthonormal columns = Gram–Schmidt of M's rows in order.
    q, _ = np.linalg.qr(M.T)  # q: (d, k)
    out = q.T.astype(np.float32)  # (k, d)
    # Sign-align each output row with its source row (QR may flip signs).
    for i in range(k):
        if float(np.dot(out[i], M[i])) < 0.0:
            out[i] = -out[i]
    return out


def distinctness_order(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return an index permutation ordering rows most-distinct-first.

    Greedy farthest-point selection on cosine similarity: the first pick is the most isolated row
    (smallest maximum similarity to any other), then each subsequent pick is the remaining row
    *farthest* (smallest maximum cosine) from the already-selected set.  The most-redundant rows
    therefore land last — which is exactly where Gram–Schmidt distortion is least harmful.

    Returns
    -------
    np.ndarray of int indices, length ``k``.
    """
    V = np.asarray(vectors, dtype=np.float32)
    k = V.shape[0]
    if k <= 1:
        return np.arange(k, dtype=int)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    unit = V / np.maximum(norms, eps)
    sim = unit @ unit.T  # (k, k) cosine similarity
    np.fill_diagonal(sim, -np.inf)  # ignore self-similarity
    first = int(np.argmin(sim.max(axis=1)))  # most isolated row
    order = [first]
    remaining = [j for j in range(k) if j != first]
    while remaining:
        sel = np.asarray(order)
        max_sim_to_sel = sim[np.ix_(remaining, sel)].max(axis=1)
        nxt = remaining[int(np.argmin(max_sim_to_sel))]  # farthest from the selected set
        order.append(nxt)
        remaining.remove(nxt)
    return np.asarray(order, dtype=int)
