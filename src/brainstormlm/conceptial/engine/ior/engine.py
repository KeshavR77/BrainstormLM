from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from brainstormlm.conceptial.engine.config import SteeringConfig, SteeringStrategy
from brainstormlm.conceptial.engine.core.hidden_capture import last_token_hidden_at_layer, normalize_hidden
from brainstormlm.conceptial.engine.ior.rejection import orthogonal_reject
from brainstormlm.conceptial.engine.pipeline import SteeringEngine

_LOG = logging.getLogger(__name__)


def _concat_prompt_completion(tok: Any, prompt: str, completion: str) -> str:
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    c_ids = tok(completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_ids = torch.cat([p_ids, c_ids])
    return tok.decode(full_ids, skip_special_tokens=True)


def _device_cache_sweep_after_eviction() -> None:
    """Best-effort unified-memory / VRAM cache release after dropping tensor history."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover - driver edge cases
            _LOG.debug("cuda.empty_cache skipped: %s", e)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "is_available") and mps.is_available():
        try:
            mps.empty_cache()
        except Exception as e:  # pragma: no cover
            _LOG.debug("mps.empty_cache skipped: %s", e)


def _repulsion_direction(
    explored: torch.Tensor,
    *,
    hidden_size: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Gaussian probe rejected off span(explored); normalized."""
    dev_cpu = torch.device("cpu")
    g = generator or torch.Generator(device=dev_cpu)
    if explored.shape[0] == 0:
        w = torch.randn(hidden_size, generator=g, dtype=torch.float32, device=dev_cpu)
        return normalize_hidden(w)

    w = torch.randn(hidden_size, generator=g, dtype=torch.float32, device=dev_cpu)
    r = orthogonal_reject(w, explored.to(dtype=torch.float32))
    n = torch.linalg.vector_norm(r)
    if float(n) < 1e-5:
        w2 = torch.randn(hidden_size, generator=g, dtype=torch.float32, device=dev_cpu)
        r = orthogonal_reject(w2, explored.to(dtype=torch.float32))
    return normalize_hidden(r)


@dataclass
class IORRunState:
    """One completed IOR step (for logging / export)."""

    index: int
    prompt: str
    completion: str
    full_text: str
    do_steering: bool
    repulsion_vector_cpu: torch.Tensor | None
    last_token_hidden_cpu: torch.Tensor


@dataclass
class IOREngine:
    """
    Iterative Orthogonal Rejection in **native hidden space** (no external embedder).

    After each generation, stores the last-token hidden state at ``capture_layer`` for the
    (prompt + completion) string. The next step injects a repulsion vector — a unit direction
    in the orthogonal complement of the span of stored hiddens — at ``steering_layer``.

    Rejection memory (``_explored``) and ``history`` are FIFO-capped by ``max_history`` (default 5).
    Evicted CPU tensors are explicitly dropped and the allocator is nudged to keep footprint flat
    on long rollout chains (unified memory).

    Use :attr:`explored_len` as the routing step index (0 before the first stored completion) when
    pairing with :mod:`shrooms_llm.steering.semantic_router` on Conceptial’s **Corpius 1** edge API.
    """

    engine: SteeringEngine
    capture_layer: int = 28
    steering_layer: int = 28
    repulsion_alpha: float = 0.5
    #: Exponential decay on older FIFO hiddens when building repulsion span (V8.1); newest row weight 1.
    memory_decay_gamma: float = 0.65
    max_history: int = 5
    #: V8.2.1 manifold-stability clip on the repulsion direction before it enters the residual stream.
    #: Default ~25% of empirical residual-stream norm (~10 on Qwen2.5-3B L28). Env override:
    #: ``CONCEPTIAL_IOR_MAX_NORM``. NOTE: the repulsion vector is unit-normalized inside
    #: :func:`_repulsion_direction`, so the *direction*'s norm is 1.0; the effective injection
    #: magnitude that actually enters the residual stream is :attr:`repulsion_alpha` × 1.0 = 0.5
    #: by default. The clip therefore binds only if some upstream path bypasses the normalization.
    #: It still ships as the V8.2.1 safety bar.
    repulsion_max_norm: float = 2.5
    #: When True, run ``gc.collect()`` after FIFO evictions (recommended for long sweeps).
    eviction_gc: bool = True
    generator: torch.Generator | None = field(default=None, repr=False)
    _explored: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    history: list[IORRunState] = field(default_factory=list, init=False)
    #: V8.2.1 telemetry. Updated by ``_set_repulsion_steering`` on every arm.
    #: ``_last_ior_norm`` is the raw pre-clip norm of the repulsion direction tensor in
    #: model-dtype on the model device. ``_last_ior_norm_clipped`` is the value after the
    #: τ clip is (or is not) applied. ``_last_ior_effective`` is ``repulsion_alpha × clipped``,
    #: i.e. the actual L2 magnitude added per token to the residual stream at ``steering_layer``.
    _last_ior_norm: float = field(default=0.0, init=False, repr=False)
    _last_ior_norm_clipped: float = field(default=0.0, init=False, repr=False)
    _last_ior_effective: float = field(default=0.0, init=False, repr=False)

    @property
    def explored_len(self) -> int:
        return len(self._explored)

    def _evict_explored_fifo(self) -> int:
        """Pop oldest explored rows while over capacity. Returns number of evictions."""
        cap = max(0, int(self.max_history))
        n = 0
        while len(self._explored) > cap:
            old = self._explored.pop(0)
            del old
            n += 1
        return n

    def _evict_history_fifo(self) -> int:
        """Pop oldest IORRunState rows while over capacity. Returns number of evictions."""
        cap = max(0, int(self.max_history))
        n = 0
        while len(self.history) > cap:
            st = self.history.pop(0)
            del st
            n += 1
        return n

    def _append_explored(self, h_unit: torch.Tensor) -> None:
        row = h_unit.detach().cpu().contiguous()
        self._explored.append(row)
        evicted = self._evict_explored_fifo()
        if evicted and self.eviction_gc:
            gc.collect()
            _device_cache_sweep_after_eviction()

    def _append_history(self, st: IORRunState) -> None:
        self.history.append(st)
        evicted = self._evict_history_fifo()
        if evicted and self.eviction_gc:
            gc.collect()
            _device_cache_sweep_after_eviction()

    def _set_repulsion_steering(self, v_cpu: torch.Tensor) -> None:
        p = next(self.engine.model.parameters())
        v = v_cpu.to(device=p.device, dtype=p.dtype)
        # V8.2.1: clip the repulsion direction's L2 norm to τ before injection. ``v`` is normally
        # unit-normalized upstream, so this is a safety bar that fires only if some path delivers
        # a non-unit direction. Telemetry below reports raw and clipped norms either way.
        tau_env = os.environ.get("CONCEPTIAL_IOR_MAX_NORM")
        tau = float(tau_env) if tau_env is not None else float(self.repulsion_max_norm)
        v_norm = float(torch.linalg.vector_norm(v).item())
        if tau > 0.0 and v_norm > tau:
            v = v * (tau / v_norm)
            v_norm_clipped = tau
        else:
            v_norm_clipped = v_norm
        # V8.2.1: env-tunable alpha so a future smoke can drop magnitude without code change.
        alpha_env = os.environ.get("CONCEPTIAL_IOR_REPULSION_ALPHA")
        alpha = float(alpha_env) if alpha_env is not None else float(self.repulsion_alpha)
        self._last_ior_norm = v_norm
        self._last_ior_norm_clipped = v_norm_clipped
        self._last_ior_effective = float(alpha) * v_norm_clipped
        self.engine.config.layer_injections = []
        self.engine.config.layer_alphas = {int(self.steering_layer): float(alpha)}
        self.engine.config.strategy = SteeringStrategy.SINGLE
        self.engine.set_steering_vectors({int(self.steering_layer): v})

    def arm_repulsion_from_memory(self) -> torch.Tensor | None:
        """
        If ``_explored`` is non-empty, load the model and set steering to the repulsion direction
        (same construction as :meth:`step` when memory is present). Returns the CPU repulsion unit
        vector, or ``None`` when there is nothing to arm.
        """
        if not self._explored:
            return None
        self.engine.load()
        hidden_size = int(self.engine.model.config.hidden_size)
        em = torch.stack(self._explored, dim=0)
        n = int(em.shape[0])
        g = float(self.memory_decay_gamma)
        if n > 1 and 0.0 < g < 1.0:
            w = torch.tensor(
                [g ** (n - 1 - r) for r in range(n)],
                dtype=em.dtype,
                device=em.device,
            ).view(n, 1)
            em = em * w
        rep_vec_cpu = _repulsion_direction(
            em,
            hidden_size=hidden_size,
            generator=self.generator,
        )
        self._set_repulsion_steering(rep_vec_cpu)
        return rep_vec_cpu

    def step(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, IORRunState]:
        """One generation + latent capture. Returns (completion_only, state)."""
        self.engine.load()
        model = self.engine.model
        tokenizer = self.engine.tokenizer

        idx = len(self.history)
        rep_vec_cpu: torch.Tensor | None = None
        if len(self._explored) == 0:
            ctx_steering = False
            completion = self.engine.complete(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                do_steering=False,
            )
        else:
            rep_vec_cpu = self.arm_repulsion_from_memory()
            ctx_steering = True
            completion = self.engine.complete(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                do_steering=True,
            )

        full_text = _concat_prompt_completion(tokenizer, prompt, completion)
        h_last = last_token_hidden_at_layer(model, tokenizer, full_text, self.capture_layer)
        h_unit = normalize_hidden(h_last)

        st = IORRunState(
            index=idx,
            prompt=prompt,
            completion=completion,
            full_text=full_text,
            do_steering=ctx_steering,
            repulsion_vector_cpu=rep_vec_cpu.clone() if rep_vec_cpu is not None else None,
            last_token_hidden_cpu=h_unit.clone(),
        )
        self._append_history(st)
        self._append_explored(h_unit)

        return completion, st

    def run(
        self,
        prompt: str,
        *,
        iterations: int,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> list[IORRunState]:
        """Run ``iterations`` sequential IOR steps on the same ``prompt``."""
        out: list[IORRunState] = []
        for _ in range(int(iterations)):
            _, st = self.step(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            out.append(st)
        return out


def attach_iorengine(config: SteeringConfig | None = None, engine: SteeringEngine | None = None, **kwargs: Any) -> IOREngine:
    """Helper: build :class:`IOREngine` from an existing or freshly configured engine."""
    eng = engine or SteeringEngine(config or SteeringConfig())
    return IOREngine(eng, **kwargs)
