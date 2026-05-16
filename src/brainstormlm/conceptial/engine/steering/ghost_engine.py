"""
Real-time ghost (unsteered) hidden capture and orthogonal rejection on the main path.

Each decode step: ghost forward without steering captures :math:`\\vec{g}_l` at listed layers;
the main forward applies hooks that remove :math:`\\mathrm{proj}_{\\vec{g}_l}(\\vec{h}_l)` from
activations before steering hooks run (nesting order: ghost outside, steering inside).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import PreTrainedModel

from brainstormlm.conceptial.engine.core.entropy_logits import RollingEntropyGammaCalibrator, topk_shannon_entropy_bits
from brainstormlm.conceptial.engine.hooks import _hidden_tensor, resolve_steering_module


def subtract_ghost_projection(
    h: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float = 1e-8,
    ghost_strength: float = 0.6,
) -> torch.Tensor:
    """
    Soft-remove component of ``h`` along ``g``: ``h - ghost_strength * proj(h, g)``.

    ``ghost_strength`` in ``[0, 1]``; ``1.0`` matches full orthogonal subtraction. Lower values
    preserve more of ``h`` for grammar/coherence under aggressive steering.
    Uses fp32 intermediates, then casts back to ``h.dtype``.
    """
    orig_dtype = h.dtype
    h32 = h.float()
    g32 = g.reshape(-1).to(device=h.device, dtype=torch.float32)
    g_exp = g32.view(1, 1, -1).expand_as(h32)
    gn = (g32 * g32).sum().clamp_min(torch.tensor(float(eps), device=h.device, dtype=torch.float32))
    # Scalar (h·g) / ||g||^2; projection vector is coeff * g (broadcast).
    coeff = (h32 * g_exp).sum(dim=-1, keepdim=True) / gn
    proj = coeff * g_exp
    st = max(0.0, min(float(ghost_strength), 1.0))
    out32 = h32 - st * proj
    return out32.to(dtype=orig_dtype)


def _ghost_hook_factory(
    layer_index: int,
    buf: dict[int, torch.Tensor | None],
    eps: float,
    gamma_holder: dict[str, float],
):
    def hook(_module: torch.nn.Module, _inp: Any, out: object) -> object:
        h = _hidden_tensor(out)
        g_vec = buf.get(layer_index)
        if g_vec is None:
            return out
        if h.shape[1] != 1:
            return out
        gs = float(gamma_holder.get("gamma", 0.6))
        h_new = subtract_ghost_projection(h, g_vec, eps=eps, ghost_strength=gs)
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new

    return hook


@contextmanager
def ghost_rejection_context(
    model: PreTrainedModel,
    ghost_layers: list[int],
    ghost_hidden_buf: dict[int, torch.Tensor | None],
    *,
    injection_site: str = "residual",
    eps: float = 1e-8,
    gamma_holder: dict[str, float] | None = None,
    ghost_strength: float = 0.6,
) -> Iterator[None]:
    """
    Forward hooks that apply soft ghost projection at ``ghost_layers``.

    ``gamma_holder`` (mutable ``{"gamma": float}``) is read on each forward so ghost strength can
    change per decode step (entropy gating). If omitted, a holder seeded with ``ghost_strength`` is used.
    """
    holder = gamma_holder if gamma_holder is not None else {"gamma": float(ghost_strength)}
    if "gamma" not in holder:
        holder["gamma"] = float(ghost_strength)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for L in ghost_layers:
            mod = resolve_steering_module(model, int(L), injection_site)
            handles.append(
                mod.register_forward_hook(
                    _ghost_hook_factory(int(L), ghost_hidden_buf, eps, gamma_holder=holder)
                )
            )
        yield
    finally:
        for h in handles:
            h.remove()


# V3.6 repetition penalty (HF-style on logits for tokens already in the prefix).
_REPETITION_PENALTY_RHO: float = 1.2


def build_grammar_safe_token_ids(tokenizer: Any) -> frozenset[int]:
    """Token ids that skip repetition penalty and (when last-emitted) ghost subtraction."""
    safe: set[int] = set()
    variants = (
        ".",
        ",",
        "\n",
        "the",
        "a",
        "and",
        " the",
        " a",
        " and",
        " The",
        " A",
        " And",
    )
    for piece in variants:
        try:
            ids = tokenizer.encode(piece, add_special_tokens=False)
        except Exception:
            continue
        for tid in ids:
            if isinstance(tid, int) and tid >= 0:
                safe.add(int(tid))
    return frozenset(safe)


def apply_repetition_penalty_logits_row(
    logits_1d: torch.Tensor,
    generated_token_ids: list[int],
    *,
    rho: float = _REPETITION_PENALTY_RHO,
    safe_token_ids: frozenset[int] | None = None,
) -> torch.Tensor:
    """
    For each **distinct** token id that appears in ``generated_token_ids`` and is not in
    ``safe_token_ids``, scale logits: if ``logit > 0`` divide by ``ρ``, else multiply by ``ρ``.
    """
    if rho <= 1.0 or not generated_token_ids:
        return logits_1d
    out = logits_1d.float().clone()
    safe = safe_token_ids or frozenset()
    for tid in set(int(x) for x in generated_token_ids):
        if tid in safe:
            continue
        v = out[tid]
        if v > 0:
            out[tid] = v / float(rho)
        else:
            out[tid] = v * float(rho)
    return out


def apply_min_p_filter(probs: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Min-P truncation: zero probabilities below ``rho * max(probs)``, renormalize.
    If nothing survives, fall back to argmax one-hot.
    """
    if rho <= 0.0 or probs.numel() == 0:
        return probs
    pmax = float(probs.max().item())
    if pmax <= 0.0:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    thr = float(rho) * pmax
    out = probs.clone()
    out[out < thr] = 0.0
    s = float(out.sum().item())
    if s <= 1e-12:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    return out / out.sum()


def _sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float,
    do_sample: bool,
    min_p: float,
) -> tuple[torch.Tensor, float]:
    """Sample one token from a 1D logits row; return ``(id shaped [1,1], nll)`` for the drawn token."""
    lf = logits_1d.float()
    if not do_sample:
        tid = int(lf.argmax().item())
        logps = torch.log_softmax(lf, dim=-1)
        nll = float(-logps[tid].item())
        return torch.tensor([[tid]], device=logits_1d.device, dtype=torch.long), nll
    if temperature <= 0:
        tid = int(lf.argmax().item())
        logps = torch.log_softmax(lf, dim=-1)
        nll = float(-logps[tid].item())
        return torch.tensor([[tid]], device=logits_1d.device, dtype=torch.long), nll
    scaled = lf / float(temperature)
    logps_full = torch.log_softmax(scaled, dim=-1)
    probs = logps_full.exp()
    if min_p > 0.0:
        probs = apply_min_p_filter(probs, float(min_p))
    logps = torch.log(probs.clamp_min(1e-12))
    next_id = torch.multinomial(probs.unsqueeze(0), num_samples=1).to(device=logits_1d.device)
    tid = int(next_id.item())
    nll = float(-logps[tid].item())
    return next_id, nll


def _append_completion_ppl_metrics(
    metrics_out: dict[str, Any] | None,
    token_nlls: list[float],
    *,
    sliding_window: int,
) -> None:
    if metrics_out is None:
        return
    if not token_nlls:
        metrics_out["completion_ppl_mean"] = None
        metrics_out["completion_ppl_sliding_max"] = None
        metrics_out["completion_ppl_sliding_min"] = None
        return
    from creativity_evaluator.metrics.perplexity import perplexity_stats_from_nlls

    st = perplexity_stats_from_nlls(token_nlls, sliding_window=int(sliding_window))
    metrics_out["completion_ppl_mean"] = float(st["ppl_mean"])
    metrics_out["completion_ppl_sliding_max"] = float(st["ppl_sliding_max"])
    # V8.2.1 Condition B: degenerate repetition (e.g. "the the the") produces a
    # sliding-window PPL near 1.0, which the high-PPL gate cannot detect.
    metrics_out["completion_ppl_sliding_min"] = float(st["ppl_sliding_min"])


def _max_consecutive_repeat(token_ids: list[int]) -> int:
    """V8.2.1 Condition C — longest run of identical consecutive token IDs in the completion.

    A loop like ``[the, the, the, the, the]`` returns 5; non-loopy decode is typically ≤ 2 even
    on legitimate doubled tokens (e.g. ``" — — "`` styles). The circuit breaker trips when this
    exceeds a small threshold (default 4) and forces an IOR-flush retry.
    """
    if not token_ids:
        return 0
    best = 1
    run = 1
    prev = token_ids[0]
    for tid in token_ids[1:]:
        if tid == prev:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
            prev = tid
    return int(best)


def _fill_ghost_metrics(
    metrics_out: dict[str, Any] | None,
    *,
    use_entropy: bool,
    entropy_steps: list[float],
    gamma_steps: list[float],
    ghost_strength: float,
) -> None:
    if metrics_out is None:
        return
    metrics_out["use_entropy_gated_ghost"] = bool(use_entropy)
    if use_entropy and entropy_steps:
        metrics_out["avg_entropy"] = float(sum(entropy_steps) / len(entropy_steps))
    else:
        metrics_out["avg_entropy"] = None
    if gamma_steps:
        metrics_out["avg_ghost_gamma"] = float(sum(gamma_steps) / len(gamma_steps))
    else:
        metrics_out["avg_ghost_gamma"] = float(ghost_strength)


@torch.no_grad()
def autoregressive_ghost_new_token_ids(
    model: PreTrainedModel,
    tokenizer: Any,
    prompt: str,
    steering_vectors: dict[int, torch.Tensor],
    steering_config: Any,
    ghost_layers: list[int],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.7,
    ghost_injection_site: str = "residual",
    ghost_eps: float = 1e-8,
    ghost_strength: float = 0.6,
    metrics_out: dict[str, Any] | None = None,
) -> torch.Tensor:
    """
    Return **new** token ids ``[1, T]`` (long) with ghost + steering decode loop.

    Approximately **2×** the forward passes of plain ``generate()`` per decode step (ghost + main).
    """

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    ghost_buf: dict[int, torch.Tensor | None] = {int(L): None for L in ghost_layers}
    gamma_holder: dict[str, float] = {"gamma": float(ghost_strength)}

    use_entropy = bool(getattr(steering_config, "use_entropy_gated_ghost", False))
    k_top = int(getattr(steering_config, "ghost_entropy_top_k", 50))
    entropy_cal = RollingEntropyGammaCalibrator() if use_entropy else None
    safe_ids = build_grammar_safe_token_ids(tokenizer)

    entropy_steps: list[float] = []
    gamma_steps: list[float] = []
    token_nlls: list[float] = []

    min_p = float(getattr(steering_config, "min_p", 0.05))
    ppl_win = int(getattr(steering_config, "ppl_sliding_window_tokens", 32))

    pre = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = pre.past_key_values
    logits = pre.logits[:, -1, :]
    eos_id = tokenizer.eos_token_id

    new_tokens: list[torch.Tensor] = []
    cur_len = int(input_ids.shape[1])

    for _step in range(int(max_new_tokens)):
        if use_entropy:
            h = topk_shannon_entropy_bits(logits[0], k=k_top)
            assert entropy_cal is not None
            gamma = entropy_cal.next_gamma(h, cold_start_gamma=float(ghost_strength))
            entropy_steps.append(float(h))
        else:
            gamma = float(ghost_strength)

        if new_tokens:
            if int(new_tokens[-1].item()) in safe_ids:
                gamma = 0.0

        gamma_holder["gamma"] = float(gamma)
        gamma_steps.append(float(gamma))

        rep_ids = [int(t.item()) for t in new_tokens]
        logits_row = apply_repetition_penalty_logits_row(
            logits[0], rep_ids, rho=_REPETITION_PENALTY_RHO, safe_token_ids=safe_ids
        )
        next_id, step_nll = _sample_next_token(
            logits_row,
            temperature=float(temperature),
            do_sample=bool(do_sample),
            min_p=min_p,
        )
        token_nlls.append(step_nll)
        tid = int(next_id.item())
        new_tokens.append(next_id)
        if eos_id is not None and tid == int(eos_id):
            break

        step_ids = next_id.to(device=device)
        m_out = ghost_decode_step_with_logits(
            model,
            step_ids,
            past,
            steering_vectors,
            steering_config,
            ghost_layers,
            ghost_buf,
            ghost_injection_site=ghost_injection_site,
            ghost_eps=float(ghost_eps),
            gamma_holder=gamma_holder,
        )

        past = m_out.past_key_values
        logits = m_out.logits[:, -1, :]
        cur_len += 1
        for L in ghost_layers:
            ghost_buf[int(L)] = None

        if cur_len > 8192:
            break

    _fill_ghost_metrics(
        metrics_out,
        use_entropy=use_entropy,
        entropy_steps=entropy_steps,
        gamma_steps=gamma_steps,
        ghost_strength=float(ghost_strength),
    )
    _append_completion_ppl_metrics(metrics_out, token_nlls, sliding_window=ppl_win)
    # V8.2.1 Condition C — expose the longest consecutive-same-token-id run.
    # V8.2.2 Conditions D/E/F — text-degeneracy signals the breakers needed to catch the
    # "comma-bleed" salad V8.2.1 missed:
    #   D: comma_density (commas / words) — catches "and,if,the,c,dy,is,lost" patterns
    #   E: output_token_count                — catches near-empty completions
    #   F: non_ascii_ratio                   — catches stray-script contamination
    if metrics_out is not None:
        emitted_tids = [int(t.item()) for t in new_tokens]
        metrics_out["max_consecutive_repeat"] = _max_consecutive_repeat(emitted_tids)
        metrics_out["output_token_count"] = len(emitted_tids)
        if emitted_tids:
            try:
                decoded_for_metrics = tokenizer.decode(
                    torch.tensor(emitted_tids, dtype=torch.long), skip_special_tokens=True
                )
            except Exception:
                decoded_for_metrics = ""
            if decoded_for_metrics:
                comma_count = decoded_for_metrics.count(",")
                # Word count: split on whitespace; for the comma-bleed pattern many "words"
                # are single tokens like "and,if" so we also fall back to splitting on commas
                # plus whitespace to get a finer count.
                whitespace_words = sum(1 for w in decoded_for_metrics.split() if w.strip(",.;:!?"))
                total_chars = sum(1 for c in decoded_for_metrics if not c.isspace())
                non_ascii_chars = sum(1 for c in decoded_for_metrics if ord(c) > 127)
                metrics_out["comma_density"] = float(
                    comma_count / max(1, whitespace_words)
                )
                metrics_out["non_ascii_ratio"] = float(non_ascii_chars / max(1, total_chars))
            else:
                metrics_out["comma_density"] = 0.0
                metrics_out["non_ascii_ratio"] = 0.0
        else:
            metrics_out["comma_density"] = 0.0
            metrics_out["non_ascii_ratio"] = 0.0

    if not new_tokens:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.cat(new_tokens, dim=1)


@torch.no_grad()
def ghost_decode_step_with_logits(
    model: PreTrainedModel,
    step_ids: torch.Tensor,
    past: Any,
    steering_vectors: dict[int, torch.Tensor],
    steering_config: Any,
    ghost_layers: list[int],
    ghost_buf: dict[int, torch.Tensor | None],
    *,
    ghost_injection_site: str,
    ghost_eps: float,
    gamma_holder: dict[str, float],
) -> Any:
    """One ghost capture + steered main forward; returns model output dict (``logits``, ``past_key_values``)."""
    from brainstormlm.conceptial.engine.hooks import steering_context

    g_out = model(
        input_ids=step_ids,
        past_key_values=past,
        attention_mask=None,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = g_out.hidden_states
    if hs is None:
        raise RuntimeError("Ghost forward requires output_hidden_states=True.")
    for L in ghost_layers:
        li = int(L) + 1
        if li >= len(hs):
            raise ValueError(f"ghost layer {L} out of range for hidden_states (len={len(hs)})")
        ghost_buf[int(L)] = hs[li][0, -1, :].detach()

    with ghost_rejection_context(
        model,
        [int(x) for x in ghost_layers],
        ghost_buf,
        injection_site=ghost_injection_site,
        eps=float(ghost_eps),
        gamma_holder=gamma_holder,
        ghost_strength=float(gamma_holder.get("gamma", 0.6)),
    ):
        with steering_context(model, steering_vectors, steering_config):
            m_out = model(
                input_ids=step_ids,
                past_key_values=past,
                attention_mask=None,
                use_cache=True,
                return_dict=True,
            )
    for L in ghost_layers:
        ghost_buf[int(L)] = None
    return m_out


@torch.no_grad()
def autoregressive_ghost_complete(
    model: PreTrainedModel,
    tokenizer: Any,
    prompt: str,
    steering_vectors: dict[int, torch.Tensor],
    steering_config: Any,
    ghost_layers: list[int],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.7,
    ghost_injection_site: str = "residual",
    ghost_eps: float = 1e-8,
    ghost_strength: float = 0.6,
    metrics_out: dict[str, Any] | None = None,
) -> str:
    """Decode **new** tokens only (same contract as :meth:`SteeringEngine.complete`)."""
    gen = autoregressive_ghost_new_token_ids(
        model,
        tokenizer,
        prompt,
        steering_vectors,
        steering_config,
        ghost_layers,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        ghost_injection_site=ghost_injection_site,
        ghost_eps=ghost_eps,
        ghost_strength=ghost_strength,
        metrics_out=metrics_out,
    )
    if gen.numel() == 0:
        return ""
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()
