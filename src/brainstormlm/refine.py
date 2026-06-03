"""Refine step — wires the three generation-diversity modules into one flow.

Per ``docs/brainstorm_pipeline_architecture.md`` the refine step turns a raw prompt
into a structured *brief* (enabling, not prescriptive) and then drives the diverge
side. Here that means, in one call:

    brief-refine (N1 host model)
        -> planner_angles.plan_angles      (N2 — K orthogonal angles, embedding-deduped)
            -> verbalized_sample + tail_sample per angle  (N1 — set-aware generation)
                -> diversity_critic.find_least_distinct    (X4 — flag, regenerate, never rewrite)

Selection-to-``n`` (MMR / Conditional-Vendi) is deliberately NOT done here — that lives
in the selection worktree (``pipeline/select.py``). ``refine`` returns the diverse
*candidate pool* (one idea per surviving angle) plus diagnostics.

The LLM and embedder are injected (``make_llm`` / ``make_embedder`` build production
adapters; tests pass fakes), so the orchestration is exercised with no API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from brainstormlm.config import get_settings
from brainstormlm.conceptial.anti_mode import OUTPUT_MODE_INSTRUCTIONS
from brainstormlm.generation.diversity_critic import find_least_distinct, mean_distinctness
from brainstormlm.generation.planner_angles import plan_angles
from brainstormlm.generation.verbalized_sampling import tail_sample, verbalized_sample


# ---------------------------------------------------------------------------
# Protocols (tests inject plain callables)
# ---------------------------------------------------------------------------


class LLMCallable(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Adapters — bridge the real clients (tuple returns, system+messages signature)
# to the simple ``llm(prompt)->str`` / ``embedder(texts)->ndarray`` the modules want.
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a precise assistant. Follow the user's instructions exactly and "
    "return only what is asked, in the requested format."
)


def _provider_client(provider: str):
    if provider == "gemini":
        from brainstormlm.clients import gemini as client
    elif provider == "openai":
        from brainstormlm.clients import openai as client
    elif provider == "together":
        from brainstormlm.clients import together as client
    else:
        raise ValueError(f"unknown refine provider: {provider!r}")
    return client


def make_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    system: str = _REFINE_SYSTEM,
    max_tokens: int = 512,
    temperature: float | None = None,
) -> LLMCallable:
    """Return ``llm(prompt, **kw) -> str`` backed by the configured chat client.

    ``temperature`` / ``max_tokens`` may be overridden per-call via kwargs — the
    calibration sweep relies on this to vary temperature.
    """
    settings = get_settings()
    provider = provider or settings.refine_provider
    model = model or settings.refine_model
    client = _provider_client(provider)

    def _llm(prompt: str, **kwargs) -> str:
        temp = kwargs.pop("temperature", temperature)
        if temp is None:
            temp = settings.default_temperature
        mt = kwargs.pop("max_tokens", max_tokens)
        text, _usage = client.complete(
            system,
            [{"role": "user", "content": prompt}],
            model,
            max_tokens=mt,
            temperature=temp,
        )
        return text

    return _llm


def make_embedder(*, model: str | None = None) -> EmbedCallable:
    """Return ``embedder(texts) -> (N, D) ndarray`` backed by ``openai.embed``."""
    from brainstormlm.clients import openai as client

    def _embed(texts: list[str]) -> np.ndarray:
        matrix, _usage = client.embed(texts, model=model)
        return matrix

    return _embed


# ---------------------------------------------------------------------------
# Brief refine (enabling, not prescriptive)
# ---------------------------------------------------------------------------

_BRIEF_SYSTEM = (
    "You are a brief-refinement assistant for a creative brainstorming system. "
    "Rewrite the user's request into a compact, structured brief covering: objective, "
    "audience, hard constraints, and desired output format. CRITICAL: the brief must be "
    "ENABLING, NOT PRESCRIPTIVE — sharpen WHAT is being asked and pin constraints, but do "
    "NOT choose a creative direction, propose ideas, or commit to any single approach. "
    "If background context is given, extract only facts from it, never ideas."
)


def refine_brief(
    prompt: str,
    context: str | None = None,
    *,
    llm: LLMCallable,
    output_mode: str = "generic",
) -> str:
    """One LLM call: raw prompt (+ optional context) -> a structured, enabling brief."""
    parts = [f"Request: {prompt.strip()}"]
    if context:
        parts.append(f"Context (extract facts only, no ideas): {context.strip()}")
    mode_instr = OUTPUT_MODE_INSTRUCTIONS.get(output_mode)
    if mode_instr:
        parts.append(f"Intended downstream idea style (for the output-format section only): {mode_instr}")
    parts.append(_BRIEF_SYSTEM)
    return llm("\n\n".join(parts)).strip()


# ---------------------------------------------------------------------------
# Result + orchestration
# ---------------------------------------------------------------------------


@dataclass
class RefineResult:
    """Output of one refine pass.

    Attributes:
        brief: the structured, enabling brief.
        angles: the deduped orthogonal angles the pool was built from.
        ideas: one candidate idea per surviving angle (the diverse pool).
        distinctness: mean pairwise distinctness (1 - cosine sim) of ``ideas``.
        replaced: indices regenerated by the diversity-critic loop, in order.
    """

    brief: str
    angles: list[str]
    ideas: list[str]
    distinctness: float
    replaced: list[int] = field(default_factory=list)


def _generate_for_angle(
    brief: str,
    angle: str,
    *,
    llm: LLMCallable,
    n: int,
    modal_floor: float,
    rng: np.random.Generator,
) -> str:
    """Verbalized-sample for one angle and tail-sample a single candidate from it."""
    vs_brief = f"{brief}\n\nAngle: {angle}"
    items = verbalized_sample(vs_brief, llm=llm, n=n)
    if not items:
        return ""
    return tail_sample(items, rng=rng, modal_floor=modal_floor).text


def refine(
    prompt: str,
    context: str | None = None,
    *,
    k: int = 6,
    vs_n: int = 4,
    modal_floor: float = 0.1,
    llm: LLMCallable,
    embedder: EmbedCallable,
    dedupe_threshold: float = 0.95,
    replace_threshold: float = 0.9,
    max_replacements: int = 2,
    output_mode: str = "generic",
    rng: np.random.Generator | None = None,
) -> RefineResult:
    """Full refine/diverge pass returning a diverse candidate pool + diagnostics.

    Args:
        prompt: the raw user brainstorming request.
        context: optional background; facts are extracted, never ideas.
        k: number of orthogonal angles to request (pool upper bound after dedupe).
        vs_n: candidates per verbalized-sampling call (we tail-sample one).
        modal_floor: tail-sampling floor for the modal candidate (0 = never pick it).
        llm: ``llm(prompt, **kw) -> str`` (use ``make_llm`` in prod).
        embedder: ``embedder(texts) -> (N, D) ndarray`` (use ``make_embedder`` in prod).
        dedupe_threshold: cosine cutoff for angle dedup.
        replace_threshold: regenerate a slot whose mean cosine sim to the rest exceeds this.
        max_replacements: hard cap on critic-driven regenerations.
        output_mode: brief output-style hint (anti_mode modes).
        rng: optional numpy generator for reproducible tail sampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    brief = refine_brief(prompt, context, llm=llm, output_mode=output_mode)
    angles = plan_angles(
        brief, k=k, llm=llm, embedder=embedder, dedupe_threshold=dedupe_threshold
    )

    ideas: list[str] = [
        _generate_for_angle(brief, angle, llm=llm, n=vs_n, modal_floor=modal_floor, rng=rng)
        for angle in angles
    ]

    replaced: list[int] = []
    for _ in range(max_replacements):
        if len(ideas) < 2:
            break
        req = find_least_distinct(ideas, embedder=embedder)
        if req is None or req.mean_sim_to_others <= replace_threshold:
            break
        i = req.index
        angle = angles[i] if i < len(angles) else angles[-1]
        # Regenerate the redundant slot via the generator — authorship stays in the
        # diverger; the critic only flags (arXiv 2604.01029).
        ideas[i] = _generate_for_angle(
            brief, angle, llm=llm, n=vs_n, modal_floor=modal_floor, rng=rng
        )
        replaced.append(i)

    distinctness = mean_distinctness(ideas, embedder=embedder) if len(ideas) >= 2 else 0.0

    return RefineResult(
        brief=brief,
        angles=angles,
        ideas=ideas,
        distinctness=distinctness,
        replaced=replaced,
    )


__all__ = [
    "LLMCallable",
    "EmbedCallable",
    "make_llm",
    "make_embedder",
    "refine_brief",
    "RefineResult",
    "refine",
]
