"""Gather responses and cache their per-layer activations (derivation step 1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from creativity_steering.config import SamplingPreset
from creativity_steering.derivation.prompts import PromptSet
from creativity_steering.derivation.records import ResponseExample
from creativity_steering.models.backend import ModelBackend


def gather_responses(
    backend: ModelBackend,
    prompt_set: PromptSet,
    *,
    positions: Sequence[str],
    samples_per_prompt: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
    sampling: Mapping[str, SamplingPreset] | None = None,
    progress: bool = False,
) -> list[ResponseExample]:
    """Generate ``samples_per_prompt`` responses for every (framing, topic) and cache activations.

    Each sample uses a distinct seed so stochastic decoding yields a varied pool to filter.
    ``sampling`` maps a framing (``"creative"`` / ``"neutral"``) to its decoding preset; when
    given, the matching preset drives each pool's decode (e.g. high-temperature sampling for
    creative, greedy for neutral). When omitted, the scalar ``temperature`` is used for both.
    """
    items = prompt_set.items()
    out: list[ResponseExample] = []
    for idx, (framing, system, topic) in enumerate(items):
        preset = sampling.get(framing) if sampling is not None else None
        for s in range(samples_per_prompt):
            rec = backend.generate_and_capture(
                topic,
                system=system,
                positions=positions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed * 100_000 + idx * 100 + s,
                sampling=preset,
            )
            out.append(
                ResponseExample(
                    framing=framing,
                    system=system,
                    prompt=topic,
                    response=rec.text,
                    activations=rec.activations,
                )
            )
        if progress and (idx + 1) % 16 == 0:
            print(f"  gathered {idx + 1}/{len(items)} (framing,topic) pairs")
    return out


def gather_responses_unframed(
    backend: ModelBackend,
    topics: Sequence[str],
    *,
    positions: Sequence[str],
    samples_per_prompt: int = 3,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
    sampling: SamplingPreset | None = None,
    progress: bool = False,
) -> list[ResponseExample]:
    """Single-context generation for the ``score_split`` derivation.

    Generate ``samples_per_prompt`` responses per topic with **no system prompt** and one
    decoding preset, so the only thing distinguishing responses is the content itself (and
    sampling noise) — not a creative/neutral instruction. Each example is tagged
    ``framing="unframed"``; the filter splits the pool by evaluator score, not framing. Use
    ``samples_per_prompt > 1`` so a single topic yields a spread of creativity.
    """
    out: list[ResponseExample] = []
    for idx, topic in enumerate(topics):
        for s in range(samples_per_prompt):
            rec = backend.generate_and_capture(
                topic,
                system=None,
                positions=positions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed * 100_000 + idx * 100 + s,
                sampling=sampling,
            )
            out.append(
                ResponseExample(
                    framing="unframed",
                    system="",
                    prompt=topic,
                    response=rec.text,
                    activations=rec.activations,
                )
            )
        if progress and (idx + 1) % 16 == 0:
            print(f"  gathered {idx + 1}/{len(topics)} topics (unframed)")
    return out
