"""Gather responses and cache their per-layer activations (derivation step 1)."""

from __future__ import annotations

from collections.abc import Sequence

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
    progress: bool = False,
) -> list[ResponseExample]:
    """Generate ``samples_per_prompt`` responses for every (framing, topic) and cache activations.

    Each sample uses a distinct seed so stochastic decoding yields a varied pool to filter.
    """
    items = prompt_set.items()
    out: list[ResponseExample] = []
    for idx, (framing, system, topic) in enumerate(items):
        for s in range(samples_per_prompt):
            rec = backend.generate_and_capture(
                topic,
                system=system,
                positions=positions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed * 100_000 + idx * 100 + s,
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
