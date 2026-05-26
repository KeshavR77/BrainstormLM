"""Two-stage creativity evaluator.

Stage 1 — cheap, deterministic, model-free signals (DSI, lexical/slop redundancy,
length-normalised perplexity, a quality reward model). Stage 2 — an LLM-as-judge ensemble
scored on a coarse 1–5 Torrance-style rubric. ``composite`` fuses both into one scalar that
the caller thresholds to assign DiffMeans class membership.

Quickstart::

    from creativity_evaluators import evaluate, EvalConfig, PromptCategory

    result = evaluate(
        prompt="Invent an unusual use for a paperclip.",
        response="A tiny lightning rod for a dollhouse, grounding static into the carpet.",
        category=PromptCategory.CREATIVE,
        config=EvalConfig(enable_stage2=False),  # Stage 1 only — no API keys needed
    )
    print(result.creativity_score)

The pipeline entry points are imported lazily so that merely importing this package never
pulls in torch / transformers / API clients.
"""

from __future__ import annotations

from .types import (
    DIMENSIONS,
    DSIResult,
    EnsembleResult,
    EvalConfig,
    EvalResult,
    JudgeResult,
    JudgeSpec,
    LexicalResult,
    PPLResult,
    PromptCategory,
)

__all__ = [
    "DIMENSIONS",
    "DSIResult",
    "EnsembleResult",
    "EvalConfig",
    "EvalResult",
    "JudgeResult",
    "JudgeSpec",
    "LexicalResult",
    "PPLResult",
    "PromptCategory",
    "evaluate",
    "evaluate_batch",
]


def __getattr__(name: str):
    # Lazy re-export of the pipeline entry points (keeps `import creativity_evaluators`
    # free of heavy imports).
    if name in ("evaluate", "evaluate_batch"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
