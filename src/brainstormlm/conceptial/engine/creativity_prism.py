# This module has been superseded by the creativity_evaluator package.
# All symbols are re-exported from there for backwards compatibility.
# Import directly from creativity_evaluator.prism in new code.
from creativity_evaluator.prism import (
    parse_winner_arm_line,
    CREATIVITY_PRISM_SYSTEM,
    CREATIVITY_PRISM_SYSTEM_IDEA,
)
from creativity_evaluator.prism.judge import OpenAIJudge as _OpenAIJudge
from creativity_evaluator.prism.rubrics import (
    resolve_creativity_prism_rubric as _resolve,
)


def resolve_creativity_prism_rubric(name: str) -> str:
    return _resolve(name)


def run_creativity_prism_arm_comparison(
    original_prompt,
    arm_completions,
    *,
    model=None,
    temperature=0.2,
    max_chars_per_completion=8000,
    rubric="default",
):
    """Backwards-compatible wrapper — delegates to OpenAIJudge.compare_arms()."""
    import os
    resolved_model = (model or os.environ.get("SOBER_MODEL") or "gpt-4o").strip()
    judge = _OpenAIJudge(model=resolved_model)
    return judge.compare_arms(
        original_prompt,
        list(arm_completions),
        rubric=rubric,
        temperature=temperature,
        max_chars_per_completion=max_chars_per_completion,
    )


def run_sober_editor_responses(original_prompt, responses, *, model=None, temperature=0.2, rubric="default"):
    """Backwards-compatible wrapper — delegates to OpenAIJudge.compare_arms() with unnamed arms."""
    import os
    resolved_model = (model or os.environ.get("SOBER_MODEL") or "gpt-4o").strip()
    judge = _OpenAIJudge(model=resolved_model)
    seq = [str(x) for x in responses]
    if not (1 <= len(seq) <= 3):
        raise ValueError(f"Expected 1–3 candidate responses, got {len(seq)}")
    arm_completions = [(f"Candidate {i}", t) for i, t in enumerate(seq, 1)]
    verdict, _ = judge.compare_arms(
        original_prompt, arm_completions, rubric=rubric, temperature=temperature,
    )
    return verdict


def run_sober_editor(original_prompt, candidate_ideas, *, model=None, temperature=0.2, rubric="default"):
    ideas = list(candidate_ideas)
    if len(ideas) != 3:
        raise ValueError(f"Expected exactly 3 candidate ideas, got {len(ideas)}")
    return run_sober_editor_responses(original_prompt, ideas, model=model, temperature=temperature, rubric=rubric)


__all__ = [
    "CREATIVITY_PRISM_SYSTEM",
    "CREATIVITY_PRISM_SYSTEM_IDEA",
    "parse_winner_arm_line",
    "resolve_creativity_prism_rubric",
    "run_creativity_prism_arm_comparison",
    "run_sober_editor_responses",
    "run_sober_editor",
]
