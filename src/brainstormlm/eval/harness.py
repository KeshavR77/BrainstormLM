"""Four-arm ablation harness: base / base+mem / trained / trained+mem.

Per the plan-04 verification gate, we score every held-out prompt under each
arm and report mean creativity+logic plus a paired-bootstrap p-value vs the
base arm. Held-out prompts MUST NOT be persisted to the training pool — the
caller is responsible for routing rows to a separate eval JSONL.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from brainstormlm.eval.diversity import diversity_report
from brainstormlm.eval.stats import paired_bootstrap_p_value
from brainstormlm.feedback.evaluator import EvaluatorResult
from brainstormlm.memory import LessonStore, augmented_system, retrieve

ARMS: tuple[str, ...] = ("base", "base+mem", "trained", "trained+mem")


class _GenWithMeta(Protocol):
    def generate_with_meta(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, bool]: ...


ScoreFn = Callable[[str, str], EvaluatorResult]


@dataclass
class AblationRow:
    arm: str
    domain: str
    prompt: str
    response: str
    creativity: int
    logic: int
    total: int
    memory_context_ids: list[str] = field(default_factory=list)
    adapter_path: str | None = None
    generator_cached: bool = False


@dataclass
class AblationSummary:
    arm: str
    n: int
    mean_creativity: float
    mean_logic: float
    mean_total: float
    mean_delta_vs_base: float | None
    p_value: float | None


def _arm_uses_memory(arm: str) -> bool:
    return arm.endswith("+mem")


def _arm_uses_trained(arm: str) -> bool:
    return arm.startswith("trained")


def _system_for_arm(
    arm: str, prompt: str, store: LessonStore | None, *, k_success: int, k_failure: int
) -> tuple[str | None, list[str]]:
    if not _arm_uses_memory(arm) or store is None:
        return None, []
    lessons = retrieve(store, prompt, k_success=k_success, k_failure=k_failure)
    return augmented_system(None, lessons), [lesson.id for lesson in lessons]


def run_ablation(
    held_out: Sequence[tuple[str, str]],
    *,
    base_generator: _GenWithMeta,
    trained_generator: _GenWithMeta,
    score_fn: ScoreFn,
    store: LessonStore | None = None,
    adapter_path: Path | None = None,
    k_success: int = 3,
    k_failure: int = 2,
    temperature: float = 1.0,
    max_tokens: int = 600,
    arm_label_prefix: str = "",
) -> list[AblationRow]:
    """Run 4 arms × every (domain, prompt) pair; return one row per arm × prompt.

    The caller decides what to do with the rows (persist, summarize). Memory
    arms are skipped silently when ``store`` is None.
    """
    rows: list[AblationRow] = []
    adapter_str = str(adapter_path) if adapter_path is not None else None

    for arm in ARMS:
        if _arm_uses_memory(arm) and store is None:
            continue
        gen = trained_generator if _arm_uses_trained(arm) else base_generator
        for domain, prompt in held_out:
            system, mem_ids = _system_for_arm(
                arm, prompt, store, k_success=k_success, k_failure=k_failure
            )
            response, cached = gen.generate_with_meta(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = score_fn(prompt, response)
            rows.append(
                AblationRow(
                    arm=f"{arm_label_prefix}{arm}",
                    domain=domain,
                    prompt=prompt,
                    response=response,
                    creativity=result.creativity,
                    logic=result.logic,
                    total=result.creativity + result.logic,
                    memory_context_ids=mem_ids,
                    adapter_path=adapter_str if _arm_uses_trained(arm) else None,
                    generator_cached=cached,
                )
            )
    return rows


def summarize(
    rows: Sequence[AblationRow],
    *,
    n_iterations: int = 10_000,
    seed: int = 42,
) -> list[AblationSummary]:
    """Compute per-arm summary stats with paired bootstrap p-values vs base.

    The "base" arm is whichever arm name ends in ``base`` (so a prefixed
    label like ``steered_base`` still anchors its own table).
    """
    by_arm: dict[str, list[AblationRow]] = {}
    for row in rows:
        by_arm.setdefault(row.arm, []).append(row)

    # Pick the arm whose name ends in "base" — supports prefixed labels.
    base_arm_name = next(
        (name for name in by_arm if name.endswith("base")),
        "base",
    )
    base_rows = by_arm.get(base_arm_name, [])
    base_by_prompt: dict[tuple[str, str], int] = {
        (r.domain, r.prompt): r.total for r in base_rows
    }

    # Iterate in ARMS order, matching by suffix so prefixed arms still sort
    # consistently (base, base+mem, trained, trained+mem).
    ordered_names: list[str] = []
    for canonical in ARMS:
        for actual in by_arm:
            if actual.endswith(canonical) and actual not in ordered_names:
                ordered_names.append(actual)
    for actual in by_arm:
        if actual not in ordered_names:
            ordered_names.append(actual)

    rng = random.Random(seed)
    summaries: list[AblationSummary] = []
    for arm in ordered_names:
        arm_rows = by_arm.get(arm, [])
        if not arm_rows:
            continue
        n = len(arm_rows)
        mean_c = sum(r.creativity for r in arm_rows) / n
        mean_l = sum(r.logic for r in arm_rows) / n
        mean_t = sum(r.total for r in arm_rows) / n
        if arm == base_arm_name or not base_by_prompt:
            mean_delta: float | None = None
            p: float | None = None
        else:
            deltas = [
                float(r.total - base_by_prompt[(r.domain, r.prompt)])
                for r in arm_rows
                if (r.domain, r.prompt) in base_by_prompt
            ]
            mean_delta = sum(deltas) / len(deltas) if deltas else None
            p = (
                paired_bootstrap_p_value(deltas, n_iterations=n_iterations, rng=rng)
                if deltas
                else None
            )
        summaries.append(
            AblationSummary(
                arm=arm,
                n=n,
                mean_creativity=mean_c,
                mean_logic=mean_l,
                mean_total=mean_t,
                mean_delta_vs_base=mean_delta,
                p_value=p,
            )
        )
    return summaries


def format_summary_table(summaries: Sequence[AblationSummary]) -> str:
    """Render the 4-row ablation table as text."""
    header = f"{'arm':<14}{'creativity':>12}{'logic':>8}{'total':>8}{'Δ vs base':>12}{'p':>10}"
    lines = [header, "-" * len(header)]
    for s in summaries:
        delta = "—" if s.mean_delta_vs_base is None else f"{s.mean_delta_vs_base:+.3f}"
        p = "—" if s.p_value is None else f"{s.p_value:.4f}"
        lines.append(
            f"{s.arm:<14}{s.mean_creativity:>12.3f}{s.mean_logic:>8.3f}"
            f"{s.mean_total:>8.3f}{delta:>12}{p:>10}"
        )
    return "\n".join(lines)


@dataclass
class DiversitySummary:
    arm: str
    n: int
    distinct2: float
    repetition_index: float
    near_duplicate_rate: float
    vendi_score: float


def compute_arm_diversity(rows: Sequence[AblationRow]) -> list[DiversitySummary]:
    """Compute per-arm diversity metrics (Distinct-2, Vendi, near-dup) for a set of ablation rows.

    Responses per arm are pooled together for the diversity computation, measuring
    how varied the outputs are across all prompts for that arm.
    """
    by_arm: dict[str, list[str]] = {}
    for row in rows:
        by_arm.setdefault(row.arm, []).append(row.response)

    summaries: list[DiversitySummary] = []
    for arm, responses in by_arm.items():
        report = diversity_report(responses)
        summaries.append(
            DiversitySummary(
                arm=arm,
                n=len(responses),
                distinct2=report["distinct2"],
                repetition_index=report["repetition_index"],
                near_duplicate_rate=report["near_duplicate_rate"],
                vendi_score=report["vendi_score"],
            )
        )
    return summaries


def format_diversity_table(summaries: Sequence[DiversitySummary]) -> str:
    """Render per-arm diversity metrics as a text table."""
    header = (
        f"{'arm':<14}{'n':>6}{'distinct2':>12}{'rep_idx':>10}"
        f"{'near_dup':>10}{'vendi':>10}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.arm:<14}{s.n:>6}{s.distinct2:>12.4f}{s.repetition_index:>10.4f}"
            f"{s.near_duplicate_rate:>10.4f}{s.vendi_score:>10.4f}"
        )
    return "\n".join(lines)
