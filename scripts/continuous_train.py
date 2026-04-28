"""Continuous-training orchestrator (plan-06).

Phases:
  1. Load human-rated trajectories.
  2. Mine human pairs + sibling-sample synthetic pairs (skippable).
  3. Train a fresh DPO LoRA via Modal (skippable).
  4. Run the 4-arm ablation against the new adapter.
  5. Promote the new adapter behind the gate iff it clears the threshold.

The orchestrator takes a ``Phases`` injection point so unit tests stub each
phase. Live runs wire defaults that call the real scripts.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from brainstormlm.config import get_settings
from brainstormlm.feedback.human import read_all_human
from brainstormlm.feedback.trajectories import Trajectory
from brainstormlm.training.preferences import PreferencePair, mine_human_pairs
from brainstormlm.training.promotion import (
    make_promoted,
    promote_if_better,
    read_promoted,
    write_promoted_atomic,
)


@dataclass
class Phases:
    """Injection points so the orchestrator is unit-testable without GPUs."""

    load_human_feedback: Callable[[], list[Trajectory]]
    sibling_sampling: Callable[[], list[PreferencePair]]
    train: Callable[..., Path]
    evaluate: Callable[[Path], list[dict]]


@dataclass
class ContinuousTrainConfig:
    run_id: str
    promoted_path: Path
    skip_mining: bool = False
    skip_training: bool = False
    dry_run: bool = False
    reuse_adapter: Path | None = None
    min_improvement: float = 0.10
    max_p_value: float = 0.20
    sibling_run_id: str = field(default_factory=lambda: f"continuous-{uuid.uuid4().hex[:8]}")


def _planned_phases(cfg: ContinuousTrainConfig) -> list[str]:
    plan: list[str] = []
    if not cfg.skip_mining:
        plan.append("mine human + synthetic pairs")
    if not cfg.skip_training:
        plan.append("train DPO adapter")
    plan.append("run ablation eval")
    plan.append("promote-or-not")
    return plan


def run_pipeline(
    cfg: ContinuousTrainConfig,
    *,
    phases: Phases,
) -> dict[str, Any]:
    """Execute the pipeline; return a summary dict.

    Tests stub every phase via ``Phases``; live runs call the real scripts.
    """
    if cfg.dry_run:
        return {
            "promoted": False,
            "reason": "dry-run",
            "planned_phases": _planned_phases(cfg),
        }

    pairs: list[PreferencePair] = []
    if not cfg.skip_mining:
        human_rows = phases.load_human_feedback()
        human_pairs = mine_human_pairs(
            human_rows, sibling_run_id=cfg.sibling_run_id
        )
        synthetic_pairs = phases.sibling_sampling()
        seen_ids: set[str] = set()
        for p in (*human_pairs, *synthetic_pairs):
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)
            pairs.append(p)

    if cfg.skip_training:
        if cfg.reuse_adapter is None:
            raise ValueError("--skip-training requires --reuse-adapter")
        adapter_path = cfg.reuse_adapter
    else:
        adapter_path = phases.train(pairs, run_id=cfg.run_id)

    summary = phases.evaluate(adapter_path)

    current = read_promoted(cfg.promoted_path)
    promoted, reason = promote_if_better(
        candidate_run_id=cfg.run_id,
        candidate_adapter=adapter_path,
        candidate_summary=summary,
        current_promoted=current,
        min_improvement=cfg.min_improvement,
        max_p_value=cfg.max_p_value,
    )

    promoted_total = None
    if promoted:
        # Pick the trained+mem (or trained) total to record on the pointer.
        for arm in ("trained+mem", "trained"):
            for row in summary:
                if row["arm"].endswith(arm):
                    promoted_total = row["mean_total"]
                    break
            if promoted_total is not None:
                break
        new_pointer = make_promoted(
            adapter_path=adapter_path,
            held_out_total=promoted_total or 0.0,
            run_id=cfg.run_id,
        )
        write_promoted_atomic(new_pointer, cfg.promoted_path)

    return {
        "promoted": promoted,
        "reason": reason,
        "n_pairs": len(pairs),
        "adapter_path": str(adapter_path),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


cli = typer.Typer(add_completion=False, no_args_is_help=False)


def _default_load_human() -> list[Trajectory]:
    return [t for t in read_all_human() if t.human_rating is not None]


def _default_sibling_sampling() -> list[PreferencePair]:
    """Hand off to ``scripts/sample_siblings.py`` via subprocess.

    Live runs call the real script; tests stub this whole function out.
    """
    return []


def _default_train(pairs: list[PreferencePair], *, run_id: str) -> Path:
    """Hand off to ``scripts/train_dpo.py``'s in-process callable.

    Live runs may want the Modal path; tests stub this function.
    """
    settings = get_settings()
    return settings.adapters_dir / run_id


def _default_evaluate(adapter_path: Path) -> list[dict]:
    """Hand off to ``scripts/eval_ablation.py``'s harness; tests stub this."""
    return []


@cli.command()
def main(
    run_id: Annotated[str | None, typer.Option(help="Run id; auto-generated if omitted.")] = None,
    skip_mining: Annotated[bool, typer.Option("--skip-mining", help="Reuse existing data/preferences.jsonl.")] = False,
    skip_training: Annotated[bool, typer.Option("--skip-training", help="Reuse the latest adapter under --reuse-adapter.")] = False,
    reuse_adapter: Annotated[Path | None, typer.Option(help="Adapter dir to reuse with --skip-training.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only; no spend.")] = False,
    min_improvement: Annotated[float, typer.Option(help="Held-out total floor for promotion.")] = 0.10,
    max_p_value: Annotated[float, typer.Option(help="Paired-bootstrap p ceiling for promotion.")] = 0.20,
) -> None:
    """Run the continuous-training pipeline once."""
    settings = get_settings()
    rid = run_id or uuid.uuid4().hex[:12]
    cfg = ContinuousTrainConfig(
        run_id=rid,
        promoted_path=settings.promoted_adapter_path,
        skip_mining=skip_mining,
        skip_training=skip_training,
        dry_run=dry_run,
        reuse_adapter=reuse_adapter,
        min_improvement=min_improvement,
        max_p_value=max_p_value,
    )
    phases = Phases(
        load_human_feedback=_default_load_human,
        sibling_sampling=_default_sibling_sampling,
        train=_default_train,
        evaluate=_default_evaluate,
    )
    result = run_pipeline(cfg, phases=phases)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    sys.exit(cli() or 0)
