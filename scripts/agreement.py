"""Compute evaluator-vs-human agreement on human-rated trajectories (plan-06).

Closes Phase 4 exit criterion 3: a numeric, archived metric of how well the
``gpt-4o-mini`` evaluator's score tracks the human thumbs/rating.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from brainstormlm.config import get_settings
from brainstormlm.feedback.human import read_all_human
from brainstormlm.feedback.trajectories import Trajectory


def _human_score(t: Trajectory) -> float | None:
    """Return a numeric encoding of the human rating, or None if missing."""
    if t.human_rating is None:
        return None
    if t.human_rating.rating is not None:
        return float(t.human_rating.rating)
    return 1.0 if t.human_rating.thumbs == "up" else 0.0


def _evaluator_total(t: Trajectory) -> float:
    return float(t.creativity + t.logic)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _signed_agreement(rows: list[Trajectory], threshold: float = 7.0) -> float | None:
    if not rows:
        return None
    matches = 0
    for t in rows:
        thumbs_up = (t.human_rating is not None) and (t.human_rating.thumbs == "up")
        score_high = _evaluator_total(t) >= threshold
        if thumbs_up == score_high:
            matches += 1
    return matches / len(rows)


def compute_agreement(
    rows: Iterable[Trajectory],
    *,
    threshold: float = 7.0,
) -> dict:
    """Return a JSON-friendly summary of evaluator-vs-human agreement."""
    rated = [t for t in rows if t.human_rating is not None]
    if not rated:
        return {
            "n": 0,
            "signed_agreement": None,
            "pearson": None,
            "per_domain": {},
            "threshold": threshold,
        }

    xs = [_evaluator_total(t) for t in rated]
    ys = [_human_score(t) for t in rated]
    ys_clean = [y for y in ys if y is not None]
    pearson = _pearson(xs, ys_clean) if len(xs) == len(ys_clean) else None

    per_domain: dict[str, dict] = {}
    by_domain: dict[str, list[Trajectory]] = {}
    for t in rated:
        by_domain.setdefault(t.domain, []).append(t)
    for domain, group in by_domain.items():
        per_domain[domain] = {
            "n": len(group),
            "signed_agreement": _signed_agreement(group, threshold),
            "pearson": _pearson(
                [_evaluator_total(t) for t in group],
                [s for s in (_human_score(t) for t in group) if s is not None],
            ),
        }

    return {
        "n": len(rated),
        "signed_agreement": _signed_agreement(rated, threshold),
        "pearson": pearson,
        "per_domain": per_domain,
        "threshold": threshold,
    }


cli = typer.Typer(add_completion=False, no_args_is_help=False)


@cli.command()
def main(
    feedback_path: Annotated[Path | None, typer.Option(help="Override feedback_human.jsonl path.")] = None,
    out: Annotated[Path | None, typer.Option(help="Where to write the JSON summary.")] = None,
    threshold: Annotated[float, typer.Option(help="Score threshold considered 'thumbs-up worthy'.")] = 7.0,
) -> None:
    """Print + persist evaluator-vs-human agreement over feedback_human.jsonl."""
    settings = get_settings()
    rows = list(read_all_human(path=feedback_path))
    summary = compute_agreement(rows, threshold=threshold)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))

    if out is None:
        out = settings.data_dir / "eval" / "agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(f"Summary written to: {out}", err=True)


if __name__ == "__main__":
    sys.exit(cli() or 0)
