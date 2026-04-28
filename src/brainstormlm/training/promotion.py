"""Adapter-promotion gate + atomic pointer write (plan-06).

The promoted-adapter pointer lives at ``settings.promoted_adapter_path`` and
is the single source of truth the API server reads on each request to know
which adapter to load. Writes are atomic (temp file → ``os.replace``) so a
mid-write crash never leaves a half-promoted pointer.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Promoted(BaseModel):
    """Pointer to the currently-promoted adapter (and optional steering)."""

    model_config = ConfigDict(extra="forbid")

    adapter_path: str | None = None
    steering_path: str | None = None
    held_out_total: float | None = None
    promoted_at: str | None = None
    run_id: str | None = None


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_promoted(path: Path) -> Promoted:
    if not path.exists():
        return Promoted()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Promoted.model_validate(payload)


def write_promoted_atomic(promoted: Promoted, path: Path) -> None:
    """Write JSON then ``os.replace`` so the pointer flips atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(promoted.model_dump(), sort_keys=True, indent=2))
    try:
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _pick_arm_total(summaries: Iterable[dict], preferred: tuple[str, ...]) -> tuple[float | None, float | None]:
    """Return (mean_total, p_value) for the first matching arm in ``preferred``."""
    by_arm = {row["arm"]: row for row in summaries}
    for name in preferred:
        for arm, row in by_arm.items():
            if arm.endswith(name):
                return row.get("mean_total"), row.get("p_value")
    return None, None


def promote_if_better(
    *,
    candidate_run_id: str,
    candidate_adapter: Path,
    candidate_summary: list[dict],
    current_promoted: Promoted,
    candidate_steering: Path | None = None,
    min_improvement: float = 0.10,
    max_p_value: float = 0.20,
) -> tuple[bool, str]:
    """Return ``(promoted, reason)``.

    Decision rule:
      1. First-ever promotion (current.held_out_total is None) → promote.
      2. Improvement < ``min_improvement`` → reject.
      3. Paired-bootstrap p-value > ``max_p_value`` → reject.
      4. Otherwise → promote.
    """
    candidate_total, p_value = _pick_arm_total(
        candidate_summary, preferred=("trained+mem", "trained")
    )
    if candidate_total is None:
        return False, "no trained arm in summary"

    if current_promoted.held_out_total is None:
        return True, f"first promotion (total={candidate_total:.3f})"

    delta = candidate_total - current_promoted.held_out_total
    if delta < min_improvement:
        return False, f"improvement {delta:+.3f} below {min_improvement:+.3f}"

    if p_value is not None and p_value > max_p_value:
        return False, f"p-value {p_value:.3f} > {max_p_value:.3f}"

    return True, f"passed gate (total={candidate_total:.3f}, Δ={delta:+.3f})"


def make_promoted(
    *,
    adapter_path: Path,
    held_out_total: float,
    run_id: str,
    steering_path: Path | None = None,
) -> Promoted:
    return Promoted(
        adapter_path=str(adapter_path),
        steering_path=str(steering_path) if steering_path is not None else None,
        held_out_total=held_out_total,
        promoted_at=_utc_now_iso(),
        run_id=run_id,
    )
