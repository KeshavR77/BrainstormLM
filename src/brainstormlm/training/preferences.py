"""PreferencePair pydantic schema + append-only JSONL helpers.

Mirrors the Trajectory schema in `feedback/trajectories.py`: schema changes
are additive only; bump `schema_version` for incompatible changes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from brainstormlm.feedback import evaluator as _evaluator
from brainstormlm.feedback.trajectories import Trajectory
from brainstormlm.model.interface import Generator


def pair_id(prompt: str, chosen: str, rejected: str) -> str:
    """Deterministic 64-char hex id for a (prompt, chosen, rejected) triple."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(chosen.encode("utf-8"))
    h.update(b"\x00")
    h.update(rejected.encode("utf-8"))
    return h.hexdigest()


class PreferencePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: str
    timestamp: str
    domain: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    score_gap: float
    chosen_temperature: float
    rejected_temperature: float
    generator_model: str
    evaluator_model: str
    rubric_version: int
    source: Literal["sibling_sampling"] = "sibling_sampling"
    sibling_run_id: str


def append(p: PreferencePair, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(p.model_dump(), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_all(path: Path) -> Iterator[PreferencePair]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield PreferencePair.model_validate_json(line)


@dataclass
class SiblingSample:
    """One scored generator sample drawn for a single prompt at a given temperature."""

    response: str
    temperature: float
    creativity: int
    logic: int
    rationale: str
    failure_modes: list[str] = field(default_factory=list)
    generator_cached: bool = False
    evaluator_cached: bool = False

    @property
    def score(self) -> float:
        return float(self.creativity + self.logic)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sample_and_score_siblings(
    prompt: str,
    *,
    n: int,
    temperatures: list[float],
    generator: Generator,
    max_tokens: int = 600,
) -> list[SiblingSample]:
    """Draw n generator samples for `prompt`, score each, and return them.

    Cycles through `temperatures`; the cache key naturally varies because
    temperature is part of it (see `clients/together.py`).
    """
    if not temperatures:
        raise ValueError("temperatures must be a non-empty list")
    if n <= 0:
        raise ValueError("n must be positive")

    siblings: list[SiblingSample] = []
    for i in range(n):
        t = temperatures[i % len(temperatures)]
        response, g_cached = generator.generate_with_meta(
            prompt,
            system=None,
            temperature=t,
            max_tokens=max_tokens,
        )
        result, e_usage = _evaluator.score(prompt, response)
        siblings.append(
            SiblingSample(
                response=response,
                temperature=t,
                creativity=result.creativity,
                logic=result.logic,
                rationale=result.rationale,
                failure_modes=list(result.failure_modes),
                generator_cached=g_cached,
                evaluator_cached=e_usage.cached,
            )
        )
    return siblings


def mine_pairs(
    prompt: str,
    *,
    domain: str,
    siblings: list[SiblingSample],
    sibling_run_id: str,
    generator_model: str,
    evaluator_model: str,
    rubric_version: int,
    min_gap: float = 2.0,
) -> list[PreferencePair]:
    """All (chosen, rejected) sibling combinations whose gap clears ``min_gap``.

    Each ordered pair where ``chosen.score - rejected.score >= min_gap`` and
    the response texts differ becomes one ``PreferencePair``. With 6 siblings
    this typically yields 4–10 pairs per prompt — substantially more training
    signal than best-vs-worst at no extra sampling cost (the siblings are
    already drawn and cache-served on re-runs).
    """
    if len(siblings) < 2:
        return []

    timestamp = _utc_now_iso()
    pairs: list[PreferencePair] = []
    for chosen in siblings:
        for rejected in siblings:
            if chosen is rejected:
                continue
            gap = chosen.score - rejected.score
            if gap < min_gap:
                continue
            if chosen.response == rejected.response:
                continue
            pairs.append(
                PreferencePair(
                    id=pair_id(prompt, chosen.response, rejected.response),
                    timestamp=timestamp,
                    domain=domain,
                    prompt=prompt,
                    chosen=chosen.response,
                    rejected=rejected.response,
                    chosen_score=chosen.score,
                    rejected_score=rejected.score,
                    score_gap=gap,
                    chosen_temperature=chosen.temperature,
                    rejected_temperature=rejected.temperature,
                    generator_model=generator_model,
                    evaluator_model=evaluator_model,
                    rubric_version=rubric_version,
                    source="sibling_sampling",
                    sibling_run_id=sibling_run_id,
                )
            )
    return pairs


def _human_score(t: Trajectory) -> tuple[int, int, int]:
    """Sort key for human-rated trajectories: (rating, critique_len, response_len).

    Higher is better; ``rating`` defaults to 0 if absent.
    """
    rating = (
        t.human_rating.rating
        if (t.human_rating is not None and t.human_rating.rating is not None)
        else 0
    )
    critique_len = (
        len(t.human_rating.critique)
        if (t.human_rating is not None and t.human_rating.critique is not None)
        else 0
    )
    return (rating, critique_len, len(t.response))


def mine_human_pairs(
    rows: list[Trajectory],
    *,
    sibling_run_id: str,
) -> list[PreferencePair]:
    """Per-prompt: pair the best thumbs-up against the worst thumbs-down.

    Returns at most one ``PreferencePair`` per distinct ``prompt``. Rows
    without a ``human_rating`` are skipped. Ties are broken by ``rating``,
    then critique length, then response length.
    """
    by_prompt: dict[str, list[Trajectory]] = {}
    for r in rows:
        if r.human_rating is None:
            continue
        by_prompt.setdefault(r.prompt, []).append(r)

    timestamp = _utc_now_iso()
    pairs: list[PreferencePair] = []
    for prompt, group in by_prompt.items():
        ups = [r for r in group if r.human_rating and r.human_rating.thumbs == "up"]
        downs = [r for r in group if r.human_rating and r.human_rating.thumbs == "down"]
        if not ups or not downs:
            continue
        chosen = max(ups, key=_human_score)
        rejected = min(downs, key=_human_score)
        if chosen.response == rejected.response:
            continue

        chosen_score = float(chosen.creativity + chosen.logic)
        rejected_score = float(rejected.creativity + rejected.logic)
        pairs.append(
            PreferencePair(
                id=pair_id(prompt, chosen.response, rejected.response),
                timestamp=timestamp,
                domain=chosen.domain,
                prompt=prompt,
                chosen=chosen.response,
                rejected=rejected.response,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                score_gap=chosen_score - rejected_score,
                chosen_temperature=chosen.generator_temperature,
                rejected_temperature=rejected.generator_temperature,
                generator_model=chosen.generator_model,
                evaluator_model=chosen.evaluator_model,
                rubric_version=chosen.rubric_version,
                source="sibling_sampling",
                sibling_run_id=sibling_run_id,
            )
        )
    return pairs
