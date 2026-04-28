"""Deterministic per-session A/B routing between two adapter/steering arms (plan-06)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ABConfig(BaseModel):
    """Two-arm config persisted at ``settings.ab_config_path``."""

    model_config = ConfigDict(extra="forbid")

    adapter_a: Path | None = None
    adapter_b: Path | None = None
    steering_a: Path | None = None
    steering_b: Path | None = None
    traffic_a_pct: int = Field(default=50, ge=0, le=100)


class ABResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: Literal["a", "b"]
    adapter_path: Path | None = None
    steering_path: Path | None = None


def _both_arms_present(cfg: ABConfig) -> bool:
    a_set = cfg.adapter_a is not None or cfg.steering_a is not None
    b_set = cfg.adapter_b is not None or cfg.steering_b is not None
    return a_set and b_set


def _hash_pct(session_id: str) -> int:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def resolve_arm(session_id: str, cfg: ABConfig) -> ABResolution:
    """Pick an arm for ``session_id`` against ``cfg``.

    Stable per session: same id → same arm until ``cfg`` changes.
    Degrades to whichever arm is fully configured when only one is set.
    """
    a_set = cfg.adapter_a is not None or cfg.steering_a is not None
    b_set = cfg.adapter_b is not None or cfg.steering_b is not None

    if not a_set and not b_set:
        # Fall back to the base generator (no adapter, no steering).
        # Pick arm "a" by convention for log/metric coherence.
        return ABResolution(arm="a", adapter_path=None, steering_path=None)

    if a_set and not b_set:
        return ABResolution(arm="a", adapter_path=cfg.adapter_a, steering_path=cfg.steering_a)
    if b_set and not a_set:
        return ABResolution(arm="b", adapter_path=cfg.adapter_b, steering_path=cfg.steering_b)

    bucket = _hash_pct(session_id)
    if bucket < cfg.traffic_a_pct:
        return ABResolution(arm="a", adapter_path=cfg.adapter_a, steering_path=cfg.steering_a)
    return ABResolution(arm="b", adapter_path=cfg.adapter_b, steering_path=cfg.steering_b)


def load_ab_config(path: Path | None = None) -> ABConfig:
    """Read ``ABConfig`` from JSON; return defaults when the file is missing."""
    if path is None or not path.exists():
        return ABConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ABConfig.model_validate(payload)
