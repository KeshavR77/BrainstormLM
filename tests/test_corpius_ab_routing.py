"""Tests for Corpius-aware A/B routing extensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from brainstormlm.api.routing import ABConfig, ABResolution, resolve_arm
from brainstormlm.model.corpius import CorpiusGenerator


def test_corpius_arm_a_resolved() -> None:
    cfg = ABConfig(corpius_a=Path("/tmp/corpius_1.json"), traffic_a_pct=100)
    res = resolve_arm("any-session", cfg)
    assert res.arm == "a"
    assert res.corpius_config_path == Path("/tmp/corpius_1.json")
    assert res.adapter_path is None
    assert res.steering_path is None


def test_corpius_arm_b_resolved() -> None:
    cfg = ABConfig(corpius_b=Path("/tmp/corpius_1.json"), traffic_a_pct=0)
    res = resolve_arm("any-session", cfg)
    assert res.arm == "b"
    assert res.corpius_config_path == Path("/tmp/corpius_1.json")


def test_no_arms_set_gives_no_corpius() -> None:
    cfg = ABConfig()
    res = resolve_arm("any-session", cfg)
    assert res.corpius_config_path is None


def test_corpius_a_vs_adapter_b_routing() -> None:
    cfg = ABConfig(
        corpius_a=Path("/tmp/corpius_1.json"),
        adapter_b=Path("/tmp/adapter"),
        traffic_a_pct=50,
    )
    # Bucket 0..49 → arm a (Corpius), 50..99 → arm b (adapter)
    # Just confirm both arms are resolvable.
    res_a = resolve_arm("session-0000", cfg)  # deterministic but we check type
    res_b = resolve_arm("session-9999", cfg)
    corpius_res = res_a if res_a.arm == "a" else res_b
    adapter_res = res_a if res_a.arm == "b" else res_b
    assert corpius_res.corpius_config_path is not None
    assert adapter_res.adapter_path is not None
    assert adapter_res.corpius_config_path is None
