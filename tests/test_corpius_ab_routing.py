"""Tests for Corpius-aware A/B routing extensions."""

from __future__ import annotations

from pathlib import Path


from brainstormlm.api.routing import ABConfig, resolve_arm


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
    # session-A hashes to bucket 37 (arm a), session-B to bucket 62 (arm b).
    res_a = resolve_arm("session-A", cfg)
    res_b = resolve_arm("session-B", cfg)
    corpius_res = res_a if res_a.arm == "a" else res_b
    adapter_res = res_a if res_a.arm == "b" else res_b
    assert corpius_res.corpius_config_path is not None
    assert adapter_res.adapter_path is not None
    assert adapter_res.corpius_config_path is None
