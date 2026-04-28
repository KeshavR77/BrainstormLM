"""A/B routing tests (plan-06)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from brainstormlm.api.routing import (
    ABConfig,
    ABResolution,
    load_ab_config,
    resolve_arm,
)


def test_resolve_arm_is_deterministic():
    cfg = ABConfig(
        adapter_a=Path("data/adapters/a"),
        adapter_b=Path("data/adapters/b"),
        traffic_a_pct=50,
    )
    sid = "session-keshav"
    first = resolve_arm(sid, cfg)
    for _ in range(99):
        assert resolve_arm(sid, cfg) == first


def test_zero_pct_routes_all_to_b():
    cfg = ABConfig(
        adapter_a=Path("/a"),
        adapter_b=Path("/b"),
        traffic_a_pct=0,
    )
    for _ in range(50):
        assert resolve_arm(uuid.uuid4().hex, cfg).arm == "b"


def test_full_pct_routes_all_to_a():
    cfg = ABConfig(
        adapter_a=Path("/a"),
        adapter_b=Path("/b"),
        traffic_a_pct=100,
    )
    for _ in range(50):
        assert resolve_arm(uuid.uuid4().hex, cfg).arm == "a"


def test_only_a_configured_degrades_to_single_arm():
    cfg = ABConfig(adapter_a=Path("/a"), traffic_a_pct=10)
    for _ in range(20):
        res = resolve_arm(uuid.uuid4().hex, cfg)
        assert res.arm == "a"
        assert res.adapter_path == Path("/a")


def test_only_b_configured_degrades_to_single_arm():
    cfg = ABConfig(adapter_b=Path("/b"), traffic_a_pct=90)
    for _ in range(20):
        res = resolve_arm(uuid.uuid4().hex, cfg)
        assert res.arm == "b"
        assert res.adapter_path == Path("/b")


def test_empty_config_routes_to_base_generator():
    cfg = ABConfig()
    res = resolve_arm("anyone", cfg)
    assert res.adapter_path is None
    assert res.steering_path is None


def test_steering_paths_thread_through_resolution():
    cfg = ABConfig(
        adapter_a=Path("/a"),
        steering_a=Path("/s/a"),
        adapter_b=Path("/b"),
        steering_b=Path("/s/b"),
        traffic_a_pct=100,
    )
    res = resolve_arm("foo", cfg)
    assert res.arm == "a"
    assert res.adapter_path == Path("/a")
    assert res.steering_path == Path("/s/a")


def test_distribution_within_tolerance():
    cfg = ABConfig(
        adapter_a=Path("/a"),
        adapter_b=Path("/b"),
        traffic_a_pct=30,
    )
    a_count = 0
    n = 1000
    for _ in range(n):
        if resolve_arm(uuid.uuid4().hex, cfg).arm == "a":
            a_count += 1
    fraction = a_count / n
    assert 0.22 <= fraction <= 0.38, f"got {fraction}"


def test_load_ab_config_missing_file_returns_default(tmp_path):
    cfg = load_ab_config(tmp_path / "nonexistent.json")
    assert isinstance(cfg, ABConfig)
    assert cfg.adapter_a is None
    assert cfg.adapter_b is None


def test_load_ab_config_round_trip(tmp_path):
    payload = {
        "adapter_a": "/path/to/a",
        "adapter_b": "/path/to/b",
        "traffic_a_pct": 70,
    }
    cfg_path = tmp_path / "ab_config.json"
    cfg_path.write_text(json.dumps(payload))
    cfg = load_ab_config(cfg_path)
    assert cfg.adapter_a == Path("/path/to/a")
    assert cfg.adapter_b == Path("/path/to/b")
    assert cfg.traffic_a_pct == 70


def test_resolution_dataclass_shape():
    res = ABResolution(arm="a", adapter_path=None, steering_path=None)
    assert res.arm == "a"
