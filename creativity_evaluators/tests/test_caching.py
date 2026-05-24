"""Tests for the disk cache used by the reward model and (expensive) judge calls."""

from __future__ import annotations

from creativity_evaluators.utils.caching import DiskCache


def test_set_get_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    c.set("k", {"a": 1, "b": [1, 2, 3]})
    assert c.get("k") == {"a": 1, "b": [1, 2, 3]}


def test_missing_key_returns_none(tmp_path):
    assert DiskCache(tmp_path).get("absent") is None


def test_make_key_is_stable_and_order_sensitive():
    assert DiskCache.make_key("a", "b") == DiskCache.make_key("a", "b")
    assert DiskCache.make_key("a", "b") != DiskCache.make_key("b", "a")


def test_persists_across_instances(tmp_path):
    DiskCache(tmp_path).set("x", 42)
    assert DiskCache(tmp_path).get("x") == 42
