"""Routing tests for make_generator across the (adapter × steering) matrix."""

from __future__ import annotations

from pathlib import Path

from brainstormlm.model import (
    HFGenerator,
    SteeredGenerator,
    TogetherGenerator,
    make_generator,
)


def _fake_steering_dir(tmp_path: Path) -> Path:
    out = tmp_path / "fake-steering"
    out.mkdir()
    return out


def test_no_adapter_no_steering_returns_together() -> None:
    gen = make_generator(None)
    assert isinstance(gen, TogetherGenerator)


def test_adapter_only_returns_hf(tmp_path: Path) -> None:
    gen = make_generator(tmp_path / "adapter")
    assert isinstance(gen, HFGenerator)
    assert not isinstance(gen, SteeredGenerator)


def test_steering_only_returns_steered(tmp_path: Path) -> None:
    gen = make_generator(None, steering_path=_fake_steering_dir(tmp_path))
    assert isinstance(gen, SteeredGenerator)
    assert gen.adapter_path is None


def test_steering_plus_adapter_returns_steered_with_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    steering = _fake_steering_dir(tmp_path)
    gen = make_generator(adapter, steering_path=steering)
    assert isinstance(gen, SteeredGenerator)
    assert gen.adapter_path == adapter
    assert gen.steering_path == steering


def test_strength_default_falls_through_settings(tmp_path: Path) -> None:
    gen = make_generator(None, steering_path=_fake_steering_dir(tmp_path))
    assert gen.steering_strength == 1.0  # matches default_steering_strength


def test_strength_override_propagates(tmp_path: Path) -> None:
    gen = make_generator(
        None,
        steering_path=_fake_steering_dir(tmp_path),
        steering_strength=2.5,
    )
    assert gen.steering_strength == 2.5
