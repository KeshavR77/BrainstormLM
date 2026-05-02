"""Tests for CorpiusGenerator — no GPU, no model load, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from brainstormlm import config
from brainstormlm.clients.cache import cache_key, write_cache
from brainstormlm.model.corpius import CorpiusGenerator
from brainstormlm.model.factory import make_generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Create minimal stub artifact files so path checks pass."""
    vectors_dir = tmp_path / "corpius"
    vectors_dir.mkdir()
    creative = vectors_dir / "hyper_creative_purified_vector_L14.pt"
    technical = vectors_dir / "hyper_technical_purified_vector_L18.pt"
    creative.write_bytes(b"stub")
    technical.write_bytes(b"stub")
    config_file = vectors_dir / "corpius_1.json"
    config_file.write_text(
        """{
  "id": "corpius_1",
  "display_name": "Corpius 1",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "benchmark_arm": "qwen_v17",
  "local_variants": "v17",
  "local_arms": "additive",
  "steering_method": "v10_parametric_preset_v17",
  "steering_vectors": {
    "creative": "CREATIVE_PATH",
    "technical": "TECHNICAL_PATH"
  },
  "schedule": {
    "creative_start_alpha": 2.05,
    "exp_zero_by_step": 132,
    "technical_start_alpha": 0.75,
    "sigmoid_mid_step": 58.0,
    "sigmoid_steepness": 0.11,
    "steering_math": "additive"
  }
}""".replace("CREATIVE_PATH", str(creative)).replace("TECHNICAL_PATH", str(technical))
    )
    return {"config": config_file, "creative": creative, "technical": technical}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BRAINSTORMLM_CACHE_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def artifacts(tmp_path: Path) -> dict[str, Path]:
    return _make_artifacts(tmp_path)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_corpius_generator_has_generate_method() -> None:
    gen = CorpiusGenerator.__new__(CorpiusGenerator)
    assert callable(getattr(gen, "generate", None))


def test_corpius_generator_satisfies_generator_protocol() -> None:
    import inspect

    gen = CorpiusGenerator.__new__(CorpiusGenerator)
    sig = inspect.signature(gen.generate)
    params = sig.parameters
    assert "prompt" in params
    assert "system" in params
    assert "temperature" in params
    assert "max_tokens" in params
    assert "use_cache" in params


# ---------------------------------------------------------------------------
# Cache hit skips engine load
# ---------------------------------------------------------------------------


def test_cache_hit_skips_engine_load(
    isolated_cache: Path, artifacts: dict[str, Path]
) -> None:
    gen = CorpiusGenerator(config_path=artifacts["config"])
    key = cache_key(
        "corpius",
        "Qwen/Qwen2.5-3B-Instruct",
        "",
        "brainstorm ideas",
        f"{1.0:.4f}",
        "1024",
    )
    write_cache(key, "cached corpius response")

    with patch.object(gen, "_ensure_loaded") as mock_load:
        text, cached = gen.generate_with_meta("brainstorm ideas")

    assert text == "cached corpius response"
    assert cached is True
    mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# Cache miss triggers engine
# ---------------------------------------------------------------------------


def test_cache_miss_calls_engine_and_writes_cache(
    isolated_cache: Path, artifacts: dict[str, Path]
) -> None:
    gen = CorpiusGenerator(config_path=artifacts["config"])

    fake_engine = MagicMock()
    fake_engine.generate.return_value = "corpius output"

    gen._engine = fake_engine
    gen._spec = MagicMock()
    gen._spec.base_model = "Qwen/Qwen2.5-3B-Instruct"

    with patch.object(gen, "_ensure_loaded", side_effect=lambda: None):
        text, cached = gen.generate_with_meta("new prompt")

    assert text == "corpius output"
    assert cached is False


# ---------------------------------------------------------------------------
# System prompt is threaded through
# ---------------------------------------------------------------------------


def test_system_prompt_passed_to_engine(
    isolated_cache: Path, artifacts: dict[str, Path]
) -> None:
    gen = CorpiusGenerator(config_path=artifacts["config"])

    fake_engine = MagicMock()
    fake_engine.generate.return_value = "response"
    gen._engine = fake_engine
    gen._spec = MagicMock()
    gen._spec.base_model = "Qwen/Qwen2.5-3B-Instruct"

    with patch.object(gen, "_ensure_loaded", side_effect=lambda: None):
        gen.generate("hello", system="think creatively", use_cache=False)

    call_kwargs = fake_engine.generate.call_args
    assert call_kwargs.kwargs.get("system") == "think creatively" or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == "think creatively"
    )


# ---------------------------------------------------------------------------
# factory routing
# ---------------------------------------------------------------------------


def test_make_generator_corpius_flag(artifacts: dict[str, Path]) -> None:
    gen = make_generator(corpius_config_path=artifacts["config"])
    assert isinstance(gen, CorpiusGenerator)


def test_make_generator_corpius_does_not_return_together() -> None:
    from brainstormlm.model.base import TogetherGenerator

    tmp = Path("/tmp/stub_corpius_config_nonexistent.json")
    gen = make_generator(corpius_config_path=tmp)
    assert not isinstance(gen, TogetherGenerator)
