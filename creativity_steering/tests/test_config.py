from pathlib import Path

import pytest

from creativity_steering.config import ExperimentConfig, load_config
from creativity_steering.models.factory import build_backend
from creativity_steering.models.mock import MockBackend

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_save_and_load_roundtrip(tmp_path):
    cfg = ExperimentConfig(backend="mock", num_layers=8, hidden_dim=16)
    cfg.save(tmp_path / "c.json")
    loaded = load_config(tmp_path / "c.json")
    assert loaded.backend == "mock"
    assert loaded.num_layers == 8
    assert loaded.patching.alphas == cfg.patching.alphas
    assert loaded.generation.samples_per_prompt == cfg.generation.samples_per_prompt


def test_shipped_configs_parse():
    for name in ["local_mock.json", "qwen_0_5b_local.json", "runpod_qwen14b.json"]:
        cfg = load_config(CONFIGS / name)
        assert cfg.backend in {"mock", "hf"}
        assert cfg.capture_position in cfg.all_capture_positions


def test_invalid_capture_position_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(capture_position="nonsense")


def test_factory_builds_mock_backend():
    cfg = load_config(CONFIGS / "local_mock.json")
    backend = build_backend(cfg)
    assert isinstance(backend, MockBackend)
    assert backend.num_layers == 24


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_backend(ExperimentConfig(backend="bogus"))
