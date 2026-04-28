"""Unit tests for SteeredGenerator (no GPU, no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from brainstormlm import config
from brainstormlm.clients.cache import cache_key, write_cache
from brainstormlm.model.steered import SteeredGenerator
from brainstormlm.model.steering import SteeringVector


# ---------------------------------------------------------------------------
# Fakes — modelled on tests/test_hf_generator.py but with a `.model.layers`
# topology so the steering hooks can actually attach.
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    def to(self, _device: str) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "RENDERED"

    def __call__(self, text: str, return_tensors: str) -> _FakeBatch:
        assert return_tensors == "pt"
        return _FakeBatch(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])

    def decode(self, ids: Any, *, skip_special_tokens: bool) -> str:
        return "  steered-response  "


class _FakeBlock:
    def __init__(self) -> None:
        self.hooks: list[Any] = []
        self.removed: list[bool] = []

    def register_forward_hook(self, fn: Any) -> Any:
        self.hooks.append(fn)
        idx = len(self.removed)
        self.removed.append(False)
        block = self

        class _Handle:
            def remove(self_inner) -> None:
                block.removed[idx] = True

        return _Handle()


class _FakeInner:
    def __init__(self, n_layers: int) -> None:
        self.layers = [_FakeBlock() for _ in range(n_layers)]


class _FakeModel:
    device = "cpu"

    def __init__(self, n_layers: int = 4) -> None:
        self.model = _FakeInner(n_layers)
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, _device: str) -> "_FakeModel":
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        return [[1, 2, 3, 99, 100]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAINSTORMLM_CACHE_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def hf_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake torch / transformers / peft inside hf.py + steered.py."""
    state: dict[str, Any] = {
        "tokenizer_calls": 0,
        "model_calls": 0,
        "peft_calls": 0,
        "tokenizers": [],
        "models": [],
    }

    def fake_tokenizer_from_pretrained(model_id: str, *args, **kwargs) -> _FakeTokenizer:
        state["tokenizer_calls"] += 1
        tok = _FakeTokenizer()
        state["tokenizers"].append(tok)
        return tok

    def fake_model_from_pretrained(model_id: str, *args, **kwargs) -> _FakeModel:
        state["model_calls"] += 1
        m = _FakeModel(n_layers=4)
        state["models"].append(m)
        return m

    def fake_peft_from_pretrained(model: Any, adapter_path: str, *args, **kwargs):
        state["peft_calls"] += 1
        return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=fake_tokenizer_from_pretrained
    )
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=fake_model_from_pretrained
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = types.SimpleNamespace(from_pretrained=fake_peft_from_pretrained)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    # Use the real torch — SteeringVector tensors need it.
    return state


@pytest.fixture
def steering_artifact(tmp_path: Path) -> Path:
    settings = config.get_settings()
    sv = SteeringVector(
        base_model_id=settings.hf_base_model,
        layers=(1, 2),
        hidden_dim=4,
        n_contrast_pairs=8,
        derived_at="2026-04-27T00:00:00Z",
        vectors={
            1: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            2: torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
    )
    out = tmp_path / "sv"
    sv.save(out)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cache_hit_skips_model_load_and_hook_install(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    settings = config.get_settings()
    base_model = settings.hf_base_model
    key = cache_key(
        "hf-steered",
        base_model,
        "no-adapter",
        str(steering_artifact),
        f"{1.0:.4f}",
        "manifest",
        "rescale",
        "",
        "hello",
        f"{1.0:.4f}",
        "1024",
    )
    write_cache(key, "served-from-cache")

    gen = SteeredGenerator(steering_path=steering_artifact)
    text, cached = gen.generate_with_meta("hello")

    assert text == "served-from-cache"
    assert cached is True
    assert hf_stubs["model_calls"] == 0
    assert hf_stubs["tokenizer_calls"] == 0


def test_first_call_loads_model_and_registers_hooks_then_removes(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    gen = SteeredGenerator(steering_path=steering_artifact)
    text, cached = gen.generate_with_meta("hello")

    assert text == "steered-response"
    assert cached is False
    assert hf_stubs["model_calls"] == 1

    fake_model: _FakeModel = hf_stubs["models"][0]
    # Vector covers layers (1, 2); only those blocks should have been hooked.
    assert len(fake_model.model.layers[0].hooks) == 0
    assert len(fake_model.model.layers[1].hooks) == 1
    assert len(fake_model.model.layers[2].hooks) == 1
    assert len(fake_model.model.layers[3].hooks) == 0
    # Every handle was removed in the finally block.
    assert all(fake_model.model.layers[1].removed)
    assert all(fake_model.model.layers[2].removed)


def test_second_call_reuses_loaded_model_and_re_registers(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    gen = SteeredGenerator(steering_path=steering_artifact)
    gen.generate_with_meta("first")
    gen.generate_with_meta("second")

    fake_model: _FakeModel = hf_stubs["models"][0]
    # `from_pretrained` ran once.
    assert hf_stubs["model_calls"] == 1
    # Hooks were registered on each call; both removals tracked.
    assert len(fake_model.model.layers[1].hooks) == 2
    assert len(fake_model.model.layers[2].hooks) == 2
    assert all(fake_model.model.layers[1].removed)
    assert all(fake_model.model.layers[2].removed)


def test_cache_key_differs_from_unsteered(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    """A cache value written for `HFGenerator` must not be served by SteeredGenerator."""
    settings = config.get_settings()
    base_model = settings.hf_base_model
    hf_key = cache_key(
        "hf",
        base_model,
        "no-adapter",
        "",
        "hello",
        f"{1.0:.4f}",
        "1024",
    )
    write_cache(hf_key, "WRONG-IMPL-LEAK")

    gen = SteeredGenerator(steering_path=steering_artifact)
    text, cached = gen.generate_with_meta("hello")
    assert text == "steered-response"
    assert cached is False


def test_steering_layer_override_validates_against_artifact(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    gen = SteeredGenerator(
        steering_path=steering_artifact,
        steering_layers=[1, 99],  # 99 is not in the artifact
    )
    with pytest.raises(ValueError, match="not present"):
        gen.generate_with_meta("hello")


def test_steering_strength_change_busts_cache(
    isolated_cache: Path, hf_stubs: dict[str, Any], steering_artifact: Path
) -> None:
    gen_a = SteeredGenerator(steering_path=steering_artifact, steering_strength=1.0)
    gen_a.generate_with_meta("hi")  # writes cache for strength=1.0

    gen_b = SteeredGenerator(steering_path=steering_artifact, steering_strength=2.0)
    _, cached_b = gen_b.generate_with_meta("hi")
    assert cached_b is False  # different strength → different key


# ---------------------------------------------------------------------------
# Hook math (no model load) — exercise _make_residual_hook directly.
# ---------------------------------------------------------------------------


def test_residual_hook_norm_rescaled_addition() -> None:
    from brainstormlm.model.steered import _make_residual_hook

    v = torch.tensor([1.0, 0.0, 0.0])
    hook = _make_residual_hook(v, strength=1.0, rescale=True)
    hidden = torch.tensor([[[2.0, 0.0, 0.0]]])  # (1, 1, 3), norm=2
    result = hook(None, None, hidden)

    # rescale: scaled = (v / ||v||) * ||h|| * strength = [1, 0, 0] * 2 = [2, 0, 0]
    expected = hidden + torch.tensor([[[2.0, 0.0, 0.0]]])
    assert torch.allclose(result, expected)


def test_residual_hook_no_rescale() -> None:
    from brainstormlm.model.steered import _make_residual_hook

    v = torch.tensor([0.5, 0.0, 0.0])
    hook = _make_residual_hook(v, strength=2.0, rescale=False)
    hidden = torch.tensor([[[3.0, 1.0, 1.0]]])
    result = hook(None, None, hidden)

    expected = hidden + v * 2.0
    assert torch.allclose(result, expected)


def test_residual_hook_handles_tuple_output() -> None:
    from brainstormlm.model.steered import _make_residual_hook

    v = torch.tensor([0.0, 1.0, 0.0])
    hook = _make_residual_hook(v, strength=1.0, rescale=False)
    hidden = torch.tensor([[[1.0, 1.0, 1.0]]])
    extra = ("present_key_value",)
    result = hook(None, None, (hidden, *extra))

    assert isinstance(result, tuple)
    assert torch.allclose(result[0], hidden + v)
    assert result[1] == "present_key_value"
