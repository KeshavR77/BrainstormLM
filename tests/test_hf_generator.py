"""Unit tests for HFGenerator + make_generator (no GPU, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brainstormlm import config
from brainstormlm.clients.cache import cache_key, write_cache
from brainstormlm.model import HFGenerator, TogetherGenerator, make_generator


# ---------------------------------------------------------------------------
# Stubs for transformers + peft
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """Minimal mapping that mimics a tokenizer batch encoding."""

    def to(self, _device: str) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.template_calls: list[list[dict[str, str]]] = []
        self.decode_calls: list[Any] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.template_calls.append(list(messages))
        return "RENDERED_PROMPT"

    def __call__(self, text: str, return_tensors: str) -> _FakeBatch:
        assert return_tensors == "pt"
        # Encoded prompt = 3 tokens; output adds 2 new ones below.
        return _FakeBatch(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )

    def decode(self, ids: Any, *, skip_special_tokens: bool) -> str:
        self.decode_calls.append((list(ids), skip_special_tokens))
        return "  stubbed-response  "


class _FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, _device: str) -> "_FakeModel":
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        # Echo prompt (3 tokens) + 2 new tokens.
        return [[1, 2, 3, 42, 43]]


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorchMps:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorchBackends:
    mps = _FakeTorchMps()


class _FakeTorch:
    bfloat16 = "bf16-dtype"
    float16 = "fp16-dtype"
    float32 = "fp32-dtype"
    cuda = _FakeTorchCuda()
    backends = _FakeTorchBackends()


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
def hf_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake torch / transformers / peft modules visible inside hf.py.

    Tracks how often each constructor is called so individual tests can spy.
    """
    state: dict[str, Any] = {
        "tokenizer_calls": 0,
        "model_calls": 0,
        "peft_calls": 0,
        "tokenizers": [],
        "models": [],
        "peft_args": [],
    }

    def fake_tokenizer_from_pretrained(model_id: str, *args: Any, **kwargs: Any) -> _FakeTokenizer:
        state["tokenizer_calls"] += 1
        tok = _FakeTokenizer()
        state["tokenizers"].append(tok)
        return tok

    def fake_model_from_pretrained(model_id: str, *args: Any, **kwargs: Any) -> _FakeModel:
        state["model_calls"] += 1
        m = _FakeModel()
        state["models"].append(m)
        return m

    def fake_peft_from_pretrained(model: Any, adapter_path: str, *args: Any, **kwargs: Any) -> Any:
        state["peft_calls"] += 1
        state["peft_args"].append((model, adapter_path))
        return model

    # Patch where the names are looked up. hf.py does
    # `from transformers import AutoModelForCausalLM, AutoTokenizer` inside
    # `_ensure_loaded`, so monkeypatching the source module's attributes
    # works because the import binds the patched function at lookup time.
    import sys
    import types

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

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = _FakeTorch.bfloat16
    fake_torch.float16 = _FakeTorch.float16
    fake_torch.float32 = _FakeTorch.float32
    fake_torch.cuda = _FakeTorch.cuda
    fake_torch.backends = _FakeTorch.backends
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cache_hit_skips_model_load(isolated_cache: Path, hf_stubs: dict[str, Any]) -> None:
    settings = config.get_settings()
    base_model = settings.hf_base_model
    key = cache_key(
        "hf",
        base_model,
        "no-adapter",
        "",
        "hello",
        f"{1.0:.4f}",
        "1024",
    )
    write_cache(key, "cached response")

    gen = HFGenerator()
    text, cached = gen.generate_with_meta("hello")

    assert text == "cached response"
    assert cached is True
    assert hf_stubs["model_calls"] == 0
    assert hf_stubs["tokenizer_calls"] == 0


def test_first_call_loads_model_and_writes_cache(
    isolated_cache: Path, hf_stubs: dict[str, Any]
) -> None:
    gen = HFGenerator()
    text, cached = gen.generate_with_meta("hello", system="be helpful")

    assert text == "stubbed-response"  # whitespace stripped
    assert cached is False
    assert hf_stubs["model_calls"] == 1
    assert hf_stubs["tokenizer_calls"] == 1

    # Second call with same args is cache-served and does not reload.
    text2, cached2 = gen.generate_with_meta("hello", system="be helpful")
    assert text2 == "stubbed-response"
    assert cached2 is True
    assert hf_stubs["model_calls"] == 1


def test_second_call_reuses_loaded_model(
    isolated_cache: Path, hf_stubs: dict[str, Any]
) -> None:
    gen = HFGenerator()
    gen.generate_with_meta("first prompt")
    gen.generate_with_meta("different prompt")

    # Different prompts → both miss cache → both call model.generate, but the
    # heavy `from_pretrained` should have run only once.
    assert hf_stubs["model_calls"] == 1
    assert hf_stubs["tokenizer_calls"] == 1
    assert len(hf_stubs["models"]) == 1
    assert len(hf_stubs["models"][0].generate_calls) == 2


def test_adapter_path_triggers_peft_load(
    isolated_cache: Path, hf_stubs: dict[str, Any], tmp_path: Path
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    gen = HFGenerator(adapter_path=adapter_dir)
    gen.generate_with_meta("hello")

    assert hf_stubs["peft_calls"] == 1
    _model_arg, adapter_arg = hf_stubs["peft_args"][0]
    assert adapter_arg == str(adapter_dir)

    # Second call reuses everything — peft is not re-applied.
    gen.generate_with_meta("hello again")
    assert hf_stubs["peft_calls"] == 1


def test_no_adapter_skips_peft_import(
    isolated_cache: Path, hf_stubs: dict[str, Any]
) -> None:
    # Make peft.PeftModel.from_pretrained explode if invoked.
    import sys

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("peft must not be touched without adapter_path")

    sys.modules["peft"].PeftModel.from_pretrained = _boom  # type: ignore[attr-defined]

    gen = HFGenerator()
    text, cached = gen.generate_with_meta("hi")
    assert text == "stubbed-response"
    assert cached is False
    assert hf_stubs["peft_calls"] == 0


def test_make_generator_no_adapter_returns_together() -> None:
    gen = make_generator(None)
    assert isinstance(gen, TogetherGenerator)


def test_make_generator_with_adapter_returns_hf(tmp_path: Path) -> None:
    gen = make_generator(tmp_path / "fake-adapter")
    assert isinstance(gen, HFGenerator)
