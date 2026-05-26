"""Per-pool sampling presets: creative-pool sampling vs greedy neutral pool.

The contrast is now driven by *decoding* as well as framing — the creative pool is
generated with high-temperature nucleus sampling (a min-p floor for coherence), the
neutral pool with pure greedy decoding. These tests pin the preset values, the
HF decode-kwarg construction, and the per-framing selection in ``gather_responses``.
"""

from __future__ import annotations

import numpy as np

from creativity_steering.config import (
    CAPTURE_LAST_RESPONSE,
    CREATIVE_SAMPLING,
    GREEDY_SAMPLING,
    GenerationConfig,
    SamplingPreset,
    load_config,
)
from creativity_steering.derivation.generate import gather_responses
from creativity_steering.derivation.prompts import PromptSet
from creativity_steering.models.backend import ModelBackend
from creativity_steering.models.hf import build_generate_kwargs


def test_preset_do_sample_and_named_presets():
    # Greedy means do_sample=False; sampling means do_sample=True.
    assert GREEDY_SAMPLING.do_sample is False
    assert GREEDY_SAMPLING.temperature == 0.0
    assert CREATIVE_SAMPLING.do_sample is True
    # The agreed creative preset.
    assert CREATIVE_SAMPLING.temperature == 1.2
    assert CREATIVE_SAMPLING.top_p == 0.95
    assert CREATIVE_SAMPLING.top_k == 0
    assert CREATIVE_SAMPLING.min_p == 0.02
    assert CREATIVE_SAMPLING.repetition_penalty == 1.1


def test_legacy_default_preset_forwards_only_temperature():
    # A bare preset (legacy-equivalent) must NOT forward top_p/top_k/min_p/rep, so the
    # model's own generation defaults still apply — preserving current behaviour.
    legacy = SamplingPreset(temperature=0.9)
    kwargs = build_generate_kwargs(legacy, max_new_tokens=64, pad_token_id=7)
    assert kwargs["do_sample"] is True
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["pad_token_id"] == 7
    assert abs(kwargs["temperature"] - 0.9) < 1e-9
    for k in ("top_p", "top_k", "min_p", "repetition_penalty"):
        assert k not in kwargs


def test_greedy_preset_disables_sampling_knobs():
    kwargs = build_generate_kwargs(GREEDY_SAMPLING, max_new_tokens=32, pad_token_id=0)
    assert kwargs["do_sample"] is False
    for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        assert k not in kwargs


def test_creative_preset_forwards_active_knobs_only():
    kwargs = build_generate_kwargs(CREATIVE_SAMPLING, max_new_tokens=128, pad_token_id=1)
    assert kwargs["do_sample"] is True
    assert abs(kwargs["temperature"] - 1.2) < 1e-9
    assert kwargs["top_p"] == 0.95
    assert kwargs["min_p"] == 0.02
    assert kwargs["repetition_penalty"] == 1.1
    # top_k == 0 means "disabled" → not forwarded (top_p does the truncation).
    assert "top_k" not in kwargs


class _RecordingBackend(ModelBackend):
    """Records the sampling preset passed to each generate call, keyed by system prompt."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, SamplingPreset | None]] = []

    @property
    def model_id(self) -> str:
        return "recording"

    @property
    def num_layers(self) -> int:
        return 4

    @property
    def hidden_dim(self) -> int:
        return 8

    def generate(self, prompt, *, system=None, max_new_tokens=256, temperature=0.9, seed=0,
                 sampling=None) -> str:
        self.calls.append((system, sampling))
        return "a response"

    def capture_activations(self, prompt, response, *, system=None, positions):
        shape = (self.num_layers, self.hidden_dim)
        return {p: np.zeros(shape, dtype=np.float32) for p in positions}

    def generate_steered(self, prompt, *, injections, system=None, max_new_tokens=256,
                         temperature=0.9, seed=0) -> str:
        return ""


def test_gather_responses_selects_preset_by_framing():
    backend = _RecordingBackend()
    ps = PromptSet(creative_system="CREATIVE", neutral_system="NEUTRAL", topics=("t1", "t2"))

    gather_responses(
        backend,
        ps,
        positions=[CAPTURE_LAST_RESPONSE],
        samples_per_prompt=2,
        sampling={"creative": CREATIVE_SAMPLING, "neutral": GREEDY_SAMPLING},
    )

    creative = [s for sys, s in backend.calls if sys == "CREATIVE"]
    neutral = [s for sys, s in backend.calls if sys == "NEUTRAL"]
    assert creative and all(s == CREATIVE_SAMPLING for s in creative)
    assert neutral and all(s == GREEDY_SAMPLING for s in neutral)


def test_generation_config_defaults_and_mapping():
    g = GenerationConfig()
    assert g.creative_sampling == CREATIVE_SAMPLING
    assert g.neutral_sampling == GREEDY_SAMPLING
    mapping = g.sampling_by_framing()
    assert mapping == {"creative": CREATIVE_SAMPLING, "neutral": GREEDY_SAMPLING}


def test_config_roundtrip_preserves_presets(tmp_path):
    from creativity_steering.config import ExperimentConfig

    cfg = ExperimentConfig(
        backend="mock",
        generation=GenerationConfig(
            creative_sampling=SamplingPreset(temperature=1.4, top_p=0.9, min_p=0.05),
            neutral_sampling=GREEDY_SAMPLING,
        ),
    )
    cfg.save(tmp_path / "c.json")
    loaded = load_config(tmp_path / "c.json")
    expected = SamplingPreset(temperature=1.4, top_p=0.9, min_p=0.05)
    assert loaded.generation.creative_sampling == expected
    assert loaded.generation.neutral_sampling == GREEDY_SAMPLING
