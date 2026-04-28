"""Pure-Python tests for SteeringVector + derive_from_pairs.

No GPU, no network: a hand-built stub model returns deterministic per-layer
hidden states keyed by the input text so the math is fully predictable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainstormlm.model.steering import (
    DERIVATION_VERSION,
    SteeringVector,
    derive_from_pairs,
    render_contrast_pair,
)


# ---------------------------------------------------------------------------
# Stub model + tokenizer
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Pass-through tokenizer that maps each input text to a unique 1×3 tensor."""

    def __init__(self) -> None:
        self._next_id = 100

    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        # Encode unique-per-string id via Python's hash (stable within process).
        token = abs(hash(text)) % 1000 + 1
        return {
            "input_ids": torch.tensor([[token, token + 1, token + 2]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        # Joins role:content pairs so creative vs generic rendering differs.
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class _StubOutputs:
    def __init__(self, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.hidden_states = hidden_states


class _StubModel:
    """Deterministic per-layer hidden states keyed off the input text.

    Layer 0 returns embedding-like states; subsequent layers add layer-indexed
    bias so different layers give different last-token vectors. The "creative"
    vs "generic" distinction is encoded by adding a per-text seed.
    """

    def __init__(self, *, n_layers: int = 4, hidden_dim: int = 8) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.calls: list[torch.Tensor] = []

    def __call__(self, *, input_ids, attention_mask=None, output_hidden_states):
        assert output_hidden_states is True
        self.calls.append(input_ids)
        seed = float(input_ids[0, 0].item())
        # n_layers + 1 hidden states (index 0 is "embedding", indices 1..n_layers are blocks).
        states = []
        for layer in range(self.n_layers + 1):
            row = torch.arange(self.hidden_dim, dtype=torch.float32) + seed + layer * 0.5
            # batch=1, seq=3, hidden=hidden_dim — all positions identical except last.
            states.append(row.unsqueeze(0).unsqueeze(0).expand(1, 3, self.hidden_dim).clone())
            # Force the last token to have a slightly different signature so
            # "last token only" extraction is observable.
            states[-1][:, -1, :] = states[-1][:, -1, :] + 1.0
        return _StubOutputs(tuple(states))


# ---------------------------------------------------------------------------
# Tests — derivation math
# ---------------------------------------------------------------------------


def test_derive_from_pairs_returns_mean_of_last_token_diffs() -> None:
    model = _StubModel(n_layers=4, hidden_dim=8)
    tokenizer = _StubTokenizer()

    pairs = [
        ("creative-A", "generic-A"),
        ("creative-B", "generic-B"),
        ("creative-C", "generic-C"),
    ]
    layers = [1, 3]

    sv = derive_from_pairs(
        pairs,
        base_model_id="stub/qwen-fake",
        layers=layers,
        derived_at="2026-04-27T00:00:00Z",
        model=model,
        tokenizer=tokenizer,
    )

    # Expected: per-pair, per-layer, last-token (creative - generic), averaged.
    expected = {layer: torch.zeros(8) for layer in layers}
    for c_text, g_text in pairs:
        c_seed = float(tokenizer({"text-stub": 0}.get("text-stub", 0))["input_ids"][0, 0].item()) if False else None  # unused
        c_inputs = tokenizer(c_text)
        g_inputs = tokenizer(g_text)
        c_seed = float(c_inputs["input_ids"][0, 0].item())
        g_seed = float(g_inputs["input_ids"][0, 0].item())
        for layer in layers:
            base = torch.arange(8, dtype=torch.float32)
            c_last = base + c_seed + layer * 0.5 + 1.0
            g_last = base + g_seed + layer * 0.5 + 1.0
            expected[layer] += c_last - g_last
    for layer in layers:
        expected[layer] /= len(pairs)

    assert sv.layers == (1, 3)
    assert sv.hidden_dim == 8
    assert sv.n_contrast_pairs == 3
    assert sv.derivation_version == DERIVATION_VERSION
    for layer in layers:
        assert torch.allclose(sv.vectors[layer], expected[layer], atol=1e-6)


def test_derive_dedupes_and_sorts_layers() -> None:
    model = _StubModel(n_layers=4, hidden_dim=4)
    tokenizer = _StubTokenizer()
    sv = derive_from_pairs(
        [("c", "g")],
        base_model_id="stub/x",
        layers=[3, 1, 3],  # duplicate + unsorted
        derived_at="2026-04-27T00:00:00Z",
        model=model,
        tokenizer=tokenizer,
    )
    assert sv.layers == (1, 3)
    assert set(sv.vectors.keys()) == {1, 3}


def test_derive_rejects_empty_inputs() -> None:
    model = _StubModel()
    tokenizer = _StubTokenizer()
    with pytest.raises(ValueError, match="at least one"):
        derive_from_pairs(
            [],
            base_model_id="stub/x",
            layers=[1],
            derived_at="t",
            model=model,
            tokenizer=tokenizer,
        )
    with pytest.raises(ValueError, match="at least one layer"):
        derive_from_pairs(
            [("c", "g")],
            base_model_id="stub/x",
            layers=[],
            derived_at="t",
            model=model,
            tokenizer=tokenizer,
        )


def test_render_contrast_pair_uses_both_framings() -> None:
    tokenizer = _StubTokenizer()
    creative, generic = render_contrast_pair("invent a chair", tokenizer)
    assert "system:" in creative and "system:" in generic
    assert creative != generic
    assert "user:invent a chair" in creative
    assert "user:invent a chair" in generic


# ---------------------------------------------------------------------------
# Tests — persistence round-trip
# ---------------------------------------------------------------------------


def _build_sv() -> SteeringVector:
    return SteeringVector(
        base_model_id="stub/qwen-fake",
        layers=(1, 3),
        hidden_dim=8,
        n_contrast_pairs=4,
        derived_at="2026-04-27T00:00:00Z",
        vectors={
            1: torch.arange(8, dtype=torch.float32),
            3: torch.arange(8, dtype=torch.float32) * 2,
        },
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    sv = _build_sv()
    sv.save(tmp_path / "sv1")

    loaded = SteeringVector.load(tmp_path / "sv1")
    assert loaded.base_model_id == sv.base_model_id
    assert loaded.layers == sv.layers
    assert loaded.hidden_dim == sv.hidden_dim
    assert loaded.n_contrast_pairs == sv.n_contrast_pairs
    assert loaded.derivation_version == sv.derivation_version
    for layer in sv.layers:
        assert torch.allclose(loaded.vectors[layer], sv.vectors[layer])


def test_load_rejects_base_model_mismatch(tmp_path: Path) -> None:
    sv = _build_sv()
    sv.save(tmp_path / "sv2")
    with pytest.raises(ValueError, match="does not match"):
        SteeringVector.load(tmp_path / "sv2", expected_base_model_id="other/model")


def test_load_rejects_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No manifest"):
        SteeringVector.load(tmp_path / "empty")
