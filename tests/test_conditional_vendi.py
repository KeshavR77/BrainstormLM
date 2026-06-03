"""Tests for Conditional-Vendi — within-prompt diversity. Stubbed embeddings.

conditional_vendi reuses diversity.vendi_score, which calls diversity._embed
once per group of length >= 2 (length 0 -> 0.0, length 1 -> 1.0 short-circuit
without embedding). Tests patch diversity._embed accordingly.
"""

from __future__ import annotations

import numpy as np

from brainstormlm.eval.conditional_vendi import conditional_vendi

_ORTHO = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
_COLLAPSED = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)


def _patch_embed(monkeypatch, fn) -> None:
    import brainstormlm.eval.diversity as div

    monkeypatch.setattr(div, "_embed", fn)


def test_keys_present(monkeypatch) -> None:
    _patch_embed(monkeypatch, lambda texts: _ORTHO[: len(texts)])
    result = conditional_vendi([["a", "b"]])
    assert set(result.keys()) == {"mean_within_vendi", "per_prompt", "n_prompts"}


def test_diverse_groups_score_high(monkeypatch) -> None:
    _patch_embed(monkeypatch, lambda texts: _ORTHO[: len(texts)])
    result = conditional_vendi([["a", "b"], ["c", "d"]])
    assert abs(result["mean_within_vendi"] - 2.0) < 0.05
    assert result["n_prompts"] == 2


def test_collapsed_groups_score_one(monkeypatch) -> None:
    _patch_embed(monkeypatch, lambda texts: _COLLAPSED[: len(texts)])
    result = conditional_vendi([["a", "b"], ["c", "d"]])
    assert abs(result["mean_within_vendi"] - 1.0) < 0.05


def test_diverse_beats_collapsed(monkeypatch) -> None:
    batches = [_ORTHO, _COLLAPSED]
    calls = {"i": 0}

    def fake(texts):
        out = batches[calls["i"]]
        calls["i"] += 1
        return out[: len(texts)]

    _patch_embed(monkeypatch, fake)
    result = conditional_vendi([["a", "b"], ["c", "d"]])
    assert result["per_prompt"][0] > result["per_prompt"][1]
    assert abs(result["mean_within_vendi"] - 1.5) < 0.05


def test_empty_input() -> None:
    assert conditional_vendi([]) == {
        "mean_within_vendi": 0.0,
        "per_prompt": [],
        "n_prompts": 0,
    }


def test_singleton_group_is_bounded_at_one() -> None:
    # vendi_score(["only"]) short-circuits to 1.0 with no embedding call.
    result = conditional_vendi([["only"]])
    assert result["per_prompt"][0] == 1.0
    assert result["mean_within_vendi"] == 1.0
