"""Tests for MAUVE wrapper. Network-free — stub the mauve library."""

from __future__ import annotations

import sys
import types

from brainstormlm.eval.mauve_metric import mauve_score


def test_mauve_score_too_few_returns_zero() -> None:
    """With <2 texts on either side, MAUVE is undefined; return 0.0."""
    assert mauve_score(["one"], ["a", "b"]) == 0.0
    assert mauve_score(["a", "b"], ["one"]) == 0.0
    assert mauve_score([], []) == 0.0


def test_mauve_score_calls_library_and_returns_float(monkeypatch) -> None:
    """Wrapper passes texts through to mauve.compute_mauve and returns .mauve."""
    captured: dict = {}

    def fake_compute_mauve(*, p_text, q_text, **kwargs):
        captured["p"] = p_text
        captured["q"] = q_text
        captured["kwargs"] = kwargs
        result = types.SimpleNamespace(mauve=0.87)
        return result

    fake_mauve = types.SimpleNamespace(compute_mauve=fake_compute_mauve)
    monkeypatch.setitem(sys.modules, "mauve", fake_mauve)

    score = mauve_score(["model a", "model b", "model c"], ["ref 1", "ref 2"])
    assert score == 0.87
    assert captured["p"] == ["model a", "model b", "model c"]
    assert captured["q"] == ["ref 1", "ref 2"]


def test_mauve_score_default_kwargs(monkeypatch) -> None:
    """Ensure default featurizer / max length / device_id are forwarded."""
    captured: dict = {}

    def fake_compute_mauve(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(mauve=0.5)

    fake_mauve = types.SimpleNamespace(compute_mauve=fake_compute_mauve)
    monkeypatch.setitem(sys.modules, "mauve", fake_mauve)

    mauve_score(["a", "b"], ["x", "y"])
    assert captured["device_id"] == -1
    assert captured["max_text_length"] == 256
    assert captured["featurize_model_name"] == "gpt2-large"
