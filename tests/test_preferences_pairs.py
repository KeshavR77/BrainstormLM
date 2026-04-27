"""Pure-logic tests for sibling sampling + pair construction.

No network, no real generator, no real evaluator. Stubs everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from brainstormlm.clients.openai import Usage
from brainstormlm.feedback import evaluator as evaluator_mod
from brainstormlm.training.preferences import (
    PreferencePair,
    SiblingSample,
    mine_pairs,
    sample_and_score_siblings,
)


def _sib(
    response: str,
    *,
    creativity: int,
    logic: int,
    temperature: float = 1.0,
) -> SiblingSample:
    return SiblingSample(
        response=response,
        temperature=temperature,
        creativity=creativity,
        logic=logic,
        rationale="stub",
        failure_modes=[],
        generator_cached=False,
        evaluator_cached=False,
    )


_MINE_KWARGS = {
    "domain": "science.physics",
    "sibling_run_id": "01HZZ-RUN",
    "generator_model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "evaluator_model": "gpt-4o-mini",
    "rubric_version": 1,
}


def test_mine_pairs_emits_all_combinations_above_gap() -> None:
    siblings = [
        _sib("low", creativity=2, logic=2, temperature=0.7),    # score 4
        _sib("mid", creativity=4, logic=3, temperature=1.0),    # score 7
        _sib("high", creativity=5, logic=4, temperature=1.4),   # score 9
    ]
    pairs = mine_pairs("prompt", siblings=siblings, min_gap=2.0, **_MINE_KWARGS)
    # Combinations clearing gap=2: (high,low)=5, (high,mid)=2, (mid,low)=3.
    assert len(pairs) == 3
    by_pair = {(p.chosen, p.rejected): p for p in pairs}
    assert ("high", "low") in by_pair
    assert ("high", "mid") in by_pair
    assert ("mid", "low") in by_pair
    extreme = by_pair[("high", "low")]
    assert isinstance(extreme, PreferencePair)
    assert extreme.chosen_score == 9.0
    assert extreme.rejected_score == 4.0
    assert extreme.score_gap == 5.0
    assert extreme.chosen_temperature == 1.4
    assert extreme.rejected_temperature == 0.7


def test_mine_pairs_partial_gap_filter() -> None:
    # min_gap=3 should keep (high,low)=5 and (mid,low)=3 but drop (high,mid)=2.
    siblings = [
        _sib("low", creativity=2, logic=2),    # 4
        _sib("mid", creativity=4, logic=3),    # 7
        _sib("high", creativity=5, logic=4),   # 9
    ]
    pairs = mine_pairs("prompt", siblings=siblings, min_gap=3.0, **_MINE_KWARGS)
    assert len(pairs) == 2
    assert {(p.chosen, p.rejected) for p in pairs} == {("high", "low"), ("mid", "low")}


def test_mine_pairs_below_gap_returns_empty() -> None:
    siblings = [
        _sib("a", creativity=4, logic=3),  # 7
        _sib("b", creativity=4, logic=4),  # 8
    ]
    assert mine_pairs("prompt", siblings=siblings, min_gap=2.0, **_MINE_KWARGS) == []


def test_mine_pairs_tied_responses_returns_empty() -> None:
    # Same response text drawn at two temperatures (cache collision); even
    # though the scorer might disagree, we refuse to emit chosen == rejected.
    siblings = [
        _sib("same text", creativity=2, logic=2, temperature=0.7),
        _sib("same text", creativity=5, logic=5, temperature=1.4),
    ]
    assert mine_pairs("prompt", siblings=siblings, min_gap=2.0, **_MINE_KWARGS) == []


def test_mine_pairs_pair_id_deterministic() -> None:
    siblings = [
        _sib("low", creativity=2, logic=2),
        _sib("high", creativity=5, logic=5),
    ]
    a = mine_pairs("prompt", siblings=siblings, min_gap=2.0, **_MINE_KWARGS)
    b = mine_pairs("prompt", siblings=siblings, min_gap=2.0, **_MINE_KWARGS)
    assert a[0].id == b[0].id


def test_mine_pairs_empty_or_single_returns_empty() -> None:
    assert mine_pairs("prompt", siblings=[], min_gap=2.0, **_MINE_KWARGS) == []
    only = [_sib("solo", creativity=3, logic=3)]
    assert mine_pairs("prompt", siblings=only, min_gap=2.0, **_MINE_KWARGS) == []


# ---- sample_and_score_siblings -----------------------------------------------


@dataclass
class _StubGenerator:
    """Records generate_with_meta calls; returns canned responses."""

    calls: list[tuple[str, str | None, float, int]]

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt, *, system=None, temperature=1.0, max_tokens=1024, use_cache=True):  # noqa: D401
        text, _ = self.generate_with_meta(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        return text

    def generate_with_meta(
        self,
        prompt,
        *,
        system=None,
        temperature=1.0,
        max_tokens=1024,
        use_cache=True,
    ):
        self.calls.append((prompt, system, temperature, max_tokens))
        return f"resp@{temperature}#{len(self.calls)}", False


def _stub_score_factory(records: list[tuple[str, str]]):
    def _score(prompt: str, response: str, *, model=None, use_cache=True):
        records.append((prompt, response))
        return (
            evaluator_mod.EvaluatorResult(
                creativity=3,
                logic=3,
                rationale="stub",
                failure_modes=[],
            ),
            Usage(prompt_tokens=10, completion_tokens=5, cached=False),
        )

    return _score


def test_sample_and_score_siblings_uses_each_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _StubGenerator()
    score_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(evaluator_mod, "score", _stub_score_factory(score_calls))

    temps = [0.7, 1.0, 1.4]
    siblings = sample_and_score_siblings(
        "prompt",
        n=5,
        temperatures=temps,
        generator=gen,
        max_tokens=128,
    )
    assert len(siblings) == 5
    # Cycle through temperatures: indices 0..4 → temps[0,1,2,0,1]
    expected_temps = [temps[i % len(temps)] for i in range(5)]
    assert [c[2] for c in gen.calls] == expected_temps
    assert [s.temperature for s in siblings] == expected_temps
    assert all(c[3] == 128 for c in gen.calls)


def test_sample_and_score_siblings_calls_evaluator_per_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _StubGenerator()
    score_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(evaluator_mod, "score", _stub_score_factory(score_calls))

    siblings = sample_and_score_siblings(
        "prompt",
        n=4,
        temperatures=[0.7, 1.4],
        generator=gen,
    )
    assert len(score_calls) == 4
    # Each evaluator call sees the prompt and the matching generator response.
    assert all(p == "prompt" for p, _ in score_calls)
    assert [r for _, r in score_calls] == [s.response for s in siblings]


def test_sample_and_score_siblings_rejects_empty_temperatures() -> None:
    with pytest.raises(ValueError):
        sample_and_score_siblings(
            "prompt", n=3, temperatures=[], generator=_StubGenerator()
        )


def test_sibling_sample_score_property() -> None:
    s = _sib("x", creativity=4, logic=3)
    assert s.score == 7.0
