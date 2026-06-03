"""Tests for the tail-sampling temperature/probability calibration sweep
(src/brainstormlm/generation/calibration.py).

Deterministic: a fake LLM whose output depends on the temperature kwarg, plus a
fixed embedder. The sweep must be reproducible from a seed.
"""

from __future__ import annotations

import numpy as np


def embedder_from(mapping: dict):
    def _embed(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = mapping.get(t, mapping["__default__"])
            a = np.asarray(v, dtype=np.float32)
            out.append(a / np.linalg.norm(a))
        return np.stack(out)

    return _embed


# Coherent regime (temp < 2): one dominant modal idea + three distinct tails.
# Salad regime (temp >= 2): ultra-short, incoherent tokens.
def cal_llm():
    def _llm(prompt: str, **kw) -> str:
        temp = kw.get("temperature", 1.0)
        if temp >= 2.0:
            return "1. a (0.7)\n2. b (0.3)"
        return (
            "1. modal idea alpha (0.7)\n"
            "2. tail idea beta (0.1)\n"
            "3. tail idea gamma (0.1)\n"
            "4. tail idea delta (0.1)"
        )

    return _llm


CAL_EMB = embedder_from({
    "modal idea alpha": [1, 0, 0, 0],
    "tail idea beta": [0, 1, 0, 0],
    "tail idea gamma": [0, 0, 1, 0],
    "tail idea delta": [0, 0, 0, 1],
    "a": [1, 0, 0, 0],
    "b": [0, 1, 0, 0],
    "__default__": [1, 0, 0, 0],
})

FLOORS = [0.0, 0.5, 1.0]
TEMPS = [1.0, 2.0]


def _run(**overrides):
    from brainstormlm.generation.calibration import sweep_tail_calibration

    kwargs = dict(
        brief="brainstorm names for a coffee shop",
        llm=cal_llm(),
        embedder=CAL_EMB,
        floors=FLOORS,
        temperatures=TEMPS,
        m=60,
        seed=0,
        coherence_floor=0.5,
        min_chars=3,
    )
    kwargs.update(overrides)
    return sweep_tail_calibration(**kwargs)


def test_grid_shape():
    result = _run()
    assert len(result.cells) == len(FLOORS) * len(TEMPS)
    covered = {(c.modal_floor, c.temperature) for c in result.cells}
    assert covered == {(f, t) for f in FLOORS for t in TEMPS}


def test_metrics_in_range():
    result = _run()
    for c in result.cells:
        assert 0.0 <= c.diversity <= 1.0 + 1e-6
        assert 0.0 <= c.coherence <= 1.0
        assert c.distinct_count >= 1


def test_higher_floor_lowers_diversity():
    """At fixed temperature, flooring the modal up crowds out the tail -> less spread."""
    result = _run()
    by_key = {(c.modal_floor, c.temperature): c for c in result.cells}
    low_floor = by_key[(0.0, 1.0)].diversity
    high_floor = by_key[(1.0, 1.0)].diversity
    assert low_floor >= high_floor


def test_recommended_obeys_coherence_floor():
    result = _run()
    assert result.recommended is not None
    assert result.recommended.coherence >= 0.5
    # the salad regime (temp 2.0) must never be recommended
    assert result.recommended.temperature == 1.0


def test_recommendation_is_max_diversity_among_coherent():
    result = _run()
    coherent = [c for c in result.cells if c.coherence >= 0.5]
    best = max(c.diversity for c in coherent)
    assert result.recommended.diversity == best


def test_determinism():
    a = _run()
    b = _run()
    da = {(c.modal_floor, c.temperature): c.diversity for c in a.cells}
    db = {(c.modal_floor, c.temperature): c.diversity for c in b.cells}
    assert da == db


def test_is_coherent_helper():
    from brainstormlm.generation.calibration import is_coherent

    assert is_coherent("a perfectly fine idea", min_chars=3)
    assert not is_coherent("", min_chars=3)
    assert not is_coherent("x", min_chars=3)
