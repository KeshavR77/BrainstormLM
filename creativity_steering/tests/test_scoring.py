import pytest

from creativity_steering.scoring import FakeScorer, HeuristicScorer, make_scorer


def test_heuristic_prefers_diverse_text():
    s = HeuristicScorer()
    diverse = "lighthouse origami quasar marmalade velvet glacier ember labyrinth zephyr nebula"
    repetitive = "the the the the the the the the the the"
    assert s.score_single(diverse, "prompt") > s.score_single(repetitive, "prompt")


def test_heuristic_scores_in_unit_range():
    s = HeuristicScorer()
    for txt in ["", "a", "lighthouse origami quasar marmalade velvet glacier ember"]:
        assert 0.0 <= s.score_single(txt, "p") <= 1.0
    assert 0.0 <= s.score_collection(["a b c", "d e f"]) <= 1.0


def test_fake_scorer_uses_callable():
    s = FakeScorer(single_fn=lambda t: 0.9 if "good" in t else 0.1, collection=0.7)
    assert s.score_single("good text", "p") == 0.9
    assert s.score_single("bad text", "p") == 0.1
    assert s.score_collection(["x"]) == 0.7


def test_make_scorer_factory():
    assert isinstance(make_scorer("heuristic"), HeuristicScorer)
    with pytest.raises(ValueError):
        make_scorer("nonsense")
