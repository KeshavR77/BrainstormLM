import numpy as np
import pytest

from creativity_steering.derivation.filter import label_and_filter
from creativity_steering.derivation.records import ResponseExample
from creativity_steering.scoring import FakeScorer


def _ex(framing, response, layers=3, dim=4):
    return ResponseExample(
        framing=framing,
        system="sys",
        prompt="p",
        response=response,
        activations={"last_response_token": np.ones((layers, dim), dtype=np.float32)},
    )


# Score = number embedded in the response string.
SCORER = FakeScorer(single_fn=lambda t: float(t))


def test_keeps_only_confident_pools_and_drops_the_middle():
    examples = [
        _ex("creative", "0.9"),   # creative + high → positive
        _ex("creative", "0.5"),   # creative but middling → dropped
        _ex("neutral", "0.1"),    # neutral + low → negative
        _ex("neutral", "0.8"),    # neutral but high → dropped (wrong side)
    ]
    subset = label_and_filter(examples, SCORER, creative_min=0.6, neutral_max=0.4)
    assert subset.n_positive == 1
    assert subset.n_negative == 1
    assert subset.n_dropped == 2
    assert subset.positives[0].score == 0.9


def test_min_per_class_guardrail_raises():
    examples = [_ex("creative", "0.9"), _ex("neutral", "0.1")]
    with pytest.raises(ValueError, match="too small"):
        label_and_filter(examples, SCORER, creative_min=0.6, neutral_max=0.4, min_per_class=5)


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        label_and_filter([], SCORER, creative_min=0.3, neutral_max=0.7)  # neutral_max > creative_min


def test_stacks_have_expected_shape():
    examples = [_ex("creative", "0.9"), _ex("creative", "0.95"), _ex("neutral", "0.05")]
    subset = label_and_filter(examples, SCORER, creative_min=0.6, neutral_max=0.4)
    pos, neg = subset.stacks("last_response_token")
    assert pos.shape == (2, 3, 4)
    assert neg.shape == (1, 3, 4)
