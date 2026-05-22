import numpy as np

from creativity_steering.config import CAPTURE_LAST_RESPONSE
from creativity_steering.derivation.diffmeans import diffmeans_per_layer, unit_normalize
from creativity_steering.models.backend import Injection
from creativity_steering.models.mock import MockBackend, text_creativity

CREATIVE_SYS = "You are wildly creative and inventive. Be original and surprising."
NEUTRAL_SYS = "You are a careful, conventional assistant. Give the standard expected answer."
TOPICS = [f"Topic number {i} about everyday objects." for i in range(60)]


def test_metadata_and_determinism():
    b = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=1)
    assert b.num_layers == 12 and b.hidden_dim == 32
    assert "mock-qwen" in b.model_id
    t1 = b.generate("hello", system=CREATIVE_SYS, seed=3)
    t2 = b.generate("hello", system=CREATIVE_SYS, seed=3)
    assert t1 == t2  # deterministic
    a1 = b.capture_activations("hello", t1, positions=[CAPTURE_LAST_RESPONSE])
    a2 = b.capture_activations("hello", t1, positions=[CAPTURE_LAST_RESPONSE])
    np.testing.assert_array_equal(a1[CAPTURE_LAST_RESPONSE], a2[CAPTURE_LAST_RESPONSE])


def test_creative_framing_scores_higher_than_neutral():
    b = MockBackend(seed=2)
    creative = [text_creativity(b.generate(t, system=CREATIVE_SYS, seed=0)) for t in TOPICS]
    neutral = [text_creativity(b.generate(t, system=NEUTRAL_SYS, seed=0)) for t in TOPICS]
    assert np.mean(creative) > np.mean(neutral) + 0.3


def _gather(b, system, positions):
    stack = {p: [] for p in positions}
    for t in TOPICS:
        resp = b.generate(t, system=system, seed=0)
        acts = b.capture_activations(t, resp, system=system, positions=positions)
        for p in positions:
            stack[p].append(acts[p])
    return {p: np.stack(v) for p, v in stack.items()}


def test_diffmeans_recovers_planted_axis_peaking_at_creative_layer():
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=12, seed=0)
    pos = _gather(b, CREATIVE_SYS, [CAPTURE_LAST_RESPONSE])[CAPTURE_LAST_RESPONSE]
    neg = _gather(b, NEUTRAL_SYS, [CAPTURE_LAST_RESPONSE])[CAPTURE_LAST_RESPONSE]
    vecs = diffmeans_per_layer(pos, neg)  # (24, 128)
    norms = np.linalg.norm(vecs, axis=-1)
    # Norm should peak near the planted creative layer.
    assert abs(int(np.argmax(norms)) - 12) <= 2
    # Recovered direction at the peak should align with the planted axis.
    cos = float(np.dot(unit_normalize(vecs)[12], b.creativity_axis[12]))
    assert cos > 0.8


def test_assistant_start_carries_weaker_signal_than_response():
    from creativity_steering.config import CAPTURE_ASSISTANT_START

    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=12, seed=0)
    positions = [CAPTURE_LAST_RESPONSE, CAPTURE_ASSISTANT_START]
    pos = _gather(b, CREATIVE_SYS, positions)
    neg = _gather(b, NEUTRAL_SYS, positions)
    last = np.linalg.norm(diffmeans_per_layer(pos[CAPTURE_LAST_RESPONSE], neg[CAPTURE_LAST_RESPONSE]), axis=-1)[12]
    start = np.linalg.norm(diffmeans_per_layer(pos[CAPTURE_ASSISTANT_START], neg[CAPTURE_ASSISTANT_START]), axis=-1)[12]
    assert last > start  # response position carries far more signal than assistant-start


def test_positive_alpha_along_axis_increases_creativity():
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=12, seed=0)
    axis = b.creativity_axis[12]
    pos = b.generate_steered("Describe a chair.", injections=[Injection(12, axis, alpha=4.0)], seed=0)
    neg = b.generate_steered("Describe a chair.", injections=[Injection(12, axis, alpha=-4.0)], seed=0)
    assert text_creativity(pos) > text_creativity(neg)
