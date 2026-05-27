import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.config import (
    CAPTURE_LAST_RESPONSE,
    CAPTURE_MEAN_RESPONSE,
    ExperimentConfig,
    FilterConfig,
    GenerationConfig,
)
from creativity_steering.derivation.diffmeans import unit_normalize
from creativity_steering.derivation.pipeline import derive_vectors, save_derivation
from creativity_steering.derivation.prompts import (
    CREATIVE_SYSTEM,
    NEUTRAL_SYSTEM,
    TOPICS,
    PromptSet,
)
from creativity_steering.models.mock import MockBackend
from creativity_steering.scoring import HeuristicScorer


def _small_config():
    return ExperimentConfig(
        backend="mock",
        num_layers=12,
        hidden_dim=32,
        mock_creative_layer=6,
        capture_position=CAPTURE_LAST_RESPONSE,
        extra_capture_positions=(CAPTURE_MEAN_RESPONSE,),
        generation=GenerationConfig(samples_per_prompt=3, max_new_tokens=240),
        filtering=FilterConfig(creative_min=0.55, neutral_max=0.45, min_per_class=0),
    )


def _backend(cfg):
    return MockBackend(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        creative_layer=cfg.mock_creative_layer,
        seed=cfg.seed,
    )


def test_pipeline_recovers_planted_layer_end_to_end():
    cfg = _small_config()
    backend = _backend(cfg)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:16])

    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts)

    # Filter actually separated the framings into non-trivial pools.
    assert out.subset.n_positive > 20
    assert out.subset.n_negative > 20
    # DiffMeans at every layer.
    assert out.vector_set.n_layers == 12
    # Norm peaks near the planted creative layer (6) and direction matches the axis.
    norms = out.vector_set.layer_norms
    assert abs(int(np.argmax(norms)) - 6) <= 2
    cos = float(np.dot(unit_normalize(out.vector_set.vectors)[6], backend.creativity_axis[6]))
    assert cos > 0.8
    # Both capture positions were derived.
    assert set(out.per_position) == {CAPTURE_LAST_RESPONSE, CAPTURE_MEAN_RESPONSE}


def test_save_and_reload_roundtrip(tmp_path):
    cfg = _small_config()
    backend = _backend(cfg)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:12])
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts)

    save_derivation(out, tmp_path)
    reloaded = SteeringVectorSet.load(tmp_path / "vectors" / CAPTURE_LAST_RESPONSE)
    np.testing.assert_allclose(reloaded.vectors, out.vector_set.vectors, atol=1e-5)
    assert reloaded.capture_position == CAPTURE_LAST_RESPONSE
    assert reloaded.n_positive == out.subset.n_positive


def test_gather_responses_unframed():
    from creativity_steering.config import DIVERSE_SAMPLING
    from creativity_steering.derivation.generate import gather_responses_unframed

    cfg = _small_config()
    backend = _backend(cfg)
    ex = gather_responses_unframed(
        backend, list(TOPICS[:5]), positions=[CAPTURE_LAST_RESPONSE],
        samples_per_prompt=2, sampling=DIVERSE_SAMPLING,
    )
    assert len(ex) == 10  # 5 topics x 2 samples, no framing distinction
    assert all(e.framing == "unframed" and e.system == "" for e in ex)
    assert all(CAPTURE_LAST_RESPONSE in e.activations for e in ex)


def test_save_persists_verified_contrast_set(tmp_path):
    import json

    cfg = _small_config()
    backend = _backend(cfg)
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS[:12])
    out = derive_vectors(backend, cfg, HeuristicScorer(), prompt_set=prompts)

    save_derivation(out, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "contrast_set.jsonl").read_text().splitlines()]

    # one row per DiffMeans input, labeled by pool, with the response text + score.
    assert len(rows) == out.subset.n_positive + out.subset.n_negative
    pos = [r for r in rows if r["pool"] == "positive"]
    neg = [r for r in rows if r["pool"] == "negative"]
    assert len(pos) == out.subset.n_positive and len(neg) == out.subset.n_negative
    assert all(r["framing"] == "creative" for r in pos)
    assert all(r["framing"] == "neutral" for r in neg)
    assert all({"prompt", "response", "score"} <= r.keys() for r in rows)
