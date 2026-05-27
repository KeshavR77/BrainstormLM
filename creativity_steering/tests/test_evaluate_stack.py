"""Two-stage sweep evaluator: score steered generations with the creativity_evaluators
stack (DSI + LLM-judge), batch-relative, with judge gating to the top-k cells.

Runs offline by injecting ``creativity_evaluators.Scorers`` whose leaves are fakes
(hash embedder for DSI, distinct-word-count reward, optional fake judge ensemble) — the
same pattern as ``test_evaluator_filter``.
"""

import hashlib

import numpy as np

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage1.reward_model import RewardModelScorer
from creativity_evaluators.stage2.ensemble import Ensemble
from creativity_evaluators.types import DIMENSIONS, JudgeResult
from creativity_steering.models.backend import Injection
from creativity_steering.models.mock import MockBackend
from creativity_steering.patching.evaluate_stack import StackCell, two_stage_cell_sweep

BRIEFS = ("Describe a chair.",)


def _fake_embed(text):
    toks = text.split()
    vecs = [
        np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=16)
        for t in toks
    ]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


def _distinct(prompt, response):  # reward proxy = vocabulary richness
    return float(len(set(response.lower().split())))


class _FakeJudge:
    def __init__(self, name, family):
        self.name = name
        self.family = family

    def score(self, prompt, response):
        v = 5 if len(set(response.split())) > 4 else 1
        return JudgeResult(scores={d: v for d in DIMENSIONS}, reasoning="", judge_name=self.name)


def _scorers(ensemble=None):
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=None,
        ppl=None,
        reward=RewardModelScorer(score_fn=_distinct),
        ensemble=ensemble,
    )


def _pos_neg_cells(backend, layer=6, alpha=6.0):
    """A positive- and negative-alpha cell injecting the planted axis at the salient layer.

    With ``rescale=False`` the mock turns +alpha into more lexically-diverse text and
    -alpha into near-degenerate text, so the evaluator stack must score +alpha higher.
    """
    axis = backend.creativity_axis[layer]
    pos = StackCell(
        label=f"L{layer}_pos",
        injections=[Injection(layer=layer, vector=axis, alpha=alpha, rescale=False)],
        meta={"layer": layer, "alpha": alpha},
    )
    neg = StackCell(
        label=f"L{layer}_neg",
        injections=[Injection(layer=layer, vector=axis, alpha=-alpha, rescale=False)],
        meta={"layer": layer, "alpha": -alpha},
    )
    return pos, neg


def test_two_stage_ranks_creative_cell_above_degenerate_no_judge():
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    pos, neg = _pos_neg_cells(backend)
    res = two_stage_cell_sweep(
        backend, [pos, neg], scorers=_scorers(), enable_judge=False,
        briefs=BRIEFS, n_samples=4, seed=0,
    )
    # baseline always present and unsteered
    assert res.baseline.label == "baseline"
    # the positively-steered cell is the winner
    best = res.best_cell()
    assert best["label"] == "L6_pos"
    assert best["layer"] == 6 and best["alpha"] == 6.0
    pos_score = next(c.composite for c in res.cells if c.label == "L6_pos")
    neg_score = next(c.composite for c in res.cells if c.label == "L6_neg")
    assert pos_score > neg_score
    # no judging requested
    assert res.judged_labels == []
    assert all(c.judged is False for c in res.cells)


def test_judge_gating_runs_stage2_only_on_top_k_cells():
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    pos, neg = _pos_neg_cells(backend)
    ensemble = Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")])
    res = two_stage_cell_sweep(
        backend, [pos, neg], scorers=_scorers(ensemble=ensemble),
        enable_judge=True, judge_top_k=1, stage1_floor_percentile=0.0,
        briefs=BRIEFS, n_samples=4, seed=0,
    )
    # Stage 2 ran on the baseline + the single top cell only — the degenerate cell is skipped.
    assert set(res.judged_labels) == {"baseline", "L6_pos"}
    best = res.best_cell()
    assert best["label"] == "L6_pos"
    assert best["judged"] is True
    pos_cell = next(c for c in res.cells if c.label == "L6_pos")
    neg_cell = next(c for c in res.cells if c.label == "L6_neg")
    assert pos_cell.judged is True
    assert neg_cell.judged is False  # gated out


def test_ranks_by_novelty_when_no_quality_pole():
    # No reward model and no judges -> the final composite (novelty x quality) collapses to 0
    # for every cell. The sweep must still rank cells by the Stage-1 novelty composite, so the
    # creative cell wins. (This is the judge-gating fix; pre-fix every cell tied at 0.)
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    pos, neg = _pos_neg_cells(backend)
    scorers = Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed), lexical=None, ppl=None, reward=None, ensemble=None
    )
    res = two_stage_cell_sweep(
        backend, [pos, neg], scorers=scorers, enable_judge=False,
        briefs=BRIEFS, n_samples=4, seed=0,
    )
    pos_s = next(c.composite for c in res.cells if c.label == "L6_pos")
    neg_s = next(c.composite for c in res.cells if c.label == "L6_neg")
    assert pos_s > neg_s
    assert res.best_cell()["label"] == "L6_pos"


def test_enable_judge_without_ensemble_is_graceful():
    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    pos, neg = _pos_neg_cells(backend)
    res = two_stage_cell_sweep(
        backend, [pos, neg], scorers=_scorers(ensemble=None),
        enable_judge=True, judge_top_k=2, briefs=BRIEFS, n_samples=3, seed=0,
    )
    # No ensemble available -> no Stage 2, everything falls back to the Stage-1 score.
    assert res.judged_labels == []
    assert all(c.judged is False for c in res.cells)
    assert res.best_cell()["label"] == "L6_pos"


class _StubVectorSet:
    """Minimal stand-in exposing what the cell builders need (unit_vectors + n_layers)."""

    def __init__(self, n_layers, dim):
        self.n_layers = n_layers
        self._u = np.tile(np.arange(1, dim + 1, dtype=np.float32), (n_layers, 1))

    def unit_vectors(self):
        return self._u


def test_build_all_layers_cells():
    from creativity_steering.patching.evaluate_stack import build_all_layers_cells

    vs = _StubVectorSet(n_layers=6, dim=4)
    cells = build_all_layers_cells(vs, source_layers=[4, 2], alphas=[0.1, -0.1])
    assert len(cells) == 4  # 2 sources x 2 alphas
    c = cells[0]
    assert c.meta["experiment"] == "all_layers" and c.meta["source_layer"] == 4
    assert len(c.injections) == 6  # injected at *every* layer
    assert {inj.layer for inj in c.injections} == set(range(6))


def test_build_multi_vector_cells():
    from creativity_steering.patching.evaluate_stack import build_multi_vector_cells

    vs = _StubVectorSet(n_layers=8, dim=4)
    cells = build_multi_vector_cells(vs, layer_groups=[[5, 6, 7]], alphas=[1.0])
    assert len(cells) == 1
    c = cells[0]
    assert c.meta["experiment"] == "multi_vector" and c.meta["layers"] == [5, 6, 7]
    assert {inj.layer for inj in c.injections} == {5, 6, 7}  # each at its own layer


def test_to_dict_is_serializable_and_carries_best():
    import json

    backend = MockBackend(num_layers=12, hidden_dim=32, creative_layer=6, seed=0)
    pos, neg = _pos_neg_cells(backend)
    res = two_stage_cell_sweep(
        backend, [pos, neg], scorers=_scorers(), enable_judge=False,
        briefs=BRIEFS, n_samples=3, seed=0,
    )
    blob = json.dumps(res.to_dict())  # must not raise
    d = json.loads(blob)
    assert d["best"]["label"] == "L6_pos"
    assert d["evaluator"] == "two_stage"
    assert len(d["cells"]) == 2
