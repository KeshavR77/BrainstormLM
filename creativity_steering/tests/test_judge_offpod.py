"""Off-pod judging of persisted sweep generations.

Phase 1 sweeps run on-pod with judging OFF and dump every generation to jsonl. This module
re-loads those generations and runs the LLM-judge stack on ALL cells in one batch (no GPU,
no regeneration), producing a judged result in the same shape as an on-pod judged sweep.

Runs offline by injecting fake ``creativity_evaluators.Scorers`` — same pattern as
``test_evaluate_stack``.
"""

import hashlib
import json

import numpy as np

from creativity_evaluators.pipeline import Scorers
from creativity_evaluators.stage1.dsi import DSIScorer
from creativity_evaluators.stage2.ensemble import Ensemble
from creativity_evaluators.types import DIMENSIONS, JudgeResult
from creativity_steering.patching.judge_offpod import (
    judge_generations,
    load_generation_records,
)


def _fake_embed(text):
    toks = text.split()
    vecs = [
        np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest(), 16) % 2**32).normal(size=16)
        for t in toks
    ]
    return toks, (np.array(vecs) if vecs else np.zeros((0, 16)))


class _FakeJudge:
    def __init__(self, name, family):
        self.name = name
        self.family = family

    def score(self, prompt, response):
        # diverse text -> high marks on every dimension; degenerate -> low (fails the 2.0 floor)
        v = 5 if len(set(response.split())) > 4 else 1
        return JudgeResult(scores={d: v for d in DIMENSIONS}, reasoning="", judge_name=self.name)


def _scorers():
    return Scorers(
        dsi=DSIScorer(embed_fn=_fake_embed),
        lexical=None,
        ppl=None,
        reward=None,
        ensemble=Ensemble([_FakeJudge("a", "openai"), _FakeJudge("b", "anthropic")]),
    )


def _records():
    div = "the curious otter juggled five smooth river stones"  # >4 distinct -> prag 5
    deg = "x x x x"  # degenerate -> prag 1, disqualified by the 2.0 floor
    base = "a chair is a seat with four legs and a back"
    recs = []
    for _ in range(3):
        recs.append({"cell": "baseline", "prompt": "Describe a chair.", "response": base})
        recs.append({"cell": "L6_a0.25", "prompt": "Describe a chair.", "response": div,
                     "layer": 6, "alpha": 0.25})
        recs.append({"cell": "L7_a0.3", "prompt": "Describe a chair.", "response": deg,
                     "layer": 7, "alpha": 0.3})
    return recs


def test_judges_all_cells_and_shapes_result_like_a_sweep():
    res = judge_generations(_records(), scorers=_scorers(), quality_floor=2.0)

    # baseline present, unsteered, delta 0
    assert res["baseline"]["label"] == "baseline"
    assert res["baseline"]["delta"] == 0.0
    assert res["baseline"]["n_samples"] == 3

    cells = {c["label"]: c for c in res["cells"]}
    assert set(cells) == {"L6_a0.25", "L7_a0.3"}
    # judging ran on EVERY cell (no top-k gating off-pod)
    assert set(res["judged_labels"]) == {"baseline", "L6_a0.25", "L7_a0.3"}
    for lbl, c in cells.items():
        assert c["judged"] is True
        assert c["n_samples"] == 3
        assert {"composite", "stage1", "pragmaticality", "delta", "layer", "alpha"} <= c.keys()
        assert c["delta"] == c["composite"] - res["baseline"]["composite"]

    # raw judge Pragmaticality is carried through (1-5 scale): diverse high, degenerate low
    assert cells["L6_a0.25"]["pragmaticality"] == 5.0
    assert cells["L7_a0.3"]["pragmaticality"] == 1.0
    # diverse cell clears the floor (scored on novelty); degenerate is disqualified -> lower composite
    assert cells["L6_a0.25"]["composite"] > cells["L7_a0.3"]["composite"]

    # fully serializable
    json.dumps(res)


def test_load_generation_records_dedupes_baseline_across_files(tmp_path):
    # Each chunk's sweep re-emits the baseline; the loader must keep ONE baseline copy so it
    # isn't over-weighted in the batch z-scoring, while keeping all steered cells.
    f1 = tmp_path / "chunk1.jsonl"
    f2 = tmp_path / "chunk2.jsonl"
    f1.write_text("\n".join(json.dumps(r) for r in [
        {"cell": "baseline", "prompt": "p", "response": "b1"},
        {"cell": "baseline", "prompt": "p", "response": "b2"},
        {"cell": "L0_a0.2", "prompt": "p", "response": "x", "layer": 0, "alpha": 0.2},
    ]))
    f2.write_text("\n".join(json.dumps(r) for r in [
        {"cell": "baseline", "prompt": "p", "response": "b1"},
        {"cell": "baseline", "prompt": "p", "response": "b2"},
        {"cell": "L8_a0.2", "prompt": "p", "response": "y", "layer": 8, "alpha": 0.2},
    ]))

    recs = load_generation_records([str(f1), str(f2)])
    by_cell = {}
    for r in recs:
        by_cell.setdefault(r["cell"], []).append(r)
    assert len(by_cell["baseline"]) == 2          # baseline kept once (from the first file)
    assert set(by_cell) == {"baseline", "L0_a0.2", "L8_a0.2"}
    assert len(by_cell["L0_a0.2"]) == 1 and len(by_cell["L8_a0.2"]) == 1
