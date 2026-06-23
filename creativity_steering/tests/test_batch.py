"""Batch steer -> refine driver — offline (MockBackend + fake refiner), incl. checkpoint/resume."""

import json

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.models.mock import MockBackend
from creativity_steering.serving.batch import build_prompt, run_briefs
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.service import SteeringService


class _FakeRefiner:
    model = "fake"

    def refine(self, prompt, text):
        return f"R<<{text}>>"

    def info(self):
        return {"model": self.model}


def _svc(refiner=None):
    b = MockBackend(num_layers=24, hidden_dim=128, creative_layer=4, seed=0)
    vs = SteeringVectorSet(vectors=np.asarray(b.creativity_axis, dtype=np.float32),
                           model_id="mock", capture_position="mean_response",
                           n_positive=10, n_negative=10)
    pkg = SteeringPackage(name="t", vectors_path="(x)", layer=4, alpha=8.0)
    return SteeringService(b, vs, pkg, refiner=refiner)


BRIEFS = [
    {"id": "b1", "category": "x", "brief": "Design a chair.", "motif_excludes": ["four legs"]},
    {"id": "b2", "category": "y", "brief": "Design a lamp."},
]

SEEDED = {"id": "b3", "category": "z", "brief": "Design a clock.",
          "archetype_seeds": ["geology — rock", "botany — vine"]}


def test_build_prompt_appends_motif_excludes():
    p = build_prompt(BRIEFS[0])
    assert "Design a chair." in p and "four legs" in p
    assert build_prompt(BRIEFS[1]) == "Design a lamp."  # no excludes, no lens -> unchanged


def test_build_prompt_injects_lens():
    p = build_prompt(BRIEFS[0], lens="geology — eroded stone")
    assert "Design a chair." in p and "geology — eroded stone" in p and "four legs" in p


def test_run_briefs_rotates_archetype_seeds(tmp_path):
    run_briefs(_svc(), [SEEDED], variant="L4", samples=4, out_dir=str(tmp_path))
    rows = sorted((json.loads(line) for line in (tmp_path / "b3.jsonl").read_text().splitlines()
                   if line.strip()), key=lambda r: r["sample"])
    # 2 seeds across 4 samples -> the lenses rotate, so the set can't collapse to one territory
    assert [r["lens"] for r in rows] == ["geology — rock", "botany — vine",
                                         "geology — rock", "botany — vine"]


def test_run_briefs_no_seeds_disables_rotation(tmp_path):
    run_briefs(_svc(), [SEEDED], variant="L4", samples=2, out_dir=str(tmp_path), rotate_seeds=False)
    rows = [json.loads(line) for line in (tmp_path / "b3.jsonl").read_text().splitlines() if line.strip()]
    assert all(r["lens"] is None for r in rows)


def test_run_briefs_writes_jsonl_and_refines(tmp_path):
    summary = run_briefs(_svc(refiner=_FakeRefiner()), BRIEFS, variant="L4", samples=3,
                         out_dir=str(tmp_path), refine=True)
    assert {s["brief_id"] for s in summary} == {"b1", "b2"}
    rows = [json.loads(line) for line in (tmp_path / "b1.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert all(r["refined"] == f"R<<{r['steered']}>>" for r in rows)
    assert (tmp_path / "IMAGE_PROMPTS.md").exists()
    assert "Design a chair." in (tmp_path / "IMAGE_PROMPTS.md").read_text()


def test_run_briefs_baseline_arm(tmp_path):
    run_briefs(_svc(), BRIEFS[:1], variant="L4", samples=1, out_dir=str(tmp_path), baseline=True)
    row = json.loads((tmp_path / "b1.jsonl").read_text().splitlines()[0])
    assert "steered" in row and "baseline" in row  # both arms recorded for A/B


def test_run_briefs_resumes_without_duplicates(tmp_path):
    svc = _svc()
    run_briefs(svc, BRIEFS[:1], variant="L4", samples=2, out_dir=str(tmp_path))      # 0,1
    summary = run_briefs(svc, BRIEFS[:1], variant="L4", samples=4, out_dir=str(tmp_path))  # +2,3
    assert summary[0]["made"] == 2  # only the two missing samples were generated
    rows = [json.loads(line) for line in (tmp_path / "b1.jsonl").read_text().splitlines() if line.strip()]
    assert sorted(r["sample"] for r in rows) == [0, 1, 2, 3]  # resumed, no duplicates
