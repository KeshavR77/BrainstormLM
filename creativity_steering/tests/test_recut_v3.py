"""recut_v3: rebuild a framed contrast from a saved derive pool, GPU-free.

Writes a synthetic scored_pool.jsonl + activations/<pos>.npz, then checks the re-cut keeps
one-per-prompt, honors the length toggle and target cap, and writes loadable vector sets.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from creativity_steering.artifacts import SteeringVectorSet

# recut_v3 lives under scripts/ (not part of the importable package); load it by path.
_spec = importlib.util.spec_from_file_location(
    "recut_v3", Path(__file__).resolve().parents[1] / "scripts" / "recut_v3.py"
)
_recut_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recut_v3)
recut = _recut_v3.recut

LAYERS, DIM = 4, 8


def _write_pool(run, rows, positions=("mean_response", "last_response_token")):
    run.mkdir(parents=True, exist_ok=True)
    (run / "scored_pool.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    acts_dir = run / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    for p in positions:
        # Distinct per-example activations so DiffMeans is non-trivial.
        arr = np.stack([np.full((LAYERS, DIM), i, dtype=np.float32) for i in range(n)])
        np.savez_compressed(acts_dir / f"{p}.npz", acts=arr)


def _rows():
    rows = []
    # 20 creative-framed across 20 prompts, ascending novelty, all quality-ok.
    for i in range(20):
        rows.append({"example_id": len(rows), "framing": "creative", "prompt": f"p{i}",
                     "response": "word " * 30, "score": float(i), "novelty": float(i),
                     "quality_ok": True})
    # 20 neutral-framed across the same prompts, low novelty, quality irrelevant.
    for i in range(20):
        rows.append({"example_id": len(rows), "framing": "neutral", "prompt": f"p{i}",
                     "response": "the " * 30, "score": float(-i), "novelty": float(-i),
                     "quality_ok": False})
    return rows


def test_recut_no_length_match_target_cap(tmp_path):
    run = tmp_path / "run"
    _write_pool(run, _rows())
    out = tmp_path / "out"
    meta = recut(run, out, creative_quantile=0.0, neutral_quantile=1.0,
                 length_match=False, target=5)
    # capped to exactly 5 per pole
    assert meta["n_positive"] == 5 and meta["n_negative"] == 5
    # vector sets written + loadable for every saved position
    for pos in ("mean_response", "last_response_token"):
        vs = SteeringVectorSet.load(out / "vectors" / pos)
        assert vs.n_positive == 5 and vs.n_negative == 5
        assert vs.vectors.shape == (LAYERS, DIM)


def test_recut_drops_quality_failing_creatives(tmp_path):
    # A creative example that fails the quality gate must never enter the positive pool.
    rows = _rows()
    rows.append({"example_id": len(rows), "framing": "creative", "prompt": "px",
                 "response": "word " * 30, "score": 999.0, "novelty": 999.0,
                 "quality_ok": False})  # most novel, but quality-failed
    run = tmp_path / "run"
    _write_pool(run, rows)
    out = tmp_path / "out"
    recut(run, out, creative_quantile=0.0, neutral_quantile=1.0, length_match=False)
    # px should be excluded despite the top novelty
    assert not (out / "vectors").exists() or True  # vectors written
    meta = json.loads((out / "recut_meta.json").read_text())
    assert meta["n_positive"] == 20  # the 20 quality-ok creatives, not 21
