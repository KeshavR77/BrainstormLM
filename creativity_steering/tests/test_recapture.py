"""recapture_activations: rebuild a lost activations/<pos>.npz from saved responses.

Writes a scored_pool.jsonl (no activations), re-captures via the mock backend, and checks the
restored npz is row-aligned and the right shape — so it drops back into a checkpoint cleanly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from creativity_steering.models.mock import MockBackend

_spec = importlib.util.spec_from_file_location(
    "recapture_activations",
    Path(__file__).resolve().parents[1] / "scripts" / "recapture_activations.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
recapture = _mod.recapture

LAYERS, DIM = 6, 16


def test_recapture_restores_aligned_npz(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for i in range(8):
        rows.append({"example_id": i, "framing": "creative", "prompt": f"p{i}",
                     "response": f"diverse word number {i} aurora", "score": 1.0,
                     "novelty": float(i), "quality_ok": True})
    for i in range(8):
        rows.append({"example_id": 8 + i, "framing": "neutral", "prompt": f"p{i}",
                     "response": "the the the the", "score": 0.0,
                     "novelty": float(-i), "quality_ok": False})
    (run / "scored_pool.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    backend = MockBackend(num_layers=LAYERS, hidden_dim=DIM, creative_layer=3, seed=0)
    system_for = {"creative": "be creative", "neutral": "be plain", "unframed": None}
    written = recapture(run, backend, ["mean_response"], system_for)

    assert written["mean_response"] == (16, LAYERS, DIM)
    arr = np.load(run / "activations" / "mean_response.npz")["acts"]
    assert arr.shape == (16, LAYERS, DIM)
    # row-aligned: example i's activation matches a direct capture of row i
    a0 = backend.capture_activations(rows[0]["prompt"], rows[0]["response"],
                                     system="be creative", positions=["mean_response"])
    assert np.allclose(arr[0], a0["mean_response"])
