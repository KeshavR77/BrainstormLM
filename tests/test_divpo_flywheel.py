"""Tests for the DivPO data-logging flywheel STUB.

Embeddings are passed directly (the ``embeddings=`` param) so the rarity signal
is deterministic and offline; one case exercises the default _embed path.
"""

from __future__ import annotations

import numpy as np

from brainstormlm.training import preferences
from brainstormlm.training.divpo import (
    DivpoCandidate,
    DivpoConfig,
    log_divpo_round,
    mine_divpo_pairs,
)

# idx0 is a semantic singleton (rare); idx1..3 collapse into one cluster (common).
_EMB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _rare_good_batch() -> list[DivpoCandidate]:
    return [
        DivpoCandidate("rare", 8.0),
        DivpoCandidate("common1", 2.0),
        DivpoCandidate("common2", 2.0),
        DivpoCandidate("common3", 2.0),
    ]


def test_rare_and_good_is_chosen() -> None:
    pairs = mine_divpo_pairs("brief", _rare_good_batch(), run_id="r1", embeddings=_EMB)
    assert len(pairs) == 3
    assert all(p.chosen == "rare" for p in pairs)
    assert all(p.rejected.startswith("common") for p in pairs)


def test_rare_but_low_quality_yields_no_pair() -> None:
    # The singleton (idx0) is rare but LOW quality; the common cluster is high.
    cands = [
        DivpoCandidate("rare", 2.0),
        DivpoCandidate("common1", 8.0),
        DivpoCandidate("common2", 8.0),
        DivpoCandidate("common3", 8.0),
    ]
    assert mine_divpo_pairs("brief", cands, run_id="r1", embeddings=_EMB) == []


def test_all_same_cluster_yields_no_pair() -> None:
    emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    cands = [
        DivpoCandidate("a", 9.0),
        DivpoCandidate("b", 1.0),
        DivpoCandidate("c", 1.0),
    ]
    # No rarity differences → strict rarer-than test never passes.
    assert mine_divpo_pairs("brief", cands, run_id="r1", embeddings=emb) == []


def test_fewer_than_two_candidates() -> None:
    assert mine_divpo_pairs("brief", [DivpoCandidate("solo", 9.0)], run_id="r1") == []


def test_min_quality_gap_respected() -> None:
    # Rare candidate only 1.0 above the common ones; default gap is 2.0.
    cands = [
        DivpoCandidate("rare", 3.0),
        DivpoCandidate("common1", 2.0),
        DivpoCandidate("common2", 2.0),
        DivpoCandidate("common3", 2.0),
    ]
    assert mine_divpo_pairs("brief", cands, run_id="r1", embeddings=_EMB) == []


def test_mined_pair_id_matches_helper() -> None:
    pairs = mine_divpo_pairs("brief", _rare_good_batch(), run_id="r1", embeddings=_EMB)
    expected = preferences.pair_id("brief", pairs[0].chosen, pairs[0].rejected)
    assert pairs[0].id == expected


def test_divpo_origin_encoded_in_run_id() -> None:
    pairs = mine_divpo_pairs("brief", _rare_good_batch(), run_id="run42", embeddings=_EMB)
    assert all(p.sibling_run_id == "divpo:run42" for p in pairs)
    assert all(p.source == "sibling_sampling" for p in pairs)
    assert all(p.domain == "divpo" for p in pairs)


def test_log_round_trip_is_append_only(tmp_path) -> None:
    path = tmp_path / "divpo.jsonl"
    pairs = log_divpo_round(path, "brief", _rare_good_batch(), run_id="r1", embeddings=_EMB)
    assert len(pairs) == 3
    assert len(list(preferences.read_all(path))) == 3
    # second round appends rather than overwrites
    log_divpo_round(path, "brief", _rare_good_batch(), run_id="r2", embeddings=_EMB)
    assert len(list(preferences.read_all(path))) == 6


def test_config_overrides_gap() -> None:
    cands = [
        DivpoCandidate("rare", 3.0),
        DivpoCandidate("common1", 2.0),
        DivpoCandidate("common2", 2.0),
        DivpoCandidate("common3", 2.0),
    ]
    cfg = DivpoConfig(min_quality_gap=1.0)
    pairs = mine_divpo_pairs("brief", cands, run_id="r1", config=cfg, embeddings=_EMB)
    assert len(pairs) == 3


def test_default_embed_path(monkeypatch) -> None:
    import brainstormlm.training.divpo as divpo

    monkeypatch.setattr(divpo, "_embed", lambda texts: _EMB[: len(texts)])
    pairs = mine_divpo_pairs("brief", _rare_good_batch(), run_id="r1")
    assert len(pairs) == 3
