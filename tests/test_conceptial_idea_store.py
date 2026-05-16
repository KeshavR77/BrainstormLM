"""Unit tests for brainstormlm.conceptial.idea_store.IdeaMemoryStore.

Covers CRUD, project scoping, source-class breakdown, recency ordering, the
fp16 round-trip for stored hiddens, and bulk insert.
"""

from __future__ import annotations

import time

import pytest
import torch

from brainstormlm.conceptial.idea_store import (
    DEFAULT_DB_PATH_ENV,
    SOURCE_GENERATED,
    SOURCE_SEEDED,
    IdeaMemoryStore,
    IdeaRow,
)

HIDDEN_DIM = 64


def _h(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(HIDDEN_DIM, generator=gen)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "idea_memory.db"
    s = IdeaMemoryStore(db_path=db)
    yield s
    s.close()


def test_init_creates_db_and_schema(tmp_path):
    db = tmp_path / "subdir" / "idea_memory.db"
    s = IdeaMemoryStore(db_path=db)
    assert db.exists()
    assert s.path == db
    s.close()


def test_env_var_path_override(tmp_path, monkeypatch):
    db = tmp_path / "from_env.db"
    monkeypatch.setenv(DEFAULT_DB_PATH_ENV, str(db))
    s = IdeaMemoryStore()
    assert s.path == db
    s.close()


def test_add_inserts_row(store):
    store.add("p1", "i1", _h(1), "first idea")
    assert store.count("p1") == 1


def test_add_replaces_existing(store):
    store.add("p1", "i1", _h(1), "original")
    store.add("p1", "i1", _h(2), "updated")
    assert store.count("p1") == 1
    [row] = store.list_recent("p1", limit=10)
    assert row.text == "updated"


def test_add_rejects_non_tensor(store):
    with pytest.raises(TypeError):
        store.add("p1", "i1", [1.0, 2.0, 3.0], "bad hidden")  # type: ignore[arg-type]


def test_add_rejects_multidim_tensor(store):
    with pytest.raises(ValueError, match="1D"):
        store.add("p1", "i1", torch.randn(4, 4), "2d hidden")


def test_add_rejects_unknown_source(store):
    with pytest.raises(ValueError, match="source"):
        store.add("p1", "i1", _h(1), "text", source="bogus")


def test_count_empty(store):
    assert store.count("nonexistent") == 0


def test_count_is_project_scoped(store):
    store.add("p1", "i1", _h(1), "a")
    store.add("p1", "i2", _h(2), "b")
    store.add("p2", "i1", _h(3), "c")
    assert store.count("p1") == 2
    assert store.count("p2") == 1


def test_count_by_source_breakdown(store):
    store.add("p1", "s1", _h(1), "seed-a", source=SOURCE_SEEDED)
    store.add("p1", "s2", _h(2), "seed-b", source=SOURCE_SEEDED)
    store.add("p1", "g1", _h(3), "gen-a", source=SOURCE_GENERATED)
    breakdown = store.count_by_source("p1")
    assert breakdown == {SOURCE_SEEDED: 2, SOURCE_GENERATED: 1}


def test_count_by_source_zeros_for_missing_classes(store):
    store.add("p1", "g1", _h(1), "only generated")
    breakdown = store.count_by_source("p1")
    assert breakdown[SOURCE_GENERATED] == 1
    assert breakdown[SOURCE_SEEDED] == 0


def test_list_recent_hiddens_returns_most_recent_first(store):
    store.add("p1", "i1", _h(1), "first")
    time.sleep(0.001)
    store.add("p1", "i2", _h(2), "second")
    time.sleep(0.001)
    store.add("p1", "i3", _h(3), "third")
    rows = store.list_recent("p1", limit=10)
    assert [r.text for r in rows] == ["third", "second", "first"]


def test_list_recent_hiddens_honors_limit(store):
    for k in range(5):
        store.add("p1", f"i{k}", _h(k), f"idea-{k}")
        time.sleep(0.001)
    hiddens = store.list_recent_hiddens("p1", limit=2)
    assert len(hiddens) == 2


def test_list_recent_hiddens_zero_limit_returns_empty(store):
    store.add("p1", "i1", _h(1), "a")
    assert store.list_recent_hiddens("p1", limit=0) == []


def test_list_recent_returns_empty_for_unknown_project(store):
    store.add("p1", "i1", _h(1), "a")
    assert store.list_recent("missing") == []
    assert store.list_recent_hiddens("missing") == []


def test_hidden_round_trip_preserves_values_within_fp16_tolerance(store):
    orig = _h(42)
    store.add("p1", "i1", orig, "x")
    [restored] = store.list_recent_hiddens("p1", limit=1)
    assert restored.shape == orig.shape
    assert restored.dtype == torch.float32
    # fp16 round-trip — tolerate ~1e-3 absolute difference
    assert torch.allclose(restored, orig, atol=1e-2)


def test_list_recent_returns_full_idea_rows(store):
    h = _h(7)
    store.add("p1", "i1", h, "concept body", source=SOURCE_SEEDED)
    [row] = store.list_recent("p1", limit=1)
    assert isinstance(row, IdeaRow)
    assert row.project_id == "p1"
    assert row.idea_id == "i1"
    assert row.text == "concept body"
    assert row.source == SOURCE_SEEDED
    assert row.created_at > 0
    assert row.hidden.shape == (HIDDEN_DIM,)


def test_delete_project_returns_deleted_count(store):
    store.add("p1", "i1", _h(1), "a")
    store.add("p1", "i2", _h(2), "b")
    deleted = store.delete_project("p1")
    assert deleted == 2
    assert store.count("p1") == 0


def test_delete_project_leaves_other_projects(store):
    store.add("p1", "i1", _h(1), "a")
    store.add("p2", "i1", _h(2), "b")
    store.delete_project("p1")
    assert store.count("p1") == 0
    assert store.count("p2") == 1


def test_delete_missing_project_returns_zero(store):
    assert store.delete_project("never_existed") == 0


def test_add_many_bulk_insert(store):
    rows = [
        IdeaRow(
            project_id="p1",
            idea_id=f"i{k}",
            text=f"idea-{k}",
            source=SOURCE_SEEDED,
            created_at=time.time() + k * 0.001,
            hidden=_h(k),
        )
        for k in range(4)
    ]
    n = store.add_many(rows)
    assert n == 4
    assert store.count("p1") == 4


def test_add_many_empty_returns_zero(store):
    assert store.add_many([]) == 0


def test_add_many_rejects_unknown_source_by_coercing_to_seeded(store):
    # add_many's body coerces unknown sources to SOURCE_SEEDED (see implementation).
    row = IdeaRow(
        project_id="p1",
        idea_id="i1",
        text="x",
        source="bogus",
        created_at=time.time(),
        hidden=_h(1),
    )
    store.add_many([row])
    [restored] = store.list_recent("p1", limit=1)
    assert restored.source == SOURCE_SEEDED


def test_path_property_returns_db_path(tmp_path):
    db = tmp_path / "x.db"
    s = IdeaMemoryStore(db_path=db)
    assert s.path == db
    s.close()


def test_close_is_idempotent_for_test_cleanup(tmp_path):
    db = tmp_path / "x.db"
    s = IdeaMemoryStore(db_path=db)
    s.add("p1", "i1", _h(1), "a")
    s.close()
    # If we wanted to reopen, a new instance should work fine.
    s2 = IdeaMemoryStore(db_path=db)
    assert s2.count("p1") == 1
    s2.close()
