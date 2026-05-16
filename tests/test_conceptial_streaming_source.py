"""Unit tests for brainstormlm.conceptial.streaming_source.

Covers: deterministic pack selection, anti-mode exclusion across calls,
reset behavior, store persistence, no-pacing mode for tests.
"""

from __future__ import annotations

import pytest

from brainstormlm.conceptial.demo_packs import all_demo_ideas
from brainstormlm.conceptial.idea_store import IdeaMemoryStore
from brainstormlm.conceptial.streaming_source import (
    ConceptialStreamingSource,
    StreamingIdea,
)


@pytest.fixture
def store(tmp_path):
    s = IdeaMemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_streams_n_ideas(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    ideas = list(src.stream(prompt="brand identity ideas", n=5))
    assert len(ideas) == 5
    assert all(isinstance(i, StreamingIdea) for i in ideas)


def test_no_within_call_duplicates(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    ideas = list(src.stream(prompt="brand identity", n=5))
    summaries = [i.summary for i in ideas]
    assert len(set(summaries)) == 5


def test_regeneration_excludes_prior_ideas(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    first = list(src.stream(prompt="brand identity", n=5))
    second = list(src.stream(prompt="brand identity", n=5))
    first_summaries = {i.summary for i in first}
    second_summaries = {i.summary for i in second}
    assert first_summaries.isdisjoint(second_summaries)


def test_persists_to_store(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    list(src.stream(prompt="brand identity", n=4))
    assert store.count("p1") == 4


def test_project_scoped_memory(store):
    """Two different projects with the same prompt get to use the same ideas."""
    src_a = ConceptialStreamingSource(store=store, project_id="proj-a", pacing=False)
    src_b = ConceptialStreamingSource(store=store, project_id="proj-b", pacing=False)
    list(src_a.stream(prompt="brand identity", n=3))
    list(src_b.stream(prompt="brand identity", n=3))
    assert store.count("proj-a") == 3
    assert store.count("proj-b") == 3
    # Different projects should be allowed to surface overlapping ideas because
    # each project has its own exclusion set.


def test_excluded_summaries_grow_across_emits(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    ideas = list(src.stream(prompt="brand identity", n=3))
    # First emit sees no exclusions (store was empty); subsequent grow.
    assert len(ideas[0].excluded_summaries) == 0
    assert len(ideas[1].excluded_summaries) == 1
    assert len(ideas[2].excluded_summaries) == 2


def test_streams_stop_when_catalogue_exhausted(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    total = len(all_demo_ideas())
    # Ask for more than exists. Should yield exactly `total`, not crash.
    ideas = list(src.stream(prompt="brand identity", n=total + 5))
    assert len(ideas) == total


def test_returns_idea_with_pack_field(store):
    src = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    [first, *_] = list(src.stream(prompt="brand identity", n=1))
    assert first.pack in {"brand-identity", "campaign-craft", "object-design"}


def test_deterministic_starting_point_for_same_inputs(store, tmp_path):
    """Same (prompt, project) → same first idea, even across new store instances."""
    store2 = IdeaMemoryStore(db_path=tmp_path / "test2.db")
    src_a = ConceptialStreamingSource(store=store, project_id="p1", pacing=False)
    src_b = ConceptialStreamingSource(store=store2, project_id="p1", pacing=False)
    a = next(iter(src_a.stream(prompt="brand identity", n=1)))
    b = next(iter(src_b.stream(prompt="brand identity", n=1)))
    assert a.summary == b.summary
    store2.close()
