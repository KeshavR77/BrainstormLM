"""Tests for the API-shaped step-④ selection adapter.

Embeddings are injected via a text->vector ``embed_fn`` (a dict lookup) so the
adapter is fully offline and robust to the salad gate reordering/filtering the
candidate list.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from brainstormlm.api.public_schemas import DiversityScoreV1, IdeaV1
from brainstormlm.pipeline.select_api import SelectedIdeas, select_response_ideas


def _vec_embed(mapping: dict[str, list[float]]) -> Callable[[Sequence[str]], np.ndarray]:
    return lambda texts: np.array([mapping[t] for t in texts], dtype=np.float64)


def test_returns_typed_contract() -> None:
    cands = ["Alpha Idea\nBody about alpha.", "Beta Idea\nBody about beta."]
    embed = _vec_embed({cands[0]: [1.0, 0.0], cands[1]: [0.0, 1.0]})
    res = select_response_ideas(cands, n=2, drop_salad=False, embed_fn=embed)
    assert isinstance(res, SelectedIdeas)
    assert all(isinstance(i, IdeaV1) for i in res.ideas)
    assert isinstance(res.diversity, DiversityScoreV1)
    assert res.diversity.method == "vendi"
    assert len(res.ideas) == 2


def test_parses_title_and_body() -> None:
    cands = ["Bold Idea\nThis is the body."]
    res = select_response_ideas(
        cands, n=1, drop_salad=False, embed_fn=_vec_embed({cands[0]: [1.0, 0.0]})
    )
    assert res.ideas[0].title == "Bold Idea"
    assert res.ideas[0].body == "This is the body."


def test_drops_near_duplicate_for_coverage() -> None:
    cands = [
        "Solar Roof\nA roof of solar tiles.",
        "Solar Rooftop\nRoof tiles that are solar.",
        "Wind Garden\nUrban wind turbines.",
    ]
    embed = _vec_embed(
        {cands[0]: [1.0, 0.0], cands[1]: [1.0, 0.0], cands[2]: [0.0, 1.0]}
    )
    res = select_response_ideas(cands, n=3, drop_salad=False, embed_fn=embed)
    titles = [i.title for i in res.ideas]
    assert len(res.ideas) == 2
    assert "Solar Roof" in titles
    assert "Wind Garden" in titles


def test_n_caps_count() -> None:
    cands = ["A\na", "B\nb", "C\nc"]
    embed = _vec_embed({cands[0]: [1, 0, 0], cands[1]: [0, 1, 0], cands[2]: [0, 0, 1]})
    res = select_response_ideas(cands, n=2, drop_salad=False, embed_fn=embed)
    assert len(res.ideas) == 2


def test_vendi_score_reflects_spread() -> None:
    cands = ["A\na", "B\nb"]
    embed = _vec_embed({cands[0]: [1.0, 0.0], cands[1]: [0.0, 1.0]})
    res = select_response_ideas(cands, n=2, drop_salad=False, embed_fn=embed)
    assert res.diversity.score > 1.5  # two orthogonal ideas → ~2.0


def test_collapsed_pool_yields_single_idea() -> None:
    cands = ["A\na", "B\nb"]
    embed = _vec_embed({cands[0]: [1.0, 0.0], cands[1]: [1.0, 0.0]})
    res = select_response_ideas(cands, n=2, drop_salad=False, embed_fn=embed)
    assert len(res.ideas) == 1
    assert res.diversity.score == 1.0


def test_id_is_stable_and_deterministic() -> None:
    cands = ["Same Title\nSame body."]
    embed = _vec_embed({cands[0]: [1.0, 0.0]})
    r1 = select_response_ideas(cands, n=1, drop_salad=False, embed_fn=embed)
    r2 = select_response_ideas(cands, n=1, drop_salad=False, embed_fn=embed)
    assert r1.ideas[0].id == r2.ideas[0].id
    assert r1.ideas[0].id.startswith("idea_")


def test_distinct_ideas_get_distinct_ids() -> None:
    cands = ["One\nbody one", "Two\nbody two"]
    embed = _vec_embed({cands[0]: [1.0, 0.0], cands[1]: [0.0, 1.0]})
    res = select_response_ideas(cands, n=2, drop_salad=False, embed_fn=embed)
    assert res.ideas[0].id != res.ideas[1].id


def test_salad_gate_drops_incoherent() -> None:
    clean = (
        "Tidal Lagoon Grid\nA coastal lagoon stores tidal energy in a wide "
        "shallow basin and releases it through low-head turbines to supply "
        "steady baseload power for the regional grid each day."
    )
    salad = "Broken Output\n" + " ".join(["loop"] * 25)
    embed = _vec_embed({clean: [1.0, 0.0]})  # salad is filtered before embedding
    res = select_response_ideas([clean, salad], n=2, drop_salad=True, embed_fn=embed)
    assert [i.title for i in res.ideas] == ["Tidal Lagoon Grid"]


def test_response_gate_is_coherence_only() -> None:
    # A finished idea may be legitimately terse: micro_output / comma_bleed must
    # NOT drop it from the response gate, but token-loop / non-ASCII still do.
    terse = "Solar Roof\nSolar tiles."  # < 20 tokens (micro) yet coherent
    looped = "Broken Loop\n" + " ".join(["loop"] * 25)  # token_loop
    garbled = "Garbled\n" + ("é" * 60)  # > 5% non-ASCII
    embed = _vec_embed({terse: [1.0, 0.0]})  # incoherent ones filtered pre-embed
    res = select_response_ideas(
        [terse, looped, garbled], n=3, drop_salad=True, embed_fn=embed
    )
    assert [i.title for i in res.ideas] == ["Solar Roof"]


def test_empty_pool_returns_empty_contract() -> None:
    res = select_response_ideas([], n=3)
    assert res.ideas == []
    assert res.diversity.score == 0.0
    assert res.diversity.method == "vendi"


def test_all_salad_pool_returns_empty() -> None:
    salad = "Broken\n" + " ".join(["loop"] * 25)
    res = select_response_ideas([salad], n=1, drop_salad=True)
    assert res.ideas == []
    assert res.diversity.score == 0.0
