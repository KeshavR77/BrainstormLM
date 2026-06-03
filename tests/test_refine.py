"""Tests for the refine orchestration (src/brainstormlm/refine.py).

Fully deterministic: the LLM client and embedder are injected as fakes. The fake
LLM routes by stage using marker substrings from the real module system prompts:

  - planner_angles._PLANNER_SYSTEM contains "ORTHOGONAL creative angles"
  - verbalized_sampling._VS_SYSTEM contains "assign a probability"
  - anything else is treated as the brief-refine call.

The adapter tests monkeypatch the underlying client functions so no API is hit.
"""

from __future__ import annotations

import numpy as np

# Marker substrings that must stay in sync with the module system prompts.
_PLANNER_MARKER = "ORTHOGONAL creative angles"
_VS_MARKER = "assign a probability"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def embedder_from(mapping: dict) -> "callable":
    """Build a deterministic embedder returning L2-normalised vectors per text."""

    def _embed(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = mapping.get(t, mapping["__default__"])
            a = np.asarray(v, dtype=np.float32)
            out.append(a / np.linalg.norm(a))
        return np.stack(out)

    return _embed


class FakeLLM:
    """Routes by stage; counts calls; supports per-angle multi-response (regeneration)."""

    def __init__(self, brief: str, angles_text: str, vs_map: dict[str, list[str]]):
        self.brief = brief
        self.angles_text = angles_text
        self.vs_map = vs_map
        self.n_brief = 0
        self.n_planner = 0
        self.n_vs = 0
        self._vs_counts: dict[str, int] = {}

    def __call__(self, prompt: str, **kwargs) -> str:
        if _PLANNER_MARKER in prompt:
            self.n_planner += 1
            return self.angles_text
        if _VS_MARKER in prompt:
            self.n_vs += 1
            for tok, responses in self.vs_map.items():
                if tok in prompt:
                    i = self._vs_counts.get(tok, 0)
                    self._vs_counts[tok] = i + 1
                    return responses[min(i, len(responses) - 1)]
            raise AssertionError(f"FakeLLM: no VS response for prompt: {prompt[:120]!r}")
        self.n_brief += 1
        return self.brief


# Common fixtures for the orchestration tests --------------------------------

_ANGLES_TEXT = "1. alpha\n2. beta\n3. gamma"


def _vs(modal: str, pick: str) -> str:
    return f"1. {modal} (0.7)\n2. {pick} (0.3)"


def _diverse_llm() -> FakeLLM:
    return FakeLLM(
        brief="REFINED BRIEF",
        angles_text=_ANGLES_TEXT,
        vs_map={
            "alpha": [_vs("alpha_modal", "alpha_pick")],
            "beta": [_vs("beta_modal", "beta_pick")],
            "gamma": [_vs("gamma_modal", "gamma_pick")],
        },
    )


_DIVERSE_EMB = embedder_from({
    "alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1],
    "alpha_pick": [1, 0, 0], "beta_pick": [0, 1, 0], "gamma_pick": [0, 0, 1],
    "__default__": [1, 0, 0],
})


# ---------------------------------------------------------------------------
# Adapter tests (the tuple -> plain seam)
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_make_llm_returns_str_and_drops_usage(self, monkeypatch):
        import brainstormlm.clients.gemini as gem
        from brainstormlm.refine import make_llm

        def fake_complete(system, messages, model, max_tokens=1024, temperature=1.0, **kw):
            return "RESP", gem.Usage(1, 2)

        monkeypatch.setattr(gem, "complete", fake_complete)
        llm = make_llm(provider="gemini", model="gemini-2.5-flash-lite")
        out = llm("hello")
        assert isinstance(out, str)
        assert out == "RESP"

    def test_make_llm_forwards_temperature(self, monkeypatch):
        import brainstormlm.clients.gemini as gem
        from brainstormlm.refine import make_llm

        seen = {}

        def fake_complete(system, messages, model, max_tokens=1024, temperature=1.0, **kw):
            seen["temperature"] = temperature
            seen["messages"] = messages
            return "ok", gem.Usage(0, 0)

        monkeypatch.setattr(gem, "complete", fake_complete)
        llm = make_llm(provider="gemini", model="m")
        llm("a prompt", temperature=0.3)
        assert seen["temperature"] == 0.3
        # the prompt must be delivered as a user message
        assert seen["messages"][-1]["content"] == "a prompt"

    def test_make_llm_openai_provider(self, monkeypatch):
        import brainstormlm.clients.openai as oai
        from brainstormlm.refine import make_llm

        monkeypatch.setattr(oai, "complete", lambda *a, **k: ("FROM_OPENAI", oai.Usage(1, 1)))
        llm = make_llm(provider="openai", model="gpt-4o-mini")
        assert llm("hi") == "FROM_OPENAI"

    def test_make_embedder_returns_matrix_and_drops_usage(self, monkeypatch):
        import brainstormlm.clients.openai as oai
        from brainstormlm.refine import make_embedder

        matrix = np.eye(2, dtype=np.float32)
        monkeypatch.setattr(oai, "embed", lambda texts, **k: (matrix, oai.Usage(1, 0)))
        embed = make_embedder()
        out = embed(["a", "b"])
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, matrix)


# ---------------------------------------------------------------------------
# Brief-refine
# ---------------------------------------------------------------------------


class TestRefineBrief:
    def test_returns_brief_and_calls_once(self):
        from brainstormlm.refine import refine_brief

        llm = _diverse_llm()
        brief = refine_brief("make a logo", context="for a coffee shop", llm=llm)
        assert brief == "REFINED BRIEF"
        assert llm.n_brief == 1


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


class TestRefine:
    def test_returns_refineresult_fields(self):
        from brainstormlm.refine import refine, RefineResult

        result = refine(
            "design a logo", k=3, modal_floor=0.0,
            llm=_diverse_llm(), embedder=_DIVERSE_EMB,
        )
        assert isinstance(result, RefineResult)
        assert result.brief == "REFINED BRIEF"
        assert result.angles == ["alpha", "beta", "gamma"]
        assert isinstance(result.distinctness, float)

    def test_one_idea_per_surviving_angle(self):
        from brainstormlm.refine import refine

        result = refine(
            "design a logo", k=3, modal_floor=0.0,
            llm=_diverse_llm(), embedder=_DIVERSE_EMB,
        )
        assert result.ideas == ["alpha_pick", "beta_pick", "gamma_pick"]

    def test_call_sequence_counts(self):
        from brainstormlm.refine import refine

        llm = _diverse_llm()
        refine("design a logo", k=3, modal_floor=0.0, llm=llm, embedder=_DIVERSE_EMB)
        assert llm.n_brief == 1
        assert llm.n_planner == 1
        assert llm.n_vs == 3  # one per surviving angle, no replacement

    def test_no_replacement_when_diverse(self):
        from brainstormlm.refine import refine

        result = refine(
            "design a logo", k=3, modal_floor=0.0,
            llm=_diverse_llm(), embedder=_DIVERSE_EMB, replace_threshold=0.9,
        )
        assert result.replaced == []
        assert result.distinctness > 0.5

    def test_replacement_runs_on_duplicate(self):
        from brainstormlm.refine import refine

        # gamma_pick is a near-duplicate of alpha_pick; the critic flags the
        # redundant slot and the generator regenerates it on a fresh axis.
        llm = FakeLLM(
            brief="REFINED BRIEF",
            angles_text=_ANGLES_TEXT,
            vs_map={
                "alpha": [_vs("alpha_modal", "alpha_pick"), _vs("alpha_modal2", "alpha_pick2")],
                "beta": [_vs("beta_modal", "beta_pick")],
                "gamma": [_vs("gamma_modal", "gamma_pick"), _vs("gamma_modal2", "gamma_pick2")],
            },
        )
        emb = embedder_from({
            "alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1],
            "alpha_pick": [1, 0, 0], "beta_pick": [0, 1, 0],
            "gamma_pick": [1, 0.02, 0],          # ~ alpha_pick
            "alpha_pick2": [0, 0, 1], "gamma_pick2": [0, 0, 1],  # orthogonal resolver
            "__default__": [0, 0, 1],
        })
        result = refine(
            "design a logo", k=3, modal_floor=0.0,
            llm=llm, embedder=emb, replace_threshold=0.4, max_replacements=2,
        )
        assert result.replaced, "expected at least one replacement on a duplicate set"
        # after replacement the redundant idea is gone
        assert "gamma_pick" not in result.ideas or "alpha_pick" not in result.ideas

    def test_max_replacements_bound(self):
        from brainstormlm.refine import refine

        # Every pick collapses to the same vector, so the set never diversifies;
        # the loop must still terminate at max_replacements.
        llm = FakeLLM(
            brief="REFINED BRIEF",
            angles_text=_ANGLES_TEXT,
            vs_map={
                "alpha": [_vs("am", "ap")],
                "beta": [_vs("bm", "bp")],
                "gamma": [_vs("gm", "gp")],
            },
        )
        emb = embedder_from({
            "alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1],
            "__default__": [1, 0, 0],  # all picks identical -> always redundant
        })
        result = refine(
            "design a logo", k=3, modal_floor=0.0,
            llm=llm, embedder=emb, replace_threshold=0.5, max_replacements=2,
        )
        assert len(result.replaced) <= 2
