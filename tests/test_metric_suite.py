"""Tests for the metric-suite aggregator — orchestration only.

The embedding-based metrics are patched at the suite namespace so these tests
stay offline and assert *which* blocks the report assembles. The underlying
metrics have their own dedicated tests.
"""

from __future__ import annotations

import pytest

from brainstormlm.eval.metric_suite import metric_suite_report

TEXTS = ["alpha beta gamma", "delta epsilon zeta"]


@pytest.fixture(autouse=True)
def _patch_embedding_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import brainstormlm.eval.metric_suite as ms

    monkeypatch.setattr(
        ms,
        "novelty_bench_distinct",
        lambda texts, **k: {"n_distinct": 2, "n_total": 2, "distinct_ratio": 1.0},
    )
    monkeypatch.setattr(ms, "vendi_score", lambda texts: 2.0)
    monkeypatch.setattr(
        ms,
        "conditional_vendi",
        lambda pp: {"mean_within_vendi": 2.0, "per_prompt": [2.0], "n_prompts": 1},
    )
    monkeypatch.setattr(
        ms,
        "novelty_vs_corpus",
        lambda texts, corpus, **k: {"per_candidate": [], "flags": [], "mean_novelty": 1.0},
    )


def test_texts_only_blocks() -> None:
    report = metric_suite_report(TEXTS)
    assert set(report.keys()) == {"novelty_bench", "salad_rate", "canaries"}


def test_north_star_present() -> None:
    report = metric_suite_report(TEXTS)
    assert report["novelty_bench"]["n_distinct"] == 2


def test_salad_rate_is_float() -> None:
    report = metric_suite_report(TEXTS)
    assert isinstance(report["salad_rate"], float)


def test_per_prompt_adds_conditional_vendi() -> None:
    report = metric_suite_report(TEXTS, per_prompt=[["a", "b"], ["c", "d"]])
    assert "conditional_vendi" in report
    assert report["conditional_vendi"]["mean_within_vendi"] == 2.0


def test_corpus_adds_novelty_vs_corpus() -> None:
    report = metric_suite_report(TEXTS, corpus=["old idea"])
    assert "novelty_vs_corpus" in report
    assert report["novelty_vs_corpus"]["mean_novelty"] == 1.0


def test_canaries_present_and_labelled() -> None:
    report = metric_suite_report(TEXTS)
    canaries = report["canaries"]
    assert {"vendi_score", "distinct2", "self_bleu"} <= set(canaries.keys())
    assert "never" in canaries["label"].lower()


def test_no_optional_blocks_without_inputs() -> None:
    report = metric_suite_report(TEXTS)
    assert "conditional_vendi" not in report
    assert "novelty_vs_corpus" not in report
