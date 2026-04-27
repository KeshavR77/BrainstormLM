"""Ablation harness: 4-arm plumbing with stub generators and scorer."""

from __future__ import annotations

from pathlib import Path

from brainstormlm.eval.harness import (
    ARMS,
    AblationRow,
    format_summary_table,
    row_to_dict,
    run_ablation,
    summarize,
)
from brainstormlm.feedback.evaluator import EvaluatorResult


class _FakeGen:
    """Returns canned responses keyed off (arm-tag, prompt). Records calls."""

    def __init__(self, tag: str, score_table: dict[str, tuple[int, int]]) -> None:
        self.tag = tag
        self.calls: list[dict] = []
        self.score_table = score_table

    def generate_with_meta(
        self, prompt: str, *, system: str | None, temperature: float, max_tokens: int
    ) -> tuple[str, bool]:
        self.calls.append(
            {"prompt": prompt, "system": system, "temperature": temperature, "max_tokens": max_tokens}
        )
        c, lg = self.score_table[prompt]
        return f"{self.tag}::{prompt}::c{c}l{lg}", False


def _score_fn(prompt: str, response: str) -> EvaluatorResult:
    # Decode "c{c}l{lg}" tail of the canned response.
    tail = response.rsplit("::", 1)[-1]
    c = int(tail.split("c")[1].split("l")[0])
    logic = int(tail.split("l")[1])
    return EvaluatorResult(creativity=c, logic=logic, rationale="stub", failure_modes=[])


def _held_out() -> list[tuple[str, str]]:
    return [
        ("science.physics", "p1"),
        ("art.visual", "p2"),
        ("tech.ai", "p3"),
    ]


def test_run_ablation_no_memory_yields_two_arms_only() -> None:
    base = _FakeGen("base", {"p1": (3, 3), "p2": (3, 3), "p3": (3, 3)})
    trained = _FakeGen("trained", {"p1": (4, 4), "p2": (4, 4), "p3": (4, 4)})
    rows = run_ablation(
        _held_out(),
        base_generator=base,
        trained_generator=trained,
        score_fn=_score_fn,
        store=None,
    )
    arms = sorted({r.arm for r in rows})
    assert arms == ["base", "trained"]
    assert len(rows) == 6  # 2 arms × 3 prompts
    assert all(r.adapter_path is None for r in rows if r.arm == "base")


def test_run_ablation_with_memory_yields_four_arms(monkeypatch) -> None:
    base = _FakeGen("base", {"p1": (3, 3), "p2": (3, 3), "p3": (3, 3)})
    trained = _FakeGen("trained", {"p1": (4, 4), "p2": (4, 4), "p3": (4, 4)})

    # Stub the memory layer so we don't need a real LessonStore.
    sentinel_lessons: list = []

    class _StubStore:
        pass

    def _fake_retrieve(store, prompt, *, k_success, k_failure):
        return sentinel_lessons

    def _fake_augmented(base_system, lessons):
        return "MEMORY-SYS"

    monkeypatch.setattr("brainstormlm.eval.harness.retrieve", _fake_retrieve)
    monkeypatch.setattr("brainstormlm.eval.harness.augmented_system", _fake_augmented)

    rows = run_ablation(
        _held_out(),
        base_generator=base,
        trained_generator=trained,
        score_fn=_score_fn,
        store=_StubStore(),
        adapter_path=Path("data/adapters/run-x"),
    )
    assert sorted({r.arm for r in rows}) == sorted(list(ARMS))
    assert len(rows) == 12  # 4 arms × 3 prompts

    mem_rows = [r for r in rows if r.arm.endswith("+mem")]
    base_arm_systems = [c["system"] for c in base.calls]
    # Memory arm passes the augmented system; the base arm passes None.
    assert any(c["system"] == "MEMORY-SYS" for c in base.calls)
    assert any(s is None for s in base_arm_systems)
    assert all(r.memory_context_ids == [] for r in mem_rows)  # sentinel was empty

    trained_rows = [r for r in rows if r.arm.startswith("trained")]
    assert all(r.adapter_path == "data/adapters/run-x" for r in trained_rows)


def test_summarize_picks_correct_means_and_low_p() -> None:
    rows = []
    # base: every prompt scores 6
    for d, p in _held_out():
        rows.append(AblationRow(arm="base", domain=d, prompt=p, response="r", creativity=3, logic=3, total=6))
    # trained: every prompt scores 8 — clear win
    for d, p in _held_out():
        rows.append(AblationRow(arm="trained", domain=d, prompt=p, response="r", creativity=4, logic=4, total=8))

    summaries = summarize(rows, n_iterations=2000, seed=7)
    by_arm = {s.arm: s for s in summaries}
    assert by_arm["base"].mean_total == 6.0
    assert by_arm["base"].mean_delta_vs_base is None
    assert by_arm["trained"].mean_total == 8.0
    assert by_arm["trained"].mean_delta_vs_base == 2.0
    # 3 prompts, all +2 → permutation test gives a small but non-zero p; tolerance is generous.
    assert by_arm["trained"].p_value is not None
    assert by_arm["trained"].p_value < 0.5


def test_format_summary_table_includes_all_rows() -> None:
    rows = [
        AblationRow(arm="base", domain="d", prompt="p", response="r", creativity=3, logic=3, total=6),
        AblationRow(arm="trained", domain="d", prompt="p", response="r", creativity=4, logic=4, total=8),
    ]
    summaries = summarize(rows, n_iterations=500, seed=0)
    text = format_summary_table(summaries)
    assert "base" in text and "trained" in text
    assert "creativity" in text and "logic" in text


def test_row_to_dict_round_trip() -> None:
    r = AblationRow(arm="base", domain="d", prompt="p", response="r", creativity=3, logic=3, total=6)
    d = row_to_dict(r)
    assert d["arm"] == "base"
    assert d["total"] == 6
    assert d["adapter_path"] is None
