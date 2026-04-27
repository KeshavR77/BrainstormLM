"""Unit tests for DPO config + dataset construction (no GPU, no training).

Note: ``build_dpo_trainer`` is intentionally not unit-tested here. It requires
a real model and tokenizer; full integration is exercised only via the Modal
job in ``scripts/train_dpo.py``. Per plan-04: "the actual `.train()` call is
exercised only via the Modal job."
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from brainstormlm.training import DPOConfig
from brainstormlm.training.dpo import build_lora_config, dataset_from_pairs
from brainstormlm.training.preferences import PreferencePair


def _make_pair(idx: int) -> PreferencePair:
    return PreferencePair(
        id=f"{idx:064x}",
        timestamp="2026-04-25T00:00:00Z",
        domain="science.physics",
        prompt=f"Prompt number {idx}.",
        chosen=f"Chosen answer {idx}.",
        rejected=f"Rejected answer {idx}.",
        chosen_score=8.0,
        rejected_score=4.0,
        score_gap=4.0,
        chosen_temperature=1.0,
        rejected_temperature=0.7,
        generator_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        evaluator_model="gpt-4o-mini",
        rubric_version=1,
        sibling_run_id="run-test",
    )


class _StubTokenizer:
    """Records every apply_chat_template call so tests can assert on them."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls.append(list(messages))
        return f"PROMPT::{messages[0]['content']}"


# ---------------------------------------------------------------------------
# DPOConfig
# ---------------------------------------------------------------------------


def test_dpo_config_defaults(tmp_path: Path) -> None:
    cfg = DPOConfig(adapter_out_dir=tmp_path)
    assert cfg.base_model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert cfg.beta == 0.1
    assert cfg.learning_rate == 5e-5
    assert cfg.num_train_epochs == 3
    assert cfg.per_device_batch_size == 2
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.max_length == 1024
    assert cfg.max_prompt_length == 384
    assert cfg.bf16 is True
    assert cfg.gradient_checkpointing is True
    assert cfg.seed == 42


def test_dpo_config_extra_forbidden(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DPOConfig(adapter_out_dir=tmp_path, mystery_field="nope")


def test_dpo_config_overrides(tmp_path: Path) -> None:
    cfg = DPOConfig(
        adapter_out_dir=tmp_path,
        lora_rank=32,
        beta=0.2,
        num_train_epochs=1,
    )
    assert cfg.lora_rank == 32
    assert cfg.beta == 0.2
    assert cfg.num_train_epochs == 1


# ---------------------------------------------------------------------------
# build_lora_config
# ---------------------------------------------------------------------------


def test_build_lora_config_returns_lora_with_expected_rank(tmp_path: Path) -> None:
    cfg = DPOConfig(adapter_out_dir=tmp_path)
    lora = build_lora_config(cfg)
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.lora_dropout == 0.05
    # peft stores `target_modules` as a set, so compare membership rather than order.
    assert set(lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert lora.bias == "none"
    assert lora.task_type == "CAUSAL_LM"


def test_build_lora_config_respects_overrides(tmp_path: Path) -> None:
    cfg = DPOConfig(
        adapter_out_dir=tmp_path,
        lora_rank=32,
        lora_alpha=64,
        target_modules=("q_proj", "v_proj"),
    )
    lora = build_lora_config(cfg)
    assert lora.r == 32
    assert lora.lora_alpha == 64
    assert set(lora.target_modules) == {"q_proj", "v_proj"}


# ---------------------------------------------------------------------------
# dataset_from_pairs
# ---------------------------------------------------------------------------


def test_dataset_from_pairs_columns_and_length() -> None:
    pairs = [_make_pair(i) for i in range(3)]
    tok = _StubTokenizer()

    ds = dataset_from_pairs(pairs, tok)
    assert len(ds) == 3
    assert sorted(ds.column_names) == ["chosen", "prompt", "rejected"]

    rows = list(ds)
    for i, row in enumerate(rows):
        assert row["prompt"] == f"PROMPT::Prompt number {i}."
        assert row["chosen"] == f"Chosen answer {i}."
        assert row["rejected"] == f"Rejected answer {i}."


def test_dataset_from_pairs_applies_chat_template_per_row() -> None:
    pairs = [_make_pair(i) for i in range(4)]
    tok = _StubTokenizer()

    dataset_from_pairs(pairs, tok)

    assert len(tok.calls) == len(pairs)
    # Each call should be a single user-role message containing the raw prompt.
    for call, p in zip(tok.calls, pairs):
        assert len(call) == 1
        assert call[0] == {"role": "user", "content": p.prompt}


def test_dataset_from_pairs_empty_input_yields_empty_dataset() -> None:
    tok = _StubTokenizer()
    ds = dataset_from_pairs([], tok)
    assert len(ds) == 0
    assert tok.calls == []
