"""DPO + LoRA training wiring.

Pure-Python builders (`DPOConfig`, `build_lora_config`, `dataset_from_pairs`)
are testable without a GPU. ``build_dpo_trainer`` is exercised only via the
Modal job in ``scripts/train_dpo.py``; per plan-04 we deliberately do not
unit-test it because it requires a real model + tokenizer.

All heavy imports (`peft`, `trl`, `datasets`) are deferred to the bodies of
the builder functions so the project's CLI startup stays fast on the
no-adapter path.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from brainstormlm.training.preferences import PreferencePair

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOTrainer


class DPOConfig(BaseModel):
    """Project-level run config for the LoRA + DPO training run.

    Distinct from ``trl.DPOConfig`` (the trainer-level config we build at
    ``build_dpo_trainer`` time): this is the typed surface our scripts and
    tests work with.
    """

    model_config = ConfigDict(extra="forbid")

    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_out_dir: Path
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    max_prompt_length: int = 384
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42


def build_lora_config(cfg: DPOConfig) -> LoraConfig:
    """Pure-Python LoRA config builder (no GPU; lazy import of peft)."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def dataset_from_pairs(
    pairs: list[PreferencePair],
    tokenizer: Any,
) -> Dataset:
    """Convert PreferencePairs to a TRL-ready ``datasets.Dataset``.

    Columns: ``prompt``, ``chosen``, ``rejected``. The ``prompt`` column gets
    the chat template applied via ``tokenizer.apply_chat_template`` so both
    arms share identical system+user framing; the chosen/rejected columns
    remain the raw assistant responses.
    """
    from datasets import Dataset

    rows: list[dict[str, str]] = []
    for p in pairs:
        messages = [{"role": "user", "content": p.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "prompt": formatted_prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
        )
    return Dataset.from_list(rows)


def build_dpo_trainer(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    dataset: Any,
    cfg: DPOConfig,
) -> DPOTrainer:
    """Wires up ``trl.DPOTrainer`` with our LoRA + project DPOConfig.

    Imports trl + peft lazily. Uses ``inspect.signature`` to filter kwargs we
    pass to ``trl.DPOConfig`` so the call survives kwarg renames between TRL
    versions (notably ``max_prompt_length`` was dropped in TRL 1.x).

    Returns the trainer; the caller is responsible for invoking ``.train()``.
    """
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    # Build only the kwargs that this TRL version actually accepts.
    candidate: dict[str, Any] = {
        "output_dir": str(cfg.adapter_out_dir),
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "beta": cfg.beta,
        "max_length": cfg.max_length,
        "max_prompt_length": cfg.max_prompt_length,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "seed": cfg.seed,
    }
    accepted = set(inspect.signature(TRLDPOConfig.__init__).parameters)
    trl_kwargs = {k: v for k, v in candidate.items() if k in accepted}
    trl_cfg = TRLDPOConfig(**trl_kwargs)

    lora = build_lora_config(cfg)
    return DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trl_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
