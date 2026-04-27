"""Offline preference learning: pair mining + DPO training."""

from brainstormlm.training.dpo import DPOConfig
from brainstormlm.training.preferences import (
    PreferencePair,
    append,
    pair_id,
    read_all,
)

__all__ = ["DPOConfig", "PreferencePair", "append", "pair_id", "read_all"]
