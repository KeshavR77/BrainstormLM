"""creativity_steering — DiffMeans creativity steering vectors at every layer.

Pipeline (mirrors the conformity-paper methodology, applied to creativity):

  1. derivation/  — generate responses to creative vs neutral framings, filter to an
     evaluator-verified subset, then DiffMeans at *every* layer.
  2. analysis/    — norm, layer-wise cosine geometry, and an activation-oracle probe to
     hypothesise which layers carry the creativity direction.
  3. patching/    — per-layer alpha sweep (Exp 1), best-vector-at-every-layer (Exp 2),
     and multi-vector combinations, each scored by the creativity evaluators.

Everything runs offline on a deterministic mock backend; the only change needed to run
the real Qwen2.5-14B on a RunPod GPU is the `model_id` / `backend` in the config.
"""

from creativity_steering.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
