"""Contrastive Activation Addition (CAA) steering vectors.

`SteeringVector` holds per-layer residual-stream directions; `derive_from_pairs`
computes them by averaging last-token activation differences between paired
"creative" and "generic" prompt framings.

Heavy deps (`torch`, `transformers`, `safetensors`) are imported lazily inside
function bodies so the no-steering CLI path stays fast.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DERIVATION_VERSION = 1

CREATIVE_SYSTEM = (
    "You are a wildly inventive brainstorming partner. Embrace surprising "
    "ideas, unconventional analogies, and creative leaps. Quality over "
    "consensus."
)
GENERIC_SYSTEM = (
    "You are a careful, conventional assistant. Give the most standard, "
    "expected answer. Avoid speculation or unusual ideas."
)


@dataclass
class SteeringVector:
    """Per-layer steering directions plus provenance.

    `vectors[layer_idx]` is a 1-D `torch.Tensor` of length `hidden_dim`.
    """

    base_model_id: str
    layers: tuple[int, ...]
    hidden_dim: int
    n_contrast_pairs: int
    derived_at: str
    derivation_version: int = DERIVATION_VERSION
    vectors: dict[int, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: Path) -> None:
        from safetensors.torch import save_file

        path.mkdir(parents=True, exist_ok=True)
        tensors = {f"layer_{idx}": self.vectors[idx] for idx in self.layers}
        save_file(tensors, str(path / "vector.safetensors"))
        manifest = {
            "base_model_id": self.base_model_id,
            "layers": list(self.layers),
            "hidden_dim": self.hidden_dim,
            "n_contrast_pairs": self.n_contrast_pairs,
            "derived_at": self.derived_at,
            "derivation_version": self.derivation_version,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, expected_base_model_id: str | None = None) -> "SteeringVector":
        from safetensors.torch import load_file

        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"No manifest.json under {path}")
        manifest = json.loads(manifest_path.read_text())

        base_model_id = manifest["base_model_id"]
        if expected_base_model_id is not None and expected_base_model_id != base_model_id:
            raise ValueError(
                f"Steering vector base_model_id {base_model_id!r} does not match "
                f"consumer base_model_id {expected_base_model_id!r}"
            )

        layers = tuple(manifest["layers"])
        tensors = load_file(str(path / "vector.safetensors"))
        vectors: dict[int, Any] = {}
        for idx in layers:
            key = f"layer_{idx}"
            if key not in tensors:
                raise ValueError(f"Missing layer tensor {key!r} in {path}")
            vectors[idx] = tensors[key]

        return cls(
            base_model_id=base_model_id,
            layers=layers,
            hidden_dim=int(manifest["hidden_dim"]),
            n_contrast_pairs=int(manifest["n_contrast_pairs"]),
            derived_at=manifest["derived_at"],
            derivation_version=int(manifest.get("derivation_version", DERIVATION_VERSION)),
            vectors=vectors,
        )


# ----------------------------------------------------------------------
# Derivation


def render_contrast_pair(prompt: str, tokenizer: Any) -> tuple[str, str]:
    """Apply the chat template to (creative, generic) framings of `prompt`.

    Both renderings end at the assistant turn boundary, matching how
    `HFGenerator` renders prompts at inference time. Stays pure: no model call.
    """
    creative = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CREATIVE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    generic = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": GENERIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return creative, generic


def _last_token_hidden(model: Any, tokenizer: Any, text: str, *, device: str | None) -> Any:
    """Forward `text`, return list of per-layer last-token hidden states.

    Returns the model's full `hidden_states` tuple unchanged so the caller can
    pick layers. Uses `output_hidden_states=True` and disables gradients.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    if device is not None and hasattr(inputs, "to"):
        try:
            inputs = inputs.to(device)
        except Exception:
            pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # `hidden_states` is a tuple of (n_layers + 1) tensors of shape
    # (batch, seq, hidden_dim) — index 0 is the embedding output.
    return outputs.hidden_states


def derive_from_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    base_model_id: str,
    layers: Sequence[int],
    derived_at: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | None = None,
    dtype: str = "bf16",
) -> SteeringVector:
    """Compute mean of (creative - generic) last-token hidden states per layer.

    `model` and `tokenizer` are accepted as injection points so tests can pass
    stubs; production callers pass `None` and we lazy-load the same Qwen2.5-7B
    weights `HFGenerator` uses.
    """
    if not pairs:
        raise ValueError("derive_from_pairs requires at least one (creative, generic) pair")

    layers = tuple(sorted(set(int(layer) for layer in layers)))
    if not layers:
        raise ValueError("derive_from_pairs requires at least one layer")

    if model is None or tokenizer is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if model is None:
            torch_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[dtype]
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch_dtype
            )
            if device is not None and hasattr(model, "to"):
                model = model.to(device)

    accumulators: dict[int, Any] = {}
    hidden_dim: int | None = None

    for creative_text, generic_text in pairs:
        c_hidden = _last_token_hidden(model, tokenizer, creative_text, device=device)
        g_hidden = _last_token_hidden(model, tokenizer, generic_text, device=device)
        for layer_idx in layers:
            c_last = c_hidden[layer_idx][0, -1, :]
            g_last = g_hidden[layer_idx][0, -1, :]
            delta = (c_last - g_last).detach().to("cpu").float()
            if hidden_dim is None:
                hidden_dim = int(delta.shape[-1])
            if layer_idx not in accumulators:
                accumulators[layer_idx] = delta
            else:
                accumulators[layer_idx] = accumulators[layer_idx] + delta

    if hidden_dim is None:
        raise RuntimeError("derive_from_pairs produced no hidden states")

    n = len(pairs)
    vectors = {idx: accumulators[idx] / n for idx in layers}

    return SteeringVector(
        base_model_id=base_model_id,
        layers=layers,
        hidden_dim=hidden_dim,
        n_contrast_pairs=n,
        derived_at=derived_at,
        vectors=vectors,
    )
