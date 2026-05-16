from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def last_token_hidden_at_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    layer_index: int,
    *,
    dtype_out: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Last-token hidden state at decoder ``layer_index`` (0-based) for ``text``.

    Uses ``output_hidden_states=True``. For HuggingFace causal LMs, ``hidden_states[0]`` is
    embeddings and ``hidden_states[layer_index + 1]`` is the output of decoder layer
    ``layer_index``.

    Returns a 1D tensor on CPU with shape ``[hidden_size]``.
    """
    if layer_index < 0:
        raise ValueError(f"layer_index must be >= 0, got {layer_index}")

    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = outputs.hidden_states
    if hs is None:
        raise RuntimeError("Model did not return hidden_states (output_hidden_states ignored?).")
    idx = layer_index + 1
    if idx >= len(hs):
        n_dec = len(hs) - 1
        raise ValueError(f"layer_index {layer_index} out of range for this model (num_decoder_layers≈{n_dec}).")

    h = hs[idx][0, -1, :].detach().to(dtype=dtype_out).cpu().contiguous()
    return h


def normalize_hidden(vec: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a 1D hidden vector (CPU or device)."""
    n = torch.linalg.vector_norm(vec.float())
    if float(n) < eps:
        return vec * 0.0
    return (vec.float() / n).to(dtype=vec.dtype)
