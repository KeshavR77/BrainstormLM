from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel


def decoder_layers(model: PreTrainedModel):
    """
    Return the ``ModuleList`` of decoder blocks for hooking.

    Supports:

    * Llama / Mistral / Qwen2-style: ``model.model.layers``
    * Gemma 3 / Gemma 4 multimodal (``*ForConditionalGeneration``): ``model.model.language_model.layers``
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("Expected a model with `.model` (e.g. LlamaForCausalLM, Gemma4ForConditionalGeneration).")
    layers = _decoder_layers_from_inner(inner)
    if layers is None:
        raise ValueError(
            "Unsupported architecture: expected `.model.layers` or `.model.language_model.layers` "
            "(Llama-like or Gemma 3/4 multimodal text stack)."
        )
    return layers


def _decoder_layers_from_inner(inner: Any):
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return layers
    lm = getattr(inner, "language_model", None)
    if lm is None:
        return None
    return getattr(lm, "layers", None)
