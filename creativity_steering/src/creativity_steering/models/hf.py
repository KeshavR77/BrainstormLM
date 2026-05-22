"""Real transformers backend — Qwen2.5 family (0.5B locally, 14B on RunPod).

Implements the same ``ModelBackend`` interface as the mock, so the mock-tested derivation /
analysis / patching code runs unchanged on the real model. Heavy deps (torch, transformers)
are imported lazily so importing this module stays cheap and offline-friendly.

Layer convention (matched extraction & injection): "layer ``L``" is the *output* of decoder
block ``L`` (``model.model.layers[L]``). At capture time that is ``hidden_states[L+1]``
(index 0 of ``hidden_states`` is the embedding output); at injection time it is the residual
a forward hook on block ``L`` modifies. Extraction and injection therefore hit the same
residual-stream location — the conformity paper's design, and a fix to the off-by-one in the
repo's current ``extract_14b_vectors`` (which reads ``hidden_states[L]`` but injects at block L).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from creativity_steering.config import (
    CAPTURE_ASSISTANT_START,
    CAPTURE_LAST_RESPONSE,
    CAPTURE_MEAN_RESPONSE,
)
from creativity_steering.models.backend import Injection, ModelBackend

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> None:
        self._model_id = model_id
        self._dtype = dtype
        self._device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ loading
    def _resolve_device(self) -> str:
        import torch

        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._resolve_device()
        torch_dtype = getattr(torch, _DTYPES[self._dtype])
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(self._device)
        self._model.eval()

    def _decoder_layers(self):
        inner = getattr(self._model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if layers is None:
            raise RuntimeError(f"cannot locate decoder blocks on {type(self._model).__name__}")
        return list(layers)

    # ------------------------------------------------------------------ metadata
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def num_layers(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.num_hidden_layers)

    @property
    def hidden_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    # ------------------------------------------------------------------ helpers
    def _messages(self, prompt: str, system: str | None) -> list[dict]:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _prompt_text(self, prompt: str, system: str | None) -> str:
        return self._tokenizer.apply_chat_template(
            self._messages(prompt, system), tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> str:
        import torch

        self._ensure_loaded()
        torch.manual_seed(seed)
        inputs = self._tokenizer(
            self._prompt_text(prompt, system), return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new, skip_special_tokens=True)

    # ------------------------------------------------------------------ capture
    def capture_activations(
        self,
        prompt: str,
        response: str,
        *,
        system: str | None = None,
        positions: Sequence[str],
    ) -> dict[str, np.ndarray]:
        import torch

        self._ensure_loaded()
        prompt_text = self._prompt_text(prompt, system)
        prompt_len = self._tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        full = self._tokenizer(prompt_text + response, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**full, output_hidden_states=True)
        hs = out.hidden_states  # tuple len num_layers+1
        n_layers = self.num_layers
        start_idx = max(prompt_len - 1, 0)  # assistant-start = last prompt token

        result: dict[str, np.ndarray] = {}
        for pos in positions:
            rows = []
            for L in range(n_layers):
                layer_hs = hs[L + 1][0]  # (seq, hidden)
                if pos == CAPTURE_LAST_RESPONSE:
                    v = layer_hs[-1, :]
                elif pos == CAPTURE_ASSISTANT_START:
                    v = layer_hs[start_idx, :]
                elif pos == CAPTURE_MEAN_RESPONSE:
                    span = layer_hs[prompt_len:, :]
                    v = span.mean(dim=0) if span.shape[0] > 0 else layer_hs[-1, :]
                else:
                    raise ValueError(f"unknown capture position {pos!r}")
                rows.append(v.detach().to("cpu").float().numpy())
            result[pos] = np.stack(rows).astype(np.float32)
        return result

    # ------------------------------------------------------------------ steering
    def generate_steered(
        self,
        prompt: str,
        *,
        injections: Sequence[Injection],
        system: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> str:
        import torch

        self._ensure_loaded()
        blocks = self._decoder_layers()
        handles = []
        for inj in injections:
            if not 0 <= inj.layer < len(blocks):
                raise ValueError(f"injection layer {inj.layer} out of range")
            vec = torch.tensor(np.asarray(inj.vector, dtype=np.float32), device=self._device)
            handles.append(
                blocks[inj.layer].register_forward_hook(
                    _make_hook(vec, alpha=inj.alpha, rescale=inj.rescale)
                )
            )
        try:
            return self.generate(
                prompt,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass


def _make_hook(vector, *, alpha: float, rescale: bool):
    """Forward hook adding ``alpha * vector`` to a decoder block's residual output.

    With ``rescale=True`` the addition is matched to each token's activation norm, so the
    same ``alpha`` transfers across model sizes (same convention as the repo's steered.py).
    """

    def hook(_module, _inp, output):  # noqa: ANN001 — torch hook signature
        hidden = output[0] if isinstance(output, tuple) else output
        tail = output[1:] if isinstance(output, tuple) else ()
        v = vector.to(dtype=hidden.dtype, device=hidden.device)
        vnorm = v.norm()
        if float(vnorm) <= 0.0:
            return output
        if rescale:
            h_norm = hidden.norm(dim=-1, keepdim=True)
            scaled = (v / vnorm) * h_norm * alpha
        else:
            scaled = v * alpha
        new_hidden = hidden + scaled
        return (new_hidden, *tail) if tail else new_hidden

    return hook
