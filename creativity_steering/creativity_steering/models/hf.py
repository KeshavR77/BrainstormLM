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
    SamplingPreset,
)
from creativity_steering.models.backend import Injection, ModelBackend

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def build_generate_kwargs(
    sampling: SamplingPreset, *, max_new_tokens: int, pad_token_id: int | None
) -> dict:
    """Translate a ``SamplingPreset`` into ``model.generate`` kwargs.

    Greedy presets set ``do_sample=False`` and forward no sampling knobs. For sampling
    presets, each knob is forwarded only when it is *active* (``top_p < 1``, ``top_k > 0``,
    ``min_p > 0``, ``repetition_penalty != 1``); a no-op knob is omitted so the model's own
    generation default applies — that is what keeps a bare ``SamplingPreset(temperature=t)``
    behaviourally identical to the previous temperature-only generate call.

    (``min_p`` needs transformers >= 4.41; it is only emitted when a preset sets it.)
    """
    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "do_sample": sampling.do_sample,
    }
    if sampling.do_sample:
        kwargs["temperature"] = max(sampling.temperature, 1e-5)
        if sampling.top_p < 1.0:
            kwargs["top_p"] = sampling.top_p
        if sampling.top_k > 0:
            kwargs["top_k"] = sampling.top_k
        if sampling.min_p > 0.0:
            kwargs["min_p"] = sampling.min_p
        if sampling.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = sampling.repetition_penalty
    return kwargs


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
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Authenticate downloads: unauthenticated HF requests are rate-limited, which manifests as
        # a download that stalls at a fixed byte count (the RunPod-Lessons Xet finalize/rate-limit
        # hang). A token also unlocks the faster transfer path. HF_HUB_OFFLINE=1 (respected by
        # transformers) forces cache-only loads once the weights are present.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

        self._device = self._resolve_device()
        torch_dtype = getattr(torch, _DTYPES[self._dtype])
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True, token=token
        )
        # device_map streams shards straight to the target device (peak CPU RAM ~= one shard),
        # which a 28GB+ bf16 model needs under a constrained container memory cgroup; the old
        # load-to-CPU-then-.to() pattern OOMs there. `dtype=` (not the deprecated `torch_dtype=`)
        # so the compute dtype is honored on transformers 5.x.
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, dtype=torch_dtype, trust_remote_code=True,
            device_map=self._device, token=token,
        )
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
        sampling: SamplingPreset | None = None,
    ) -> str:
        import torch

        self._ensure_loaded()
        torch.manual_seed(seed)
        if sampling is None:
            sampling = SamplingPreset(temperature=temperature)
        gen_kwargs = build_generate_kwargs(
            sampling, max_new_tokens=max_new_tokens, pad_token_id=self._tokenizer.eos_token_id
        )
        inputs = self._tokenizer(
            self._prompt_text(prompt, system), return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
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

    # ------------------------------------------------------------------ steering (batch)
    def generate_steered_batch(
        self,
        prompt: str,
        *,
        dirs: np.ndarray,
        layer: int,
        alpha: float,
        rescale: bool = True,
        system: str | None = None,
        max_new_tokens: int = 256,
        seed: int = 0,
        sampling: SamplingPreset | None = None,
    ) -> list[str]:
        """Generate one steered continuation per row of ``dirs`` for a single prompt.

        The prompt is replicated ``B = dirs.shape[0]`` times into a batch; batch row ``i`` is
        steered by ``dirs[i]`` via a per-sample forward hook on decoder block ``layer`` (the
        STARS-style "a batch fans across directions" path). ``dirs`` rows are typically an
        orthonormal frame (``stiefel_basis``) or an orthogonalized archetype set. Returns ``B``
        decoded continuations (prompt stripped). Defaults to greedy decoding so the only source of
        variation across rows is the per-sample direction.
        """
        import torch

        self._ensure_loaded()
        dirs_np = np.asarray(dirs, dtype=np.float32)
        if dirs_np.ndim != 2:
            raise ValueError(f"dirs must be 2-D (batch, hidden_dim); got shape {dirs_np.shape}")
        batch_size, hidden = dirs_np.shape
        if hidden != self.hidden_dim:
            raise ValueError(
                f"dirs hidden_dim {hidden} does not match model hidden_dim {self.hidden_dim}"
            )
        blocks = self._decoder_layers()
        if not 0 <= layer < len(blocks):
            raise ValueError(f"injection layer {layer} out of range [0, {len(blocks)})")

        torch.manual_seed(seed)
        if sampling is None:
            sampling = SamplingPreset(temperature=0.0)  # greedy isolates the steering effect
        gen_kwargs = build_generate_kwargs(
            sampling, max_new_tokens=max_new_tokens, pad_token_id=self._tokenizer.eos_token_id
        )

        enc = self._tokenizer(self._prompt_text(prompt, system), return_tensors="pt").to(self._device)
        prompt_len = int(enc["input_ids"].shape[1])
        gen_inputs = {"input_ids": enc["input_ids"].repeat(batch_size, 1)}
        if "attention_mask" in enc:
            gen_inputs["attention_mask"] = enc["attention_mask"].repeat(batch_size, 1)

        handle = blocks[layer].register_forward_hook(
            _make_batch_hook(dirs_np, alpha=alpha, rescale=rescale)
        )
        try:
            with torch.no_grad():
                out = self._model.generate(**gen_inputs, **gen_kwargs)
        finally:
            try:
                handle.remove()
            except Exception:
                pass
        new = out[:, prompt_len:]
        return [self._tokenizer.decode(row, skip_special_tokens=True) for row in new]


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


def _make_batch_hook(dirs, *, alpha: float, rescale: bool):
    """Forward hook adding a *per-sample* steering direction to a decoder block's output.

    Batch row ``i`` receives ``dirs[i]`` scaled with the same ``(v/‖v‖)·‖h‖·α`` rescale
    convention as ``_make_hook``. The per-sample math is delegated to ``inject_orthogonal_batch``,
    which broadcasts over the sequence axis — so the hook is correct at both the prefill step
    ``(B,T,H)`` and every incremental decode step ``(B,1,H)``. A zero direction row leaves that
    sample unchanged (no NaN), matching ``_make_hook``'s zero-vector tolerance.
    """
    from creativity_steering.patching.orthogonal import inject_orthogonal_batch

    def hook(_module, _inp, output):  # noqa: ANN001 — torch hook signature
        hidden = output[0] if isinstance(output, tuple) else output
        tail = output[1:] if isinstance(output, tuple) else ()
        new_hidden = inject_orthogonal_batch(hidden, dirs, alpha, rescale=rescale)
        return (new_hidden, *tail) if tail else new_hidden

    return hook
