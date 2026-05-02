"""HFGenerator — local transformers + optional PEFT adapter behind the Generator protocol.

Heavy deps (torch / transformers / peft) are imported lazily inside methods so
the no-adapter CLI path stays fast and `from brainstormlm.model import HFGenerator`
remains cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.config import get_settings


class HFGenerator:
    """Local transformers generator with optional PEFT adapter.

    Implements the `model.interface.Generator` protocol. Cache hits short-
    circuit before any model load — `brainstorm ask` against a warm cache
    stays fast even with `--adapter` set.
    """

    def __init__(
        self,
        *,
        base_model_id: str | None = None,
        adapter_path: Path | None = None,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> None:
        settings = get_settings()
        self.base_model_id = base_model_id or settings.hf_base_model
        self.adapter_path = adapter_path
        # Explicit `device=` arg wins; otherwise fall back to BRAINSTORMLM_HF_DEVICE
        # so a user on Apple Silicon can pin to "cpu" to dodge the 4GB MPS-NDArray
        # limit that Qwen2.5-7B trips during attention/KV-cache allocations.
        self._device_override = device if device is not None else settings.hf_device
        self.dtype = dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Generator protocol

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        text, _ = self.generate_with_meta(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        return text

    def generate_with_meta(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> tuple[str, bool]:
        key = cache_key(
            "hf",
            self.base_model_id,
            str(self.adapter_path) if self.adapter_path is not None else "no-adapter",
            system or "",
            prompt,
            f"{temperature:.4f}",
            str(max_tokens),
        )
        if use_cache:
            cached = read_cache(key)
            if cached is not None:
                return cached, True

        # Cache miss — heavy path. Load weights lazily, then generate.
        self._ensure_loaded()
        text = self._run_generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        write_cache(key, text)
        return text, False

    # ------------------------------------------------------------------
    # Internals

    def _ensure_loaded(self) -> None:
        """Lazy-load tokenizer + model (and optional PEFT adapter). Idempotent."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device_override or self._auto_device(torch)
        torch_dtype = self._torch_dtype(torch, self.dtype)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch_dtype,
        )

        if self.adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter_path))

        if hasattr(model, "to"):
            try:
                model = model.to(device)
            except Exception:
                # Some stubs in tests don't actually move; that's fine.
                pass

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _run_generate(
        self,
        *,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        tokenizer = self._tokenizer
        model = self._model

        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(rendered, return_tensors="pt")
        # Move tensors to the model's device when available.
        device = getattr(model, "device", None) or self._device
        if device is not None and hasattr(inputs, "to"):
            try:
                inputs = inputs.to(device)
            except Exception:
                pass

        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        prompt_len = self._sequence_length(input_ids)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        attention_mask = (
            inputs.get("attention_mask") if isinstance(inputs, dict) else None
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        output_ids = model.generate(**gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _auto_device(torch_mod: Any) -> str:
        if getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available():
            return "cuda"
        backends = getattr(torch_mod, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _torch_dtype(torch_mod: Any, dtype: str) -> Any:
        mapping = {
            "bf16": getattr(torch_mod, "bfloat16", None),
            "fp16": getattr(torch_mod, "float16", None),
            "fp32": getattr(torch_mod, "float32", None),
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        return mapping[dtype]

    @staticmethod
    def _sequence_length(input_ids: Any) -> int:
        # Support both tensor-like (.shape[-1]) and plain list-of-lists shapes.
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        try:
            return len(input_ids[0])
        except Exception:
            return len(input_ids)
