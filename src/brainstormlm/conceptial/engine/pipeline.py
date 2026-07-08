from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:  # pragma: no cover - optional until transformers pin
    Gemma3ForConditionalGeneration = None  # type: ignore[misc, assignment]

try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:  # pragma: no cover - optional until transformers pin
    Gemma4ForConditionalGeneration = None  # type: ignore[misc, assignment]

from brainstormlm.conceptial.engine.config import SteeringConfig, SteeringStrategy, validate_layer_injections
from brainstormlm.conceptial.engine.device_utils import hf_single_device_map, resolve_load_torch_dtype
from brainstormlm.conceptial.engine.hooks import steering_context


def load_steering_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a steering checkpoint dict (``vector``, ``layer_index``, optional metadata).

    For multiple checkpoints keyed by layer, use :func:`load_steering_vectors_from_layer_injections`
    with ``SteeringConfig.layer_injections``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    try:
        bundle = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        bundle = torch.load(p, map_location=map_location)
    missing = {"vector", "layer_index"} - set(bundle.keys())
    if missing:
        raise KeyError(f"steering checkpoint missing keys: {sorted(missing)}")
    return bundle


def load_steering_vectors_from_layer_injections(
    config: SteeringConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Load every ``vector_path`` in ``config.layer_injections`` and return ``layer -> tensor`` (CPU).

    Validates each checkpoint's ``layer_index`` matches the injection's ``layer``.
    """
    validate_layer_injections(config.layer_injections)
    overrides = getattr(config, "steering_vector_overrides", None) or {}
    out: dict[int, torch.Tensor] = {}
    for inj in config.layer_injections:
        L = int(inj["layer"])
        if L in overrides:
            out[L] = overrides[L]
            continue
        bundle = load_steering_checkpoint(inj["vector_path"], map_location=map_location)
        if int(bundle["layer_index"]) != L:
            raise ValueError(
                f"Checkpoint {inj['vector_path']!r} has layer_index {bundle['layer_index']}, "
                f"expected {L} for layer_injections entry."
            )
        out[L] = bundle["vector"]
    return out


def apply_steering_checkpoint(engine: "SteeringEngine", bundle: dict[str, Any]) -> None:
    """Move ``bundle['vector']`` to the model device/dtype; V2 single-layer ``layer_alphas`` path."""
    engine.load()
    assert engine._model is not None
    p = next(engine._model.parameters())
    L = int(bundle["layer_index"])
    alpha = float(bundle.get("alpha", 1.0))
    engine.config.layer_injections = []
    engine.config.layer_alphas = {L: alpha}
    engine.config.strategy = SteeringStrategy.SINGLE
    v = bundle["vector"].to(device=p.device, dtype=p.dtype)
    engine.set_steering_vectors({L: v})


def apply_layer_injections(engine: "SteeringEngine", *, map_location: str | torch.device = "cpu") -> None:
    """
    Load all ``vector_path`` entries from ``engine.config.layer_injections``, move tensors to the
    model device/dtype, and set ``engine`` steering vectors. Clears legacy ``layer_alphas`` use for
    hooks (``layer_injections`` takes precedence). Sets ``strategy`` heuristically for logging.
    """
    validate_layer_injections(engine.config.layer_injections)
    engine.load()
    assert engine._model is not None
    p = next(engine._model.parameters())
    raw = load_steering_vectors_from_layer_injections(engine.config, map_location=map_location)
    vs = {L: t.to(device=p.device, dtype=p.dtype) for L, t in raw.items()}
    engine.set_steering_vectors(vs)
    engine.config.layer_alphas = {int(i["layer"]): float(i["start_alpha"]) for i in engine.config.layer_injections}
    has_fade = any(int(i["decay_tokens"]) > 0 for i in engine.config.layer_injections)
    multi = len(engine.config.layer_injections) > 1
    if multi and has_fade:
        engine.config.strategy = SteeringStrategy.MULTI
    elif has_fade:
        engine.config.strategy = SteeringStrategy.FADING
    elif multi:
        engine.config.strategy = SteeringStrategy.MULTI
    else:
        engine.config.strategy = SteeringStrategy.SINGLE


def model_id_uses_gemma4_stack(model_id: str) -> bool:
    """True for Gemma 4 multimodal instruct checkpoints that use ``Gemma4ForConditionalGeneration``."""
    s = str(model_id).lower()
    return "gemma-4" in s or "gemma4" in s


def model_id_uses_gemma3_stack(model_id: str) -> bool:
    """True for Gemma 3 / Gemma 3n instruct checkpoints that use ``Gemma3ForConditionalGeneration``."""
    s = str(model_id).lower()
    return "gemma-3" in s or "gemma3" in s


def _load_tokenizer_for_model(model_id: str, token: str | None) -> PreTrainedTokenizerBase:
    """
    Prefer ``AutoTokenizer``; fall back to ``AutoProcessor`` tokenizer for multimodal Gemma checkpoints.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as first_err:
        try:
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(model_id, token=token)
            inner = getattr(proc, "tokenizer", None)
            if inner is None:
                raise RuntimeError(f"No tokenizer on AutoProcessor for {model_id!r}") from first_err
            tok = inner
            if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            return tok  # type: ignore[return-value]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for {model_id!r} (AutoTokenizer and AutoProcessor both failed)."
            ) from e


def resolve_hf_token() -> str | None:
    """
    Gated models (e.g. Llama 3): set one of HF_TOKEN, HUGGING_FACE_HUB_TOKEN,
    or LLAMA_ACCESS_TOKEN in the environment.
    """
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("LLAMA_ACCESS_TOKEN")
        or os.environ.get("LLAMA-ACCESS-TOKEN")
    )


class SteeringEngine:
    """Load model/tokenizer, extract v_trip, run generation with optional steering."""

    def __init__(self, config: SteeringConfig | None = None, token: str | None = None):
        self.config = config or SteeringConfig()
        self._token = token if token is not None else resolve_hf_token()
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._steering_vectors: dict[int, torch.Tensor] | None = None
        self._latent_registry: dict[str, dict[int, torch.Tensor]] | None = None
        #: Filled by ghost/speculative generation paths for API / logging (else ``None``).
        self.last_generation_metrics: dict[str, Any] | None = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("Call load() before using the model.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before using the tokenizer.")
        return self._tokenizer

    def load(self) -> None:
        if self._model is not None:
            return
        dm: str | dict[str, Any] | None = self.config.device_map
        if dm is None:
            dm = hf_single_device_map(self.config.preferred_device)

        torch_dtype: torch.dtype | str
        if isinstance(dm, dict) and "" in dm:
            accel_key = dm[""]
            if accel_key == 0 or accel_key == "cuda":
                accel_name = "cuda"
            elif accel_key == "mps":
                accel_name = "mps"
            else:
                accel_name = "cpu"
            if self.config.torch_dtype == "auto":
                torch_dtype = resolve_load_torch_dtype("auto", accel_name)
            else:
                td = self.config.resolve_dtype()
                torch_dtype = td if isinstance(td, torch.dtype) else torch.float16
        else:
            td = self.config.resolve_dtype()
            torch_dtype = td if td != "auto" else torch.float16

        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": dm,
            "token": self._token,
            "low_cpu_mem_usage": True,
        }
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        mid = str(self.config.model_id)
        self._tokenizer = _load_tokenizer_for_model(mid, self._token)
        if model_id_uses_gemma4_stack(mid):
            if Gemma4ForConditionalGeneration is None:
                raise RuntimeError(
                    "This model_id looks like Gemma 4, but Gemma4ForConditionalGeneration is not available. "
                    "Upgrade transformers (see pyproject.toml pin)."
                )
            self._model = Gemma4ForConditionalGeneration.from_pretrained(mid, **kwargs)
        elif model_id_uses_gemma3_stack(mid):
            if Gemma3ForConditionalGeneration is None:
                raise RuntimeError(
                    "This model_id looks like Gemma 3, but Gemma3ForConditionalGeneration is not available. "
                    "Upgrade transformers (see pyproject.toml pin)."
                )
            self._model = Gemma3ForConditionalGeneration.from_pretrained(mid, **kwargs)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)
        p = next(self._model.parameters())
        print(
            f"corpius runtime: model loaded device={p.device} dtype={p.dtype} device_map={dm!r}",
            flush=True,
        )

    def load_latent_registry(self, manifest_path: str | os.PathLike[str] | None = None) -> None:
        """Load tag->layer->vector registry from JSON (see :mod:`shrooms_llm.steering.latent_mix`)."""
        from brainstormlm.conceptial.engine.steering.latent_mix import load_latent_registry_from_manifest

        p = manifest_path or self.config.latent_registry_path
        if not p:
            self._latent_registry = None
            return
        self._latent_registry = load_latent_registry_from_manifest(p)

    def apply_latent_weight_mix(self) -> None:
        """Combine registry vectors with :attr:`~shrooms_llm.config.SteeringConfig.latent_weights` and set hooks."""
        from brainstormlm.conceptial.engine.steering.latent_mix import resolve_weighted_latent_vectors

        if self._latent_registry is None:
            raise RuntimeError("Call load_latent_registry() before apply_latent_weight_mix().")
        if not self.config.latent_weights:
            raise RuntimeError("SteeringConfig.latent_weights is empty.")
        self.load()
        assert self._model is not None
        p = next(self._model.parameters())
        vs = resolve_weighted_latent_vectors(
            self._latent_registry,
            self.config.latent_weights,
            device=p.device,
            dtype=p.dtype,
        )
        if not vs:
            raise RuntimeError("latent mix produced no layers (all weights zero?)")
        self.config.layer_injections = []
        self.config.layer_alphas = {int(L): 1.0 for L in sorted(vs)}
        self.config.strategy = SteeringStrategy.SINGLE if len(vs) == 1 else SteeringStrategy.MULTI
        self.set_steering_vectors(vs)

    def set_steering_vector(self, v: torch.Tensor) -> None:
        """Set steering for a single layer (``config.layer_alphas`` must have exactly one key)."""
        if self.config.layer_injections:
            raise ValueError(
                "set_steering_vector is not supported when config.layer_injections is non-empty; "
                "use apply_layer_injections or set_steering_vectors explicitly."
            )
        keys = list(self.config.layer_alphas.keys())
        if len(keys) != 1:
            raise ValueError(
                "set_steering_vector requires exactly one entry in config.layer_alphas; "
                "use set_steering_vectors instead."
            )
        self._steering_vectors = {keys[0]: v}

    def set_steering_vectors(self, vs: dict[int, torch.Tensor]) -> None:
        """Per-layer steering tensors (keys must match the active hook spec when steering)."""
        self._steering_vectors = dict(vs)

    @property
    def steering_vectors(self) -> dict[int, torch.Tensor] | None:
        return self._steering_vectors

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        do_steering: bool = True,
        do_sample: bool = True,
        temperature: float = 0.7,
        capture_first_decode_residual: tuple[int, list[torch.Tensor]] | None = None,
        use_ghost_rejection: bool = False,
    ) -> str:
        """Generate and return **new** tokens only (benchmark-friendly)."""
        self.last_generation_metrics = None
        self.load()
        assert self._model is not None and self._tokenizer is not None
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        if use_ghost_rejection:
            if not do_steering or self._steering_vectors is None:
                raise RuntimeError(
                    "use_ghost_rejection requires do_steering=True and steering vectors "
                    "(set_steering_vectors / apply_latent_weight_mix / checkpoint)."
                )
            from brainstormlm.conceptial.engine.steering.ghost_engine import autoregressive_ghost_complete

            ghost_layers = [int(x) for x in (self.config.ghost_layers or list(self._steering_vectors.keys()))]
            self.last_generation_metrics = {}
            text = autoregressive_ghost_complete(
                self._model,
                tok,
                prompt,
                self._steering_vectors,
                self.config,
                ghost_layers,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                ghost_injection_site=str(self.config.ghost_injection_site),
                ghost_eps=float(self.config.ghost_projection_eps),
                ghost_strength=float(getattr(self.config, "ghost_strength", 0.6)),
                metrics_out=self.last_generation_metrics,
            )
            return text

        if do_steering:
            if self._steering_vectors is None:
                raise RuntimeError(
                    "No steering vectors set: use apply_steering_checkpoint, set_steering_vectors, "
                    "or set_steering_vector (single layer), or pass do_steering=False."
                )
            ctx = steering_context(
                self._model,
                self._steering_vectors,
                self.config,
                capture_first_decode_residual=capture_first_decode_residual,
            )
        else:
            ctx = nullcontext()

        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "pad_token_id": tok.pad_token_id,
        }
        with torch.no_grad():
            with ctx:
                out = self._model.generate(**inputs, **gen_kw)
        new_ids = out[0, input_len:]
        return tok.decode(new_ids, skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        do_steering: bool = True,
        capture_first_decode_residual: tuple[int, list[torch.Tensor]] | None = None,
        use_ghost_rejection: bool = False,
        **gen_kwargs: Any,
    ) -> str:
        """Full decoded sequence (prompt + completion). Prefer :meth:`complete` for benchmarks."""
        self.last_generation_metrics = None
        self.load()
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        if use_ghost_rejection:
            if not do_steering or self._steering_vectors is None:
                raise RuntimeError(
                    "use_ghost_rejection requires do_steering=True and steering vectors "
                    "(set_steering_vectors / apply_latent_weight_mix / checkpoint)."
                )
            from brainstormlm.conceptial.engine.steering.ghost_engine import autoregressive_ghost_new_token_ids

            ghost_layers = [int(x) for x in (self.config.ghost_layers or list(self._steering_vectors.keys()))]
            do_sample = bool(gen_kwargs.get("do_sample", True))
            temperature = float(gen_kwargs.get("temperature", 0.7))
            self.last_generation_metrics = {}
            new_part = autoregressive_ghost_new_token_ids(
                self._model,
                self._tokenizer,
                prompt,
                self._steering_vectors,
                self.config,
                ghost_layers,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                ghost_injection_site=str(self.config.ghost_injection_site),
                ghost_eps=float(self.config.ghost_projection_eps),
                ghost_strength=float(getattr(self.config, "ghost_strength", 0.6)),
                metrics_out=self.last_generation_metrics,
            )
            full_ids = torch.cat([input_ids, new_part.to(device=device)], dim=1)
            return self._tokenizer.decode(full_ids[0], skip_special_tokens=True)

        if do_steering:
            if self._steering_vectors is None:
                raise RuntimeError(
                    "No steering vectors set: use apply_steering_checkpoint, set_steering_vectors, "
                    "or set_steering_vector (single layer), or pass do_steering=False."
                )
            ctx = steering_context(
                self._model,
                self._steering_vectors,
                self.config,
                capture_first_decode_residual=capture_first_decode_residual,
            )
        else:
            ctx = nullcontext()

        with torch.no_grad():
            with ctx:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **gen_kwargs,
                )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)
