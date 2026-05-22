"""Experiment configuration — JSON-driven, with local-mock and RunPod-14B presets.

The only fields that change between "run offline on my laptop" and "run the real
Qwen2.5-14B on a RunPod A100" are `backend` and `model_id`; everything else
(layers, alphas, thresholds, capture position) is shared methodology.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Capture positions we know how to extract a per-layer residual from.
CAPTURE_LAST_RESPONSE = "last_response_token"  # locked default (last generated token)
CAPTURE_MEAN_RESPONSE = "mean_response"  # mean over generated tokens
CAPTURE_ASSISTANT_START = "assistant_start"  # last prompt token (conformity-paper style)
VALID_CAPTURES = (CAPTURE_LAST_RESPONSE, CAPTURE_MEAN_RESPONSE, CAPTURE_ASSISTANT_START)


@dataclass
class GenerationConfig:
    """Sampling settings for response gathering and patched generation."""

    max_new_tokens: int = 256
    temperature: float = 0.9
    samples_per_prompt: int = 4  # multiple stochastic samples to grow the verified subset
    seed: int = 0


@dataclass
class FilterConfig:
    """Evaluator thresholds that turn raw responses into a verified subset.

    A creative-framing response is kept as a *positive* only if its single-response
    creativity score >= ``creative_min``; a neutral-framing response is kept as a
    *negative* only if its score <= ``neutral_max``. Everything in the ambiguous middle
    is dropped. ``min_per_class`` guards against deriving DiffMeans on too little data.
    """

    creative_min: float = 0.55
    neutral_max: float = 0.45
    min_per_class: int = 100  # the conformity paper used N=100 per condition
    scorer: str = "real"  # "real" | "heuristic" | "fake"


@dataclass
class PatchingConfig:
    """Activation-patching sweep grid."""

    # Exp 1 — per-layer matched extract-and-inject sweep. Both signs (paper used these).
    alphas: tuple[float, ...] = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
    # Exp 2 — best single vector injected at *every* layer needs a small alpha.
    all_layers_alpha: float = 0.1
    rescale: bool = True  # match injection magnitude to per-token activation norm
    eval_samples: int = 5  # samples per (layer, alpha) cell, scored for diversity


@dataclass
class ExperimentConfig:
    """Top-level config for a derive → analyze → patch run."""

    backend: str = "mock"  # "mock" | "hf"
    model_id: str = "mock-qwen"
    device: str | None = None  # None → auto (cuda/mps/cpu); only used by hf backend
    dtype: str = "bf16"  # only used by hf backend
    seed: int = 0

    # Mock-backend geometry (ignored by hf backend, which reads these from the model).
    num_layers: int = 24
    hidden_dim: int = 128
    mock_creative_layer: int = 12  # where the planted creativity axis is most salient

    capture_position: str = CAPTURE_LAST_RESPONSE
    # Cache these positions too, so a mean-pool variant can be derived without re-running.
    extra_capture_positions: tuple[str, ...] = (CAPTURE_MEAN_RESPONSE,)

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)

    out_dir: str = "artifacts/run"

    def __post_init__(self) -> None:
        if self.capture_position not in VALID_CAPTURES:
            raise ValueError(
                f"capture_position {self.capture_position!r} not in {VALID_CAPTURES}"
            )
        for p in self.extra_capture_positions:
            if p not in VALID_CAPTURES:
                raise ValueError(f"extra capture position {p!r} not in {VALID_CAPTURES}")

    @property
    def all_capture_positions(self) -> tuple[str, ...]:
        """Capture position plus extras, primary first, de-duplicated, order-stable."""
        seen: list[str] = [self.capture_position]
        for p in self.extra_capture_positions:
            if p not in seen:
                seen.append(p)
        return tuple(seen)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def _build_nested(raw: dict) -> ExperimentConfig:
    """Construct ExperimentConfig from a (possibly nested) plain dict."""
    raw = dict(raw)
    if "generation" in raw and isinstance(raw["generation"], dict):
        raw["generation"] = GenerationConfig(**raw["generation"])
    if "filtering" in raw and isinstance(raw["filtering"], dict):
        raw["filtering"] = FilterConfig(**raw["filtering"])
    if "patching" in raw and isinstance(raw["patching"], dict):
        p = dict(raw["patching"])
        if "alphas" in p:
            p["alphas"] = tuple(float(a) for a in p["alphas"])
        raw["patching"] = PatchingConfig(**p)
    for key in ("extra_capture_positions",):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])
    return ExperimentConfig(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a JSON file."""
    raw = json.loads(Path(path).read_text())
    return _build_nested(raw)
