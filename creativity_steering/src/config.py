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


@dataclass(frozen=True)
class SamplingPreset:
    """Decoding parameters for one generation pool.

    The creative/neutral contrast is driven by decoding as well as framing: the creative
    pool samples at high temperature with a nucleus + min-p floor; the neutral pool decodes
    greedily. ``temperature == 0`` means greedy (argmax), and the other knobs are then
    irrelevant. A knob left at its no-op value (``top_p == 1.0``, ``top_k == 0``,
    ``min_p == 0.0``, ``repetition_penalty == 1.0``) is *not* forwarded to the model, so its
    own generation default applies — this is how the legacy "temperature only" behaviour is
    preserved (see ``models/hf.build_generate_kwargs``).
    """

    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


# The two derivation pools. Values chosen with the user: creative = temp 1.2 with a 0.95
# nucleus, top-k off, a 0.02 min-p coherence floor, and mild 1.1 repetition penalty; neutral
# = pure greedy (the most standard/expected decode).
CREATIVE_SAMPLING = SamplingPreset(
    temperature=1.2, top_p=0.95, top_k=0, min_p=0.02, repetition_penalty=1.1
)
GREEDY_SAMPLING = SamplingPreset(temperature=0.0)


@dataclass
class GenerationConfig:
    """Sampling settings for response gathering and patched generation.

    ``temperature`` is the scalar used by the *patching* phase (steered generation). The
    *derivation* phase instead uses the per-pool presets ``creative_sampling`` /
    ``neutral_sampling`` (see ``sampling_by_framing``), so the creative and neutral pools can
    be decoded differently.
    """

    max_new_tokens: int = 256
    temperature: float = 0.9
    samples_per_prompt: int = 4  # multiple stochastic samples to grow the verified subset
    seed: int = 0
    creative_sampling: SamplingPreset = CREATIVE_SAMPLING
    neutral_sampling: SamplingPreset = GREEDY_SAMPLING

    def sampling_by_framing(self) -> dict[str, SamplingPreset]:
        """Map each prompt framing to its decoding preset (used by ``gather_responses``)."""
        return {"creative": self.creative_sampling, "neutral": self.neutral_sampling}


@dataclass
class FilterConfig:
    """Turn raw responses into a verified contrast subset.

    Two methods (``method``):

    * ``"threshold"`` (the original per-item path) — a creative-framing response is kept as a
      *positive* only if its single-response creativity score >= ``creative_min``; a neutral
      response is kept as a *negative* only if its score <= ``neutral_max``. ``scorer`` selects
      the per-item ``CreativityScorer``.
    * ``"evaluator_stack"`` — score every response with the two-stage ``creativity_evaluators``
      stack (Stage 1 model-free + optional Stage 2 LLM-as-judge), then keep the highest-scoring
      creative responses (at/above ``creative_quantile``) and lowest-scoring neutral responses
      (at/below ``neutral_quantile``). ``enable_judge`` toggles Stage 2; ``generator_family``
      drives judge-family exclusion; ``length_match`` length-matches the kept pools for
      DiffMeans hygiene.

    ``min_per_class`` guards (both methods) against deriving DiffMeans on too little data.
    """

    method: str = "threshold"  # "threshold" | "evaluator_stack"
    min_per_class: int = 100  # the conformity paper used N=100 per condition

    # --- threshold method ---
    creative_min: float = 0.55
    neutral_max: float = 0.45
    scorer: str = "real"  # "real" | "embedding" | "heuristic" | "fake"

    # --- evaluator_stack method ---
    creative_quantile: float = 0.8  # keep creative-framed at/above this quantile of their scores
    neutral_quantile: float = 0.2  # keep neutral-framed at/below this quantile of their scores
    enable_judge: bool = True  # run the Stage-2 LLM-as-judge ensemble (the "both phases" path)
    generator_family: str | None = "qwen"  # judge-family exclusion (the generator is Qwen)
    stage1_floor_percentile: float = 20.0  # Stage-2 gating floor inside the evaluator stack
    length_match: bool = False  # length-match kept pools (prevents encoding verbosity)
    length_match_tolerance: float = 0.10
    use_reward_fallback: bool = False  # use Skywork 8B reward instead of the 27B (tighter pods)


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
        g = dict(raw["generation"])
        for key in ("creative_sampling", "neutral_sampling"):
            if key in g and isinstance(g[key], dict):
                g[key] = SamplingPreset(**g[key])
        raw["generation"] = GenerationConfig(**g)
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
