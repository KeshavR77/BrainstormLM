"""Shared schemas for the creativity evaluator stack.

Every result dataclass the two stages produce, plus the user-facing ``EvalConfig``, lives
here so that ``stage1`` / ``stage2`` / ``composite`` / ``pipeline`` can all reference the
same types without importing each other (no circular imports). The stage modules re-export
the dataclasses they "own" (e.g. ``stage2.ensemble.EnsembleResult``) for ergonomics, but the
canonical definitions are here.

``EvalConfig`` is a pydantic model: it doubles as the JSON-serialisable run config and the
source of the ``config_hash`` stamped onto every ``EvalResult`` for reproducibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class PromptCategory(str, Enum):
    """Which side of the DiffMeans contrast a record belongs to.

    CREATIVE  — a creativity-invoking prompt; we keep it as a *positive* only if the
                response is actually creative (high composite score).
    NEUTRAL   — a neutral prompt; we keep it as a *negative* only if the response is
                actually not creative (low composite score).
    """

    CREATIVE = "creative"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Stage 1 result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DSIResult:
    """Divergent Semantic Integration (Johnson et al. 2022).

    ``score`` is ``None`` when the response is too short to be meaningful (see ``reason``).
    """

    score: float | None
    n_tokens: int
    layer_indices: tuple[int, int] = (6, 7)
    reason: str | None = None  # e.g. "too_short"
    is_code: bool = False  # flagged when the response looks like pure code
    n_chunks: int = 1  # >1 when long responses were chunked and averaged


@dataclass
class LexicalResult:
    """Lexical/syntactic redundancy. Higher == more redundant == *less* creative."""

    template_overlap: float  # fraction of POS 4-grams matching the LLM-template set
    slop_rate: float  # slop-phrase occurrences / total tokens
    combined: float  # weighted sum (slop weighted 2x); the composite subtracts this
    slop_hits: list[str] = field(default_factory=list)
    template_overlap_available: bool = True  # False when spaCy is unavailable


@dataclass
class PPLResult:
    """Length-normalised perplexity under a fixed small reference LM."""

    nll_per_token: float
    n_tokens: int
    model: str = "gpt2-medium"


# ---------------------------------------------------------------------------
# Stage 2 result dataclasses
# ---------------------------------------------------------------------------

# The five rubric dimensions, in canonical order (Torrance/TTCW + Saakyan pragmaticality).
DIMENSIONS: tuple[str, ...] = (
    "Originality",
    "Fluency",
    "Flexibility",
    "Elaboration",
    "Pragmaticality",
)


@dataclass
class JudgeResult:
    """One judge's 1–5 score on each rubric dimension, plus its short reasoning."""

    scores: dict[str, int]
    reasoning: str
    judge_name: str


@dataclass
class EnsembleResult:
    """Aggregated multi-judge result.

    ``per_dimension_mean`` averages each dimension across the judges that succeeded;
    ``overall`` is the mean of those dimension means. ``inter_judge_spearman`` is a health
    check — a low value means the judges disagree and the score should be distrusted.
    """

    per_judge: list[JudgeResult]
    per_dimension_mean: dict[str, float]
    overall: float
    inter_judge_spearman: float
    n_judges_succeeded: int


# ---------------------------------------------------------------------------
# Top-level evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """The full per-response evaluation record.

    Raw Stage-1 and Stage-2 components are kept alongside the final scalar so the composite
    formula can be re-derived offline without re-running the (expensive) judges.
    """

    prompt: str
    response: str
    category: PromptCategory
    # Stage 1
    dsi: float | None
    lexical: LexicalResult
    ppl: float
    reward: float
    # Stage 2 (None when filtered out at the Stage-1 floor)
    ensemble: EnsembleResult | None
    # Composite
    creativity_score: float
    # Bookkeeping
    n_tokens: int
    stage1_only: bool
    timestamp: str
    config_hash: str
    # Raw composite components (exposed so the formula can be re-tuned post-hoc)
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class JudgeSpec(BaseModel):
    """One ensemble member: provider + model + a stable display name + its model family.

    ``family`` drives judge-family exclusion (see ``stage2.ensemble``): a judge is dropped
    when its family matches the family of the model that *generated* the response, so a model
    never grades its own siblings.
    """

    provider: str  # "anthropic" | "openai" | "google"
    model: str
    name: str
    family: str  # "anthropic" | "openai" | "google" | "qwen" | ...


def _default_judges() -> list[JudgeSpec]:
    return [
        JudgeSpec(
            provider="anthropic", model="claude-sonnet-4-5", name="claude", family="anthropic"
        ),
        JudgeSpec(provider="openai", model="gpt-4o", name="gpt4o", family="openai"),
        JudgeSpec(provider="google", model="gemini-2.5-flash", name="gemini", family="google"),
    ]


class EvalConfig(BaseModel):
    """Controls which signals run, which judges vote, and how Stage 2 is gated.

    Mutating any field changes ``config_hash``, which is stamped on every ``EvalResult`` —
    so cached results never silently mix configs.
    """

    # --- Stage 1 toggles ---
    run_dsi: bool = True
    run_lexical: bool = True
    run_ppl: bool = True
    # Off by default: the default reward model (Skywork 27B) is too large for most
    # environments. With Stage 2 on, the judges' Pragmaticality dimension carries the
    # quality pole; flip this back to True (optionally with use_reward_fallback) to add a
    # model-free Stage-1 quality signal.
    run_reward: bool = False

    # --- Stage 1 model ids ---
    dsi_model: str = "bert-base-uncased"
    dsi_layers: tuple[int, int] = (6, 7)
    dsi_min_tokens: int = 5
    dsi_chunk_size: int | None = None
    ppl_model: str = "gpt2-medium"
    # NoveltyBench's choice (top RewardBench performer); fall back to the 8B if the 27B is
    # impractical for the run environment.
    reward_model: str = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    reward_model_fallback: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    use_reward_fallback: bool = False

    # --- Stage 2 ---
    enable_stage2: bool = True
    judges: list[JudgeSpec] = Field(default_factory=_default_judges)
    stage1_percentile_floor: float = 20.0  # responses below this S1-composite pct skip S2
    judge_truncate_chars: int = 4000  # EQ-Bench v3 length-bias control
    judge_temperature: float = 0.0
    judge_max_retries: int = 3
    # Family of the model that generated the responses; judges in this family are excluded.
    generator_family: str | None = None
    inter_judge_warn_threshold: float = 0.5

    # --- Composite weights (exposed for tuning) ---
    composite_creativity_weight: float = 0.85
    composite_elaboration_weight: float = 0.15

    # --- Bookkeeping ---
    cache_dir: str = ".cache/creativity_evaluators"

    @property
    def config_hash(self) -> str:
        """Stable short hash of the config (model_dump_json is key-sorted by pydantic)."""
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
