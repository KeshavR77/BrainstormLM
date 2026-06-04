"""A/B validation harness for Conceptial diversity levers.

Honestly answers: do the new diversity levers beat the shipping stack, measured properly?

==============================
HONESTY NOTES (baked in here, printed in the report)
==============================

1. ``conditional_vendi`` (from brainstormlm.eval) computes MEAN WITHIN-PROMPT Vendi score,
   NOT the prompt-projection Conditional-Vendi of arXiv:2411.02817.  It is labelled
   "mean per-prompt Vendi" throughout.  Do not read a score here as genuine
   diversity-conditioned in the Carlini / Thornton sense.

2. ``generate_steered_batch`` (the ``ortho`` arm) is GREEDY (SamplingPreset temperature=0).
   Variation across the batch comes ONLY from the per-sample steering directions.
   The temperature-sampled arms (``temp``, ``antimode``, ``repulsion``, ``full``) are
   stochastically generated.  A steering "win" over ``greedy`` therefore does NOT mean it
   beat sampling; the fair bar is the ``antimode`` and ``temp`` arms.

==============================
PRE-REGISTERED SUCCESS CRITERIA  (verbatim — harness prints PASS/FAIL against these)
==============================

A lever WINS only if:
  * NoveltyBench-Distinct n_distinct >= antimode baseline n_distinct + 1 class at K=8
  * mean per-prompt Vendi higher than antimode with NON-OVERLAPPING 95% bootstrap CIs.

VOID if quality breaks:
  * coherence_rate (= 1 - salad_rate, coherence modes: token_loop + non_ascii only) drops
    >2pp vs baseline, OR salad_rate not ≈ 0 (> 0.10 triggers void).
  * quality_pass_rate (LLM judge: coherence>=3 AND on-topic>=4) drops >5pp vs baseline.

Each lever must show positive MARGINAL contribution in the ablation or it does NOT ship.

==============================
ALPHA NOTE FOR 14B RUNS
==============================
The coherence cliff for the 0.5B model is ~0.3–0.4; the 14B cliff has NOT been
re-measured.  Start at --alpha 0.3 and sweep (0.3, 0.5, 0.8, 1.0) before committing.
The L17 vector was validated at alpha=1.5; its sweep is [1.0, 1.5, 2.0, 2.62].
The ArchetypeBank on-the-fly DiffMeans directions likely have smaller norm → different cliff.

==============================
Usage
==============================

# Mock (no GPU needed, OpenAI embeddings are fine — key is set):
    uv run python scripts/validate_diversity.py --mock --k 8

# Real 14B on RunPod pod:
    uv run python scripts/validate_diversity.py \\
        --config creativity_steering/configs/runpod_qwen14b_v2.json --layer 17 \\
        --alphas 1.0,1.5,2.0,2.62 --k 8

"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without editable install
# ---------------------------------------------------------------------------
# _SRC points to <repo>/src  (makes `import brainstormlm` work)
# _STEERING points to <repo>/creativity_steering  (the OUTER dir that CONTAINS
#   creativity_steering/creativity_steering/ — inserting it lets Python find
#   the inner package as `import creativity_steering`).
# Mirrors how creativity_steering/scripts/smoke_semantic_spread.py adds
#   parents[2]/src for brainstormlm.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_STEERING = _REPO / "creativity_steering"
for _p in (_SRC, _STEERING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# FIX (a): Built-in briefs expanded to ~40 genuinely diverse domains
# ---------------------------------------------------------------------------

BUILTIN_BRIEFS: list[str] = [
    # naming
    "Create a product name for a noise-canceling headphone brand targeting remote workers.",
    "Name a new premium sparkling water brand that targets urban wellness enthusiasts.",
    "Suggest a name for a DTC pet insurance company with a playful, millennial-friendly tone.",
    # marketing campaign
    "Design a marketing campaign concept for a sustainable coffee subscription.",
    "Pitch a summer campaign for a sunscreen brand targeting Gen-Z outdoor adventurers.",
    "Propose a guerrilla marketing activation for a new airport lounge network.",
    # logo
    "Pitch a logo concept for a fintech startup focused on micro-investing.",
    "Describe a logo direction for a community-owned urban grocery cooperative.",
    # GTM strategy
    "Develop a GTM strategy concept for launching an AI-powered legal document tool.",
    "Outline a GTM approach for a B2B carbon-accounting SaaS entering the EU market.",
    # product feature
    "Propose an app feature that helps fitness beginners stay accountable.",
    "Suggest a new feature for a recipe app to reduce household food waste.",
    # event theme
    "Suggest an event theme for a tech company's annual all-hands gathering.",
    "Design a theme for a climate-tech investor summit that inspires action without doom.",
    # packaging
    "Design packaging for a premium dark chocolate bar made with rare cacao.",
    "Propose packaging for a refillable luxury hand cream sold direct-to-consumer.",
    # slogan
    "Write a brand slogan for a children's coding education platform.",
    "Craft a slogan for a neobank that targets gig-economy workers.",
    # business model
    "Propose a business model innovation for a traditional bookstore chain.",
    "Suggest a revenue model for a neighbourhood tool-lending library going national.",
    # UX flow
    "Design a UX flow concept for a sleep-tracking wearable app's onboarding.",
    "Sketch a UX flow for a carbon-offset marketplace where consumers buy offsets at checkout.",
    # tagline
    "Write a tagline for a mental-health app targeting burned-out executives.",
    "Create a tagline for a peer-to-peer luggage-storage marketplace.",
    # mascot
    "Design a mascot concept for a children's dental hygiene brand.",
    "Propose a mascot for a cybersecurity firm that needs to appear approachable to SMEs.",
    # partnership idea
    "Propose a brand partnership between a running-shoe company and a mindfulness app.",
    "Suggest a co-marketing idea between a cloud-kitchen brand and a fitness influencer platform.",
    # pricing model
    "Design a pricing model for a B2B SaaS HR tool targeting Series A startups.",
    "Propose a pricing strategy for a premium AI tutoring service targeting high-school students.",
    # onboarding hook
    "Write an onboarding hook for a habit-tracking app that surfaces within the first 60 seconds.",
    "Create an onboarding hook for a corporate expense management tool that wins over the CFO.",
    # retention mechanic
    "Invent a retention mechanic for a language-learning app to pull back churned users.",
    "Design a loyalty mechanic for a direct-to-consumer supplement brand.",
    # content series
    "Pitch a content series concept for a B2B climate consultancy to build thought leadership.",
    "Propose a social-media content series for a zero-waste kitchen product brand.",
    # brand voice
    "Define a brand voice for a telehealth startup serving rural communities.",
    "Write a brand voice brief for a luxury pet-food brand targeting dog parents.",
    # color story
    "Develop a color story for a meditation app rebranding from clinical white to warmer tones.",
    # campaign hashtag
    "Invent a campaign hashtag and rally concept for a global reforestation charity's annual drive.",
    # additional diverse domains
    "Create a community activation idea for a craft brewery entering a new city.",
    "Pitch a pop-up retail concept for a digitally-native eyewear brand.",
]

# ---------------------------------------------------------------------------
# Canned texts for --mock mode
# ---------------------------------------------------------------------------

_MOCK_DIVERSE = [
    "Aether Hush — biomimetic noise-canceling inspired by owl feather geometry, "
    "targets deep-work freelancers with circadian focus timers.",
    "SilentBrim — a wide-brimmed form factor that blocks peripheral sound via "
    "physical acoustic shaping, no digital processing required.",
    "FlowState — subscription hardware model, the headphones learn your peak-focus "
    "hours over 30 days and auto-engage at those windows.",
    "NoiseCave — a shared-space rental model, ultra-premium cans deployed in co-working "
    "spots billed by the focus session.",
    "WhisperMesh — graphene-lattice drivers, markets on pure material science story, "
    "targets audiophile-adjacent remote professionals.",
    "ChronoFocus — combines noise-canceling with a Pomodoro coaching layer, the "
    "headset itself vibrates gently at break intervals.",
    "MindVault — black-box aesthetic, the app visualizes your daily distraction profile "
    "and the headset physically responds to alert levels.",
    "HushPod — companion capsule for open offices, the headphones pair with a "
    "fabric privacy pod that amplifies the spatial isolation.",
]

_MOCK_DUPLICATE_HEAVY = [
    "FocusWave — premium noise-canceling for remote work.",
    "FocusWave Pro — premium noise-canceling for remote work, premium edition.",
    "FocusWave Elite — premium noise-canceling for remote work, elite tier.",
    "QuietZone — a quiet zone for remote workers.",
    "QuietZone Plus — quiet zone, plus tier.",
    "SilentSpace — silent space for working from home.",
    "SilentSpace Pro — silent space pro for home offices.",
    "NoiseFree — noise-free experience for remote workers.",
]


# ---------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------


class GeneratorBase:
    """Minimal interface: given a brief and count K, return K strings."""

    def generate_k(self, brief: str, k: int) -> list[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock generator (no model — deterministic enough for CI)
# ---------------------------------------------------------------------------


class MockGenerator(GeneratorBase):
    """Returns canned texts so the full arms→metrics→CIs→PASS/FAIL path runs with no model.

    Each arm gets a slightly different mix:
    - greedy: all duplicates (simulates mode collapse)
    - temp:   diverse + 2 near-duplicates
    - antimode: fully diverse
    - ortho:  diverse but slightly less than antimode
    - ortho_temp: fully diverse (steering + sampling)
    - ortho_L17: fully diverse (production L17 vector arm)
    - vs:     diverse, different texts
    - repulsion: diverse (repulsion selects well)
    - full:   diverse
    """

    def __init__(self, arm: str, seed: int = 0) -> None:
        # Strip the alpha suffix if present (e.g. "ortho_L17@1.5" -> "ortho_L17")
        self.arm = arm.split("@")[0]
        self._rng = random.Random(seed)

    def generate_k(self, brief: str, k: int) -> list[str]:
        suffix = brief[:30].replace(" ", "_")
        if self.arm == "greedy":
            # K copies of one idea — sanity-check floor
            base = f"Standard solution for: {brief[:60]}"
            return [base] * k
        if self.arm == "temp":
            # Mostly diverse, 2 near-duplicates
            pool = list(_MOCK_DIVERSE[:k])
            if k >= 2:
                pool[0] = pool[1] + " (variant)"
            return pool[:k]
        if self.arm == "antimode":
            return list(_MOCK_DIVERSE[:k])
        if self.arm == "ortho":
            # Greedy + steering: somewhat diverse but one repeated (isolation control)
            pool = list(_MOCK_DIVERSE[:k])
            if k >= 3:
                pool[k - 1] = pool[0] + " [steered variant]"
            return pool[:k]
        if self.arm == "ortho_temp":
            # Steering + sampling (production config): fully diverse
            return list(_MOCK_DIVERSE[:k])
        if self.arm == "ortho_L17":
            # Production L17 vector arm: fully diverse (same mock as ortho_temp)
            return list(_MOCK_DIVERSE[:k])
        if self.arm == "vs":
            return [f"VS-{i}: {_MOCK_DIVERSE[i % len(_MOCK_DIVERSE)]} [angle {i}]" for i in range(k)]
        if self.arm == "repulsion":
            # Over-generate (3K) then deduplicate in the arm — mock returns k distinct
            return list(_MOCK_DIVERSE[:k])
        if self.arm == "full":
            return list(_MOCK_DIVERSE[:k])
        raise ValueError(f"Unknown arm {self.arm!r}")


# ---------------------------------------------------------------------------
# Real generators (require a loaded backend)
# ---------------------------------------------------------------------------


def _build_real_backend(config_path: str):
    """Load HF backend from an ExperimentConfig JSON (same pattern as smoke_semantic_spread)."""
    from creativity_steering.config import load_config
    from creativity_steering.models.factory import build_backend

    cfg = load_config(config_path)
    return build_backend(cfg)


def _build_archetype_bank(backend, layer: int):
    """Build a tiny real DiffMeans ArchetypeBank from creative/dull contrast activations.

    Mirrors smoke_semantic_spread._build_bank exactly.
    """
    from creativity_steering.config import CAPTURE_MEAN_RESPONSE
    from creativity_steering.derivation.archetype_bank import ArchetypeBank

    BANK_BRIEF = "Suggest one idea for revitalizing an unused public lot."
    CREATIVE = [
        "Flood it at dusk into a shallow mirror-pool so the skyline doubles underfoot and "
        "kids wade through reflected clouds.",
        "Hang a canopy of recycled-bottle prisms that scatter noon sun into a slow "
        "constellation crawling across the pavement.",
        "Grow living benches from mycelium that reshape each season and quietly compost "
        "themselves into next year's wildflowers.",
        "Tune a grove of resonant steel pipes to the prevailing wind so the lot hums a "
        "different chord every afternoon.",
    ]
    DULL = [
        "Add some benches and a few trees so people have a place to sit.",
        "Put in a small playground and maybe a basketball court.",
        "Plant grass and add lighting so the area is safe at night.",
        "Install trash cans and signs to keep the space clean.",
    ]

    def stack(texts):
        rows = [
            backend.capture_activations(
                BANK_BRIEF, t, positions=[CAPTURE_MEAN_RESPONSE]
            )[CAPTURE_MEAN_RESPONSE]
            for t in texts
        ]
        return np.stack(rows)

    pos, neg = stack(CREATIVE), stack(DULL)
    n_layers = backend.num_layers
    arch_layers = [round(n_layers * f) for f in (0.25, 0.40, 0.55, 0.70)]
    return ArchetypeBank.from_stacks(pos, neg, archetype_layers=arch_layers)


# ---------------------------------------------------------------------------
# FIX (b): LLM judge — quality_pass_rate via gpt-4o-mini
# ---------------------------------------------------------------------------

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "integer"},
        "on_topic": {"type": "integer"},
    },
    "required": ["coherence", "on_topic"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM = (
    "You are a strict creative-output quality judge. "
    "Given a brief and a generated output, score BOTH dimensions:\n"
    "  coherence (1–5): 1=incoherent/gibberish, 3=readable but weak, 5=clear and well-formed.\n"
    "  on_topic (1–5): 1=off-topic, 4=clearly addresses the brief, 5=directly relevant and specific.\n"
    "Return ONLY a JSON object with keys 'coherence' and 'on_topic' (integers). No explanation."
)


def _judge_one(brief: str, text: str, *, use_cache: bool = True) -> dict[str, int]:
    """Score a single (brief, text) pair via gpt-4o-mini.  Cached by request key."""
    from brainstormlm.clients.openai import complete_json

    messages = [
        {
            "role": "user",
            "content": f"Brief: {brief}\n\nOutput: {text}",
        }
    ]
    result, _ = complete_json(
        system=_JUDGE_SYSTEM,
        messages=messages,
        model="gpt-4o-mini",
        schema=_JUDGE_SCHEMA,
        schema_name="quality_scores",
        max_tokens=64,
        temperature=0.0,
        use_cache=use_cache,
    )
    return {"coherence": int(result.get("coherence", 1)), "on_topic": int(result.get("on_topic", 1))}


def score_all_texts(
    texts_per_brief: list[list[str]],
    briefs: list[str],
    *,
    mock: bool = False,
) -> list[list[dict[str, int]]]:
    """Return per-text judge scores, shape [n_briefs, k].

    Compute ONCE (outside bootstrap loop) and reuse.
    In mock mode returns deterministic plausible scores.
    """
    scores: list[list[dict[str, int]]] = []
    for bi, (brief, texts) in enumerate(zip(briefs, texts_per_brief)):
        brief_scores: list[dict[str, int]] = []
        for text in texts:
            if mock:
                # Deterministic mock scores: greedy/empty text gets low scores, others high
                if not text.strip() or text.startswith("Standard solution"):
                    brief_scores.append({"coherence": 2, "on_topic": 2})
                else:
                    brief_scores.append({"coherence": 4, "on_topic": 4})
            else:
                brief_scores.append(_judge_one(brief, text))
        scores.append(brief_scores)
    return scores


def quality_pass_rate(scores_per_brief: list[list[dict[str, int]]]) -> float:
    """Fraction of outputs with coherence>=3 AND on_topic>=4 (matches repo rubric)."""
    all_scores = [s for group in scores_per_brief for s in group]
    if not all_scores:
        return 1.0
    passed = sum(1 for s in all_scores if s["coherence"] >= 3 and s["on_topic"] >= 4)
    return passed / len(all_scores)


# ---------------------------------------------------------------------------
# ARM implementations
# ---------------------------------------------------------------------------


def run_arm_greedy(backend, brief: str, k: int) -> list[str]:
    """K greedy generations from the backend. Floor; expect ~1 distinct class."""
    from creativity_steering.config import SamplingPreset

    return [
        backend.generate(brief, temperature=0.0, seed=i, max_new_tokens=256)
        for i in range(k)
    ]


def run_arm_temp(backend, brief: str, k: int, temperature: float = 0.8) -> list[str]:
    """K temperature-sampled generations (temp=0.8). Cheap diversity baseline."""
    from creativity_steering.config import SamplingPreset

    return [
        backend.generate(brief, temperature=temperature, seed=i, max_new_tokens=256)
        for i in range(k)
    ]


def run_arm_antimode(backend, brief: str, k: int, temperature: float = 0.8) -> list[str]:
    """THE SHIPPING BASELINE: sequential temperature generation with prompt-level anti-mode exclusion.

    Each call adds the prior concept name to the exclusion list.
    """
    from brainstormlm.conceptial.anti_mode import extract_summary, format_anti_mode

    results: list[str] = []
    prior_summaries: list[str] = []
    for i in range(k):
        prompt = format_anti_mode(
            brief,
            prior_summaries=prior_summaries,
            output_mode="generic",
        )
        text = backend.generate(
            prompt, temperature=temperature, seed=i, max_new_tokens=256
        )
        results.append(text)
        name = extract_summary(text)
        if name:
            prior_summaries.append(name)
    return results


def run_arm_ortho(
    backend,
    brief: str,
    k: int,
    layer: int,
    alpha: float,
    bank,
    temperature: float = 0.0,
) -> list[str]:
    """Orthogonalised-archetype per-sample steering batch. Two modes:

    - temperature=0.0 (arm "ortho"): GREEDY isolation CONTROL. Variation across the batch
      comes ONLY from the steering directions (no sampling noise) — it answers "does steering
      itself create diversity?". Do NOT compare this to greedy; the bar is antimode/temp.
    - temperature>0 (arm "ortho_temp"): PRODUCTION config — steering AND sampling together,
      i.e. what V1 would actually ship. This is the fair head-to-head vs antimode/temp.
    """
    from creativity_steering.config import SamplingPreset
    from creativity_steering.patching.steering import build_orthogonal_archetype_dirs

    res = build_orthogonal_archetype_dirs(bank, k)
    return backend.generate_steered_batch(
        brief,
        dirs=res.dirs,
        layer=layer,
        alpha=alpha,
        sampling=SamplingPreset(temperature=temperature),
        max_new_tokens=256,
    )


def run_arm_ortho_L17(
    backend,
    brief: str,
    k: int,
    alpha: float,
    vec: np.ndarray,
    temperature: float = 0.0,
) -> list[str]:
    """FIX (c): Production L17 steering vector arm (14B-only, 5120-dim).

    Builds per-sample directions via stiefel_basis(vec, k), injects at layer 17.
    Only called when backend.hidden_dim == vec.shape[-1].
    """
    from creativity_steering.config import SamplingPreset
    from creativity_steering.patching.orthogonal import stiefel_basis

    dirs = stiefel_basis(vec, k)  # (k, hidden_dim), one direction per sample
    return backend.generate_steered_batch(
        brief,
        dirs=dirs,
        layer=17,
        alpha=alpha,
        sampling=SamplingPreset(temperature=temperature),
        max_new_tokens=256,
    )


def run_arm_vs(backend, brief: str, k: int, temperature: float = 0.8) -> list[str]:
    """Verbalized Sampling arm — one refine call, tail-biased.

    Uses make_llm/make_embedder (bridges to configured provider).
    Falls back to temperature sampling if refine fails.
    """
    try:
        from brainstormlm.refine import RefineResult, make_embedder, make_llm, refine

        llm = make_llm(temperature=temperature)
        embedder = make_embedder()
        result: RefineResult = refine(brief, k=k, llm=llm, embedder=embedder)
        ideas = result.ideas[:k]
        # Pad if fewer angles survived dedup
        while len(ideas) < k:
            ideas.append(ideas[-1] if ideas else "")
        return ideas
    except Exception as exc:
        print(f"  [vs arm] refine failed ({type(exc).__name__}: {exc}), falling back to temp", flush=True)
        return run_arm_temp(backend, brief, k, temperature=temperature)


def run_arm_repulsion(backend, brief: str, k: int, temperature: float = 0.8) -> list[str]:
    """Over-generate temp×3, then select_by_repulsion down to K."""
    from brainstormlm.pipeline.select import select_by_repulsion

    candidates = run_arm_temp(backend, brief, k * 3, temperature=temperature)
    result = select_by_repulsion(candidates, n=k)
    selected = result.kept_texts[:k]
    while len(selected) < k:
        # Fill up from dropped if repulsion was too aggressive
        for idx in result.dropped_indices:
            if len(selected) >= k:
                break
            if idx < len(candidates):
                selected.append(candidates[idx])
    return selected[:k]


def run_arm_full(backend, brief: str, k: int, temperature: float = 0.8) -> list[str]:
    """antimode gen + over-generate → repulsion select.

    Generate antimode × 3K, then select_by_repulsion down to K.
    """
    from brainstormlm.conceptial.anti_mode import extract_summary, format_anti_mode
    from brainstormlm.pipeline.select import select_by_repulsion

    candidates: list[str] = []
    prior_summaries: list[str] = []
    for i in range(k * 3):
        prompt = format_anti_mode(
            brief,
            prior_summaries=prior_summaries,
            output_mode="generic",
        )
        text = backend.generate(
            prompt, temperature=temperature, seed=i, max_new_tokens=256
        )
        candidates.append(text)
        name = extract_summary(text)
        if name:
            prior_summaries.append(name)

    result = select_by_repulsion(candidates, n=k)
    selected = result.kept_texts[:k]
    while len(selected) < k:
        for idx in result.dropped_indices:
            if len(selected) >= k:
                break
            if idx < len(candidates):
                selected.append(candidates[idx])
    return selected[:k]


# ---------------------------------------------------------------------------
# Coherence rate helper (token_loop + non_ascii only, per spec)
# ---------------------------------------------------------------------------


def coherence_rate_from_texts(texts: Sequence[str]) -> float:
    """1 - salad_rate using only token_loop and non_ascii checks (not micro_output or comma_bleed)."""
    from brainstormlm.eval.salad import salad_detector

    if not texts:
        return 1.0
    flagged = sum(
        1
        for t in texts
        if salad_detector(t)["token_loop"] or salad_detector(t)["non_ascii"]
    )
    return 1.0 - (flagged / len(texts))


# ---------------------------------------------------------------------------
# Metrics per arm across all briefs
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    texts_per_brief: list[list[str]]  # shape: [n_briefs, k]
    # Pre-computed judge scores; None until set by run_study (outside bootstrap)
    judge_scores_per_brief: list[list[dict[str, int]]] | None = field(default=None)

    @property
    def all_texts(self) -> list[str]:
        return [t for group in self.texts_per_brief for t in group]


def compute_metrics(arm_result: ArmResult) -> dict[str, float]:
    """Compute aggregate metrics.

    NoveltyBench-Distinct is computed PER BRIEF then averaged, so it lies in [1, k] — the count
    of genuinely-different ideas *within a single brainstorm* (the product promise). Pooling
    across briefs would just count one-class-per-brief and is meaningless (a greedy arm whose
    output differs per brief would look "diverse").

    salad_rate is the COHERENCE-ONLY rate (= 1 - coherence_rate; token_loop + non_ascii only).
    micro_output / comma_bleed are steering-degeneration signals, NOT quality signals for a
    finished idea (a terse idea is fine), so they must not gate the verdict.

    quality_pass_rate: fraction of outputs with coherence>=3 AND on_topic>=4 (LLM judge).
    Judge scores are pre-computed once outside the bootstrap loop; resampling reuses them.
    """
    from brainstormlm.eval import conditional_vendi, novelty_bench_distinct

    groups = [g for g in arm_result.texts_per_brief if g]
    per_brief = [novelty_bench_distinct(g) for g in groups]
    n_distinct = float(np.mean([nb["n_distinct"] for nb in per_brief])) if per_brief else 0.0
    distinct_ratio = float(np.mean([nb["distinct_ratio"] for nb in per_brief])) if per_brief else 0.0
    cv = conditional_vendi(arm_result.texts_per_brief)
    cr = coherence_rate_from_texts(arm_result.all_texts)
    sr = 1.0 - cr  # coherence-only salad rate, consistent with the quality gate

    # FIX (b): quality_pass_rate — uses pre-computed judge scores, zero cost inside bootstrap
    if arm_result.judge_scores_per_brief is not None:
        qpr = quality_pass_rate(arm_result.judge_scores_per_brief)
    else:
        # Fallback for bare ArmResult instances in tests (judge not wired)
        qpr = float("nan")

    return {
        "n_distinct": n_distinct,
        "distinct_ratio": distinct_ratio,
        "mean_per_prompt_vendi": float(cv["mean_within_vendi"]),
        "salad_rate": sr,
        "coherence_rate": cr,
        "quality_pass_rate": qpr,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    arm_result: ArmResult,
    metric_fn: Callable[[ArmResult], dict[str, float]],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """95% bootstrap CIs by resampling over briefs (the unit of replication).

    For each resample: draw n_briefs briefs with replacement, collect their texts
    AND their pre-computed judge scores, re-compute the metrics.
    The CI is the [alpha/2, 1-alpha/2] percentile interval.

    Returns dict mapping metric_name -> (lower, upper).
    NaN metrics (no judge scores) are passed through as (nan, nan).
    """
    rng = np.random.default_rng(seed)
    n_briefs = len(arm_result.texts_per_brief)
    boot_rows: list[dict[str, float]] = []

    for _ in range(n_resamples):
        idxs = rng.integers(0, n_briefs, size=n_briefs)
        resampled_texts = [arm_result.texts_per_brief[i] for i in idxs]
        # Carry pre-computed judge scores along with the resample
        resampled_scores = (
            [arm_result.judge_scores_per_brief[i] for i in idxs]
            if arm_result.judge_scores_per_brief is not None
            else None
        )
        resampled_arm = ArmResult(
            arm=arm_result.arm,
            texts_per_brief=resampled_texts,
            judge_scores_per_brief=resampled_scores,
        )
        boot_rows.append(metric_fn(resampled_arm))

    alpha = 1.0 - ci
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    cis: dict[str, tuple[float, float]] = {}
    for key in boot_rows[0]:
        vals = np.array([row[key] for row in boot_rows], dtype=float)
        if np.all(np.isnan(vals)):
            cis[key] = (float("nan"), float("nan"))
        else:
            cis[key] = (float(np.nanpercentile(vals, lo_p)), float(np.nanpercentile(vals, hi_p)))
    return cis


# ---------------------------------------------------------------------------
# PASS/FAIL verdict
# ---------------------------------------------------------------------------


def verdict(
    arm: str,
    metrics: dict[str, float],
    cis: dict[str, tuple[float, float]],
    baseline_metrics: dict[str, float],
    baseline_cis: dict[str, tuple[float, float]],
    k: int,
) -> tuple[str, list[str]]:
    """Evaluate the pre-registered success criteria for one non-baseline arm.

    Returns (verdict_str, list_of_notes).

    VOID if:
    - coherence_rate drops >2pp vs baseline OR salad_rate > 0.10 (cheap canary, kept)
    - quality_pass_rate drops >5pp vs baseline (LLM judge gate, new)
    """
    notes: list[str] = []

    # -- Void check (quality) -----------------------------------------------
    baseline_cr = baseline_metrics["coherence_rate"]
    arm_cr = metrics["coherence_rate"]
    void_coherence = (baseline_cr - arm_cr) > 0.02  # >2pp drop
    void_salad = metrics["salad_rate"] > 0.10

    # FIX (b): LLM judge quality gate (>5pp drop triggers void)
    baseline_qpr = baseline_metrics.get("quality_pass_rate", float("nan"))
    arm_qpr = metrics.get("quality_pass_rate", float("nan"))
    void_quality = (
        not (np.isnan(baseline_qpr) or np.isnan(arm_qpr))
        and (baseline_qpr - arm_qpr) > 0.05  # >5pp drop
    )

    if void_coherence:
        notes.append(
            f"VOID: coherence_rate dropped {(baseline_cr - arm_cr) * 100:.1f}pp vs baseline "
            f"(>{2}pp threshold)"
        )
    if void_salad:
        notes.append(f"VOID: salad_rate = {metrics['salad_rate']:.3f} > 0.10")
    if void_quality:
        notes.append(
            f"VOID: quality_pass_rate dropped {(baseline_qpr - arm_qpr) * 100:.1f}pp vs baseline "
            f"(>5pp threshold); arm={arm_qpr:.3f} baseline={baseline_qpr:.3f}"
        )

    if void_coherence or void_salad or void_quality:
        return "VOID", notes

    # -- Win criteria -------------------------------------------------------
    # 1. NoveltyBench-Distinct: arm.n_distinct >= baseline.n_distinct + 1 class at K=8
    baseline_n_distinct = baseline_metrics["n_distinct"]
    arm_n_distinct = metrics["n_distinct"]
    nb_win = arm_n_distinct >= baseline_n_distinct + 1.0

    # 2. mean per-prompt Vendi: arm higher than baseline with NON-OVERLAPPING 95% CIs
    arm_vendi_lo, arm_vendi_hi = cis["mean_per_prompt_vendi"]
    base_vendi_lo, base_vendi_hi = baseline_cis["mean_per_prompt_vendi"]
    vendi_higher = metrics["mean_per_prompt_vendi"] > baseline_metrics["mean_per_prompt_vendi"]
    vendi_non_overlapping = arm_vendi_lo > base_vendi_hi
    vendi_win = vendi_higher and vendi_non_overlapping

    if nb_win:
        notes.append(
            f"NoveltyBench OK: n_distinct={arm_n_distinct:.0f} >= baseline {baseline_n_distinct:.0f}+1"
        )
    else:
        notes.append(
            f"NoveltyBench FAIL: n_distinct={arm_n_distinct:.0f} < baseline {baseline_n_distinct:.0f}+1"
        )

    if vendi_win:
        notes.append(
            f"Vendi OK: {metrics['mean_per_prompt_vendi']:.3f} > baseline "
            f"{baseline_metrics['mean_per_prompt_vendi']:.3f}, CIs non-overlapping "
            f"arm=[{arm_vendi_lo:.3f},{arm_vendi_hi:.3f}] base=[{base_vendi_lo:.3f},{base_vendi_hi:.3f}]"
        )
    else:
        notes.append(
            f"Vendi FAIL: arm={metrics['mean_per_prompt_vendi']:.3f} "
            f"base={baseline_metrics['mean_per_prompt_vendi']:.3f} "
            f"arm_CI=[{arm_vendi_lo:.3f},{arm_vendi_hi:.3f}] "
            f"base_CI=[{base_vendi_lo:.3f},{base_vendi_hi:.3f}] "
            f"(overlapping={'YES' if not vendi_non_overlapping else 'no'})"
        )

    overall = "WIN" if (nb_win and vendi_win) else "FAIL"
    return overall, notes


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "n_distinct": "NB-Distinct",
    "distinct_ratio": "Distinct/K",
    "mean_per_prompt_vendi": "Vendi(mean/prompt)",
    "coherence_rate": "Coherence",
    "salad_rate": "Salad%",
    "quality_pass_rate": "QualPassRate",
}


def print_table(
    arm_names: list[str],
    metrics_map: dict[str, dict[str, float]],
    cis_map: dict[str, dict[str, tuple[float, float]]],
) -> None:
    col_w = 22
    metric_keys = list(_METRIC_LABELS.keys())
    header = f"{'ARM':<20}" + "".join(f"{_METRIC_LABELS[k]:>{col_w}}" for k in metric_keys)
    print("\n" + "=" * len(header))
    print("COMPARISON TABLE  (mean ± 95% CI, bootstrapped over briefs)")
    print("  NOTE: Vendi = mean per-prompt Vendi (NOT arXiv:2411.02817 Conditional-Vendi)")
    print("  NOTE: ortho arm is GREEDY; fair comparison is vs antimode/temp, not greedy")
    print("  NOTE: QualPassRate = frac(coherence>=3 AND on_topic>=4) via gpt-4o-mini judge")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for arm in arm_names:
        m = metrics_map[arm]
        ci = cis_map[arm]
        row = f"{arm:<20}"
        for k in metric_keys:
            lo, hi = ci[k]
            val = m[k]
            if np.isnan(val):
                cell = "n/a"
            else:
                cell = f"{val:.3f}[{lo:.3f},{hi:.3f}]"
            row += f"{cell:>{col_w}}"
        print(row)
    print("=" * len(header))


def print_verdict_block(
    arm_names: list[str],
    metrics_map: dict[str, dict[str, float]],
    cis_map: dict[str, dict[str, tuple[float, float]]],
    baseline_arm: str,
    k: int,
) -> None:
    print("\n" + "=" * 70)
    print("PASS/FAIL VERDICT BLOCK  (pre-registered criteria)")
    print("  Baseline (shipping): antimode")
    print("  Win requires: NB-Distinct >= baseline+1 AND Vendi higher w/ non-overlapping CIs")
    print("  Void if: coherence_rate drops >2pp OR salad_rate > 0.10")
    print("  Void if: quality_pass_rate drops >5pp vs baseline (LLM judge gate)")
    print("=" * 70)
    bm = metrics_map[baseline_arm]
    bc = cis_map[baseline_arm]
    for arm in arm_names:
        if arm in (baseline_arm, "greedy"):
            continue
        v, notes = verdict(arm, metrics_map[arm], cis_map[arm], bm, bc, k)
        print(f"\n  [{v}] {arm}")
        for note in notes:
            print(f"       {note}")
    print("\n  [BASELINE] antimode (the shipping stack — all other arms are measured against this)")
    print("  [REF] greedy (sanity-check floor — expect ~1 distinct class)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# FIX (d): De-duplicated alpha sweep orchestration
# ---------------------------------------------------------------------------

# Arms that are alpha-independent (run once regardless of sweep)
_ALPHA_INDEPENDENT_ARMS = ["greedy", "temp", "antimode", "vs", "repulsion", "full"]

# Arms that sweep over alpha (appear as arm@alpha rows in the table)
_ALPHA_DEPENDENT_ARMS = ["ortho", "ortho_temp", "ortho_L17"]


def run_study(
    briefs: list[str],
    k: int,
    arms: list[str],
    *,
    mock: bool = False,
    config_path: str | None = None,
    layer: int | None = None,
    alphas: list[float] | None = None,
    n_resamples: int = 1000,
    temperature: float = 0.8,
    verbose: bool = True,
    checkpoint_path: str | None = None,
) -> dict:
    """Run all arms over all briefs, compute metrics and CIs, return full results dict.

    FIX (d): Alpha-independent arms (greedy/temp/antimode/vs/repulsion/full) generate ONCE.
    Steering arms (ortho/ortho_temp/ortho_L17) loop over alphas, appearing as separate rows
    named arm@alpha (e.g. ortho_L17@1.5).
    """
    if alphas is None or len(alphas) == 0:
        alphas = [0.3]  # backward-compat default for non-L17 arms

    # Separate arms by alpha dependence
    alpha_independent = [a for a in arms if a in _ALPHA_INDEPENDENT_ARMS]
    alpha_dependent = [a for a in arms if a in _ALPHA_DEPENDENT_ARMS]

    # Build the expanded arm list for the table: independent first, then arm@alpha combos
    expanded_arms: list[str] = list(alpha_independent)
    for a in alpha_dependent:
        for alpha in alphas:
            expanded_arms.append(f"{a}@{alpha}")

    # -- Build backend / bank -----------------------------------------------
    backend = None
    bank = None
    l17_vec: np.ndarray | None = None
    l17_skipped = False

    if not mock:
        if config_path is None:
            raise ValueError("--config is required for real (non-mock) runs")
        if verbose:
            print(f"Loading backend from {config_path} ...", flush=True)
        backend = _build_real_backend(config_path)
        if verbose:
            print(f"  model={backend.model_id}  n_layers={backend.num_layers}", flush=True)
        inj_layer = layer if layer is not None else backend.num_layers // 2
        if verbose:
            print(f"  inject_layer={inj_layer}  alphas={alphas}", flush=True)
        if any(a in ("ortho", "ortho_temp") for a in alpha_dependent):
            if verbose:
                print("Building ArchetypeBank ...", flush=True)
            bank = _build_archetype_bank(backend, inj_layer)
            if verbose:
                print(f"  bank.k={bank.k}", flush=True)

        # FIX (c): Load the L17 production vector
        if "ortho_L17" in alpha_dependent:
            _vec_path = _REPO / "data" / "steering_14b" / "hyper_creative_purified_vector_L17.pt"
            if _vec_path.exists():
                import torch
                _raw = torch.load(str(_vec_path), map_location="cpu")
                if isinstance(_raw, torch.Tensor):
                    l17_vec = _raw.float().numpy().reshape(-1)
                elif isinstance(_raw, dict):
                    # Stored as {"vector": tensor, "layer_index": int}. Extract by the
                    # "vector" key, falling back to the first TENSOR value — NEVER a
                    # non-tensor (positional next(iter(values())) would silently grab the
                    # layer_index int if key order ever changed).
                    _t = _raw.get("vector")
                    if not isinstance(_t, torch.Tensor):
                        _t = next((x for x in _raw.values() if isinstance(x, torch.Tensor)), None)
                    if _t is None:
                        raise ValueError(
                            f"L17 .pt dict has no tensor value (keys={list(_raw.keys())})"
                        )
                    l17_vec = _t.float().numpy().reshape(-1)
                else:
                    l17_vec = np.asarray(_raw, dtype=np.float32).reshape(-1)

                # Dim guard: only 14B-compatible (5120-dim)
                if l17_vec.shape[-1] != backend.hidden_dim:
                    print(
                        f"  [ortho_L17] SKIP: vector dim={l17_vec.shape[-1]} != "
                        f"backend.hidden_dim={backend.hidden_dim}. "
                        f"This arm requires the 14B model (5120-dim).",
                        flush=True,
                    )
                    l17_skipped = True
                    l17_vec = None
                else:
                    if verbose:
                        print(f"  [ortho_L17] loaded vector dim={l17_vec.shape[-1]}", flush=True)
            else:
                print(
                    f"  [ortho_L17] SKIP: vector not found at {_vec_path}",
                    flush=True,
                )
                l17_skipped = True
    else:
        inj_layer = layer or 17

    # Remove skipped L17 combos from the table
    if l17_skipped:
        expanded_arms = [a for a in expanded_arms if not a.startswith("ortho_L17@")]

    # -- Generate per arm per brief -----------------------------------------
    arm_results: dict[str, ArmResult] = {}

    def _generate_arm_briefs(arm_key: str, base_arm: str, alpha: float) -> ArmResult:
        """Generate k texts per brief for one (arm, alpha) combination."""
        texts_per_brief: list[list[str]] = []
        for bi, brief in enumerate(briefs):
            if verbose:
                print(
                    f"  brief {bi + 1}/{len(briefs)}: {brief[:55]}...",
                    flush=True, end="\r",
                )
            if mock:
                gen = MockGenerator(arm=arm_key, seed=bi)
                texts = gen.generate_k(brief, k)
            else:
                t0 = time.time()
                if base_arm == "greedy":
                    texts = run_arm_greedy(backend, brief, k)
                elif base_arm == "temp":
                    texts = run_arm_temp(backend, brief, k, temperature=temperature)
                elif base_arm == "antimode":
                    texts = run_arm_antimode(backend, brief, k, temperature=temperature)
                elif base_arm == "ortho":
                    texts = run_arm_ortho(backend, brief, k, inj_layer, alpha, bank)
                elif base_arm == "ortho_temp":
                    texts = run_arm_ortho(
                        backend, brief, k, inj_layer, alpha, bank, temperature=temperature
                    )
                elif base_arm == "ortho_L17":
                    texts = run_arm_ortho_L17(backend, brief, k, alpha, l17_vec)
                elif base_arm == "vs":
                    texts = run_arm_vs(backend, brief, k, temperature=temperature)
                elif base_arm == "repulsion":
                    texts = run_arm_repulsion(backend, brief, k, temperature=temperature)
                elif base_arm == "full":
                    texts = run_arm_full(backend, brief, k, temperature=temperature)
                else:
                    raise ValueError(f"Unknown arm {base_arm!r}")
                if verbose:
                    print(f"  brief {bi + 1} done in {time.time()-t0:.1f}s", flush=True)

            texts = (texts + [""] * k)[:k]
            texts_per_brief.append(texts)

        return ArmResult(arm=arm_key, texts_per_brief=texts_per_brief)

    # Incremental checkpoint of raw generations after EACH arm — a crash (or a missing
    # OPENAI_API_KEY hit at the judge/metrics step) then never loses hours of GPU;
    # re-scoring from this file is free.
    _gen_ckpt = (
        Path(checkpoint_path).with_suffix(".generations.json") if checkpoint_path else None
    )

    def _save_generations() -> None:
        if _gen_ckpt is None:
            return
        _gen_ckpt.parent.mkdir(parents=True, exist_ok=True)
        _gen_ckpt.write_text(
            json.dumps({a: r.texts_per_brief for a, r in arm_results.items()}, indent=2)
        )

    # Run alpha-independent arms once
    if verbose:
        print("\n-- Alpha-independent arms (generated once) --", flush=True)
    for arm in alpha_independent:
        if verbose:
            print(f"\n-- ARM: {arm} --", flush=True)
        ar = _generate_arm_briefs(arm, arm, alpha=alphas[0])
        arm_results[arm] = ar
        _save_generations()
        if verbose:
            print(f"  [done] {arm}: generated {len(briefs)} x {k} texts", flush=True)

    # Run steering arms per alpha
    if alpha_dependent:
        if verbose:
            print(f"\n-- Steering arms over alphas={alphas} --", flush=True)
        for base_arm in alpha_dependent:
            for alpha in alphas:
                arm_key = f"{base_arm}@{alpha}"
                if arm_key not in expanded_arms:
                    continue  # skipped (e.g. ortho_L17 on 0.5B)
                if verbose:
                    print(f"\n-- ARM: {arm_key} --", flush=True)
                ar = _generate_arm_briefs(arm_key, base_arm, alpha)
                arm_results[arm_key] = ar
                _save_generations()
                if verbose:
                    print(f"  [done] {arm_key}: generated {len(briefs)} x {k} texts", flush=True)

    # -- FIX (b): Compute LLM judge scores ONCE before bootstrap --------------
    if verbose:
        print("\nScoring outputs with LLM judge (gpt-4o-mini, cached) ...", flush=True)
    for arm_key, ar in arm_results.items():
        ar.judge_scores_per_brief = score_all_texts(
            ar.texts_per_brief, briefs, mock=mock
        )
        if verbose:
            qpr = quality_pass_rate(ar.judge_scores_per_brief)
            print(f"  {arm_key}: quality_pass_rate={qpr:.3f}", flush=True)

    # -- Compute metrics and bootstrap CIs -----------------------------------
    if verbose:
        print("\nComputing metrics + bootstrap CIs ...", flush=True)
    metrics_map: dict[str, dict[str, float]] = {}
    cis_map: dict[str, dict[str, tuple[float, float]]] = {}

    for arm_key, arm_result in arm_results.items():
        if verbose:
            print(f"  {arm_key} ...", flush=True, end="\r")
        metrics_map[arm_key] = compute_metrics(arm_result)
        cis_map[arm_key] = bootstrap_ci(
            arm_result, compute_metrics, n_resamples=n_resamples
        )
        if verbose:
            print(
                f"  {arm_key}: n_distinct={metrics_map[arm_key]['n_distinct']:.0f}  "
                f"vendi={metrics_map[arm_key]['mean_per_prompt_vendi']:.3f}  "
                f"qpr={metrics_map[arm_key]['quality_pass_rate']:.3f}",
                flush=True,
            )

    return {
        "arms": expanded_arms,
        "briefs": briefs,
        "k": k,
        "metrics": metrics_map,
        "cis": {
            arm: {key: list(v) for key, v in ci.items()}
            for arm, ci in cis_map.items()
        },
        "arm_texts": {
            arm: arm_results[arm].texts_per_brief
            for arm in expanded_arms
            if arm in arm_results
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_DEFAULT_ARMS = [
    "greedy", "temp", "antimode",
    "ortho", "ortho_temp", "ortho_L17",
    "vs", "repulsion", "full",
]

# Default alphas for on-the-fly bank arms (ortho/ortho_temp) when not specified.
# L17 validated alpha is 1.5; the full sweep is [1.0, 1.5, 2.0, 2.62].
_DEFAULT_ALPHAS_L17 = [1.0, 1.5, 2.0, 2.62]


def _preflight(*, config_path: str | None, out: str | None, layer: int | None) -> None:
    """Fail-fast checks BEFORE the multi-hour generation (real-run only).

    Catches the catastrophic 'generate for hours, THEN crash at the judge/metrics step'
    failure: a missing/invalid OPENAI_API_KEY, an unparseable config, a missing/garbled L17
    vector, or an unwritable --out path — all in well under a minute, before any GPU work.
    """
    print("-- PREFLIGHT (fail-fast before any generation) --", flush=True)

    # 1) OpenAI connectivity — judge + embeddings both depend on it. A LIVE call is the ground
    #    truth (the key may resolve from env, a .env, or settings); don't hard-check os.environ,
    #    which false-fails when the client sources the key elsewhere.
    try:
        from brainstormlm.clients.openai import embed

        _m, _ = embed(["preflight connectivity test"])
        _s = _judge_one("Preflight brief.", "A concrete, coherent test idea.", use_cache=False)
        assert isinstance(_s.get("coherence"), int) and isinstance(_s.get("on_topic"), int)
        print(f"   [ok] OpenAI live (embed dim={_m.shape[-1]}, judge={_s})", flush=True)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "PREFLIGHT FAIL: OpenAI live check failed — the judge + embeddings need a working "
            "OPENAI_API_KEY (set via `export OPENAI_API_KEY=...` or a .env). "
            f"({type(exc).__name__}: {exc})"
        )

    # 2) Config parses (build_backend will load the model after this — that's the slow part).
    try:
        from creativity_steering.config import load_config

        _cfg = load_config(config_path)
        print(f"   [ok] config loads: model={_cfg.model_id}, dtype={getattr(_cfg, 'dtype', None)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"PREFLIGHT FAIL: cannot load config {config_path!r} ({type(exc).__name__}: {exc})")

    # 3) L17 vector exists + extracts to a finite, non-zero tensor (else the arm just skips).
    import torch

    _vp = _REPO / "data" / "steering_14b" / "hyper_creative_purified_vector_L17.pt"
    if not _vp.exists():
        print(f"   [warn] L17 vector missing at {_vp} — ortho_L17 arm will be SKIPPED.", flush=True)
    else:
        _raw = torch.load(str(_vp), map_location="cpu")
        if isinstance(_raw, dict):
            _t = _raw.get("vector")
            if not isinstance(_t, torch.Tensor):
                _t = next((x for x in _raw.values() if isinstance(x, torch.Tensor)), None)
            _li = _raw.get("layer_index")
        else:
            _t, _li = (_raw if isinstance(_raw, torch.Tensor) else None), None
        if _t is None:
            raise SystemExit("PREFLIGHT FAIL: L17 .pt contains no tensor value.")
        _v = _t.float().numpy().reshape(-1)
        if not np.isfinite(_v).all() or np.allclose(_v, 0):
            raise SystemExit("PREFLIGHT FAIL: L17 vector is non-finite or all-zero.")
        print(
            f"   [ok] L17 vector: dim={_v.shape[0]}, norm={float(np.linalg.norm(_v)):.3f}, "
            f"layer_index={_li}",
            flush=True,
        )
        if layer is not None and _li is not None and layer != _li:
            print(
                f"   [warn] --layer {layer} != vector layer_index {_li}; "
                f"run_arm_ortho_L17 injects at 17 regardless. Use --layer 17.",
                flush=True,
            )

    # 4) Output path writable.
    if out:
        _op = Path(out)
        try:
            _op.parent.mkdir(parents=True, exist_ok=True)
            _tf = _op.parent / ".preflight_write_test"
            _tf.write_text("ok")
            _tf.unlink()
            print(f"   [ok] output dir writable: {_op.parent}", flush=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"PREFLIGHT FAIL: cannot write to {_op.parent} ({exc})")

    print("-- PREFLIGHT OK --\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="A/B diversity validation harness for Conceptial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mock", action="store_true", help="Use mock generator (no model needed)")
    ap.add_argument("--config", default=None, help="ExperimentConfig JSON for real 14B run")
    ap.add_argument("--layer", type=int, default=None, help="Injection layer for ortho arm")
    ap.add_argument(
        "--alphas", default=None,
        help="Comma-separated alpha values for steering arms, e.g. 1.0,1.5,2.0,2.62. "
             "Default: 1.0,1.5,2.0,2.62 for L17; 0.3,0.5,0.8,1.0 otherwise."
    )
    # Legacy single-alpha flag for backward compat (overridden by --alphas if both given)
    ap.add_argument("--alpha", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--k", type=int, default=8, help="Outputs per brief per arm")
    ap.add_argument(
        "--briefs", default=None,
        help="JSON array of brief strings, or path to JSON file. Default: builtin 40 briefs"
    )
    ap.add_argument("--n-briefs", type=int, default=None, help="Use first N builtin briefs")
    ap.add_argument("--arms", nargs="+", default=_DEFAULT_ARMS, help="Arms to run")
    ap.add_argument("--out", default=None, help="Write results to this JSON path")
    ap.add_argument("--n-resamples", type=int, default=1000, help="Bootstrap resamples")
    ap.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampled arms")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    # -- Resolve alphas
    if args.alphas is not None:
        alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    elif args.alpha is not None:
        alphas = [args.alpha]
    else:
        # Default: L17 range
        alphas = _DEFAULT_ALPHAS_L17

    # -- Resolve briefs
    if args.briefs is not None:
        p = Path(args.briefs)
        if p.exists():
            briefs = json.loads(p.read_text())
        else:
            briefs = json.loads(args.briefs)
    else:
        briefs = BUILTIN_BRIEFS
    if args.n_briefs:
        briefs = briefs[: args.n_briefs]

    if len(briefs) < 2:
        print("ERROR: need at least 2 briefs for bootstrap CI over briefs.", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("Conceptial A/B Diversity Validation Harness")
    print("=" * 70)
    print(f"  mode   : {'MOCK (no model)' if args.mock else 'REAL MODEL'}")
    print(f"  k      : {args.k}")
    print(f"  briefs : {len(briefs)}")
    print(f"  arms   : {args.arms}")
    print(f"  alphas : {alphas}  (steering arms only; alpha-independent arms run once)")
    print(f"  resamples: {args.n_resamples}")
    if not args.mock:
        print(f"  config : {args.config}")
        print(f"  layer  : {args.layer}")
    print("  HONESTY: ortho arm is greedy; Vendi = mean per-prompt (not arXiv:2411.02817)")
    print("  QUALITY: LLM judge (gpt-4o-mini) scores coherence+on_topic; cached by request")
    print("  L17 NOTE: ortho_L17 requires 14B model (5120-dim); skipped on 0.5B (dim guard)")
    print("=" * 70 + "\n")

    # Fail-fast BEFORE any GPU generation (real runs only).
    if not args.mock:
        _preflight(config_path=args.config, out=args.out, layer=args.layer)

    results = run_study(
        briefs=briefs,
        k=args.k,
        arms=args.arms,
        mock=args.mock,
        config_path=args.config,
        layer=args.layer,
        alphas=alphas,
        n_resamples=args.n_resamples,
        temperature=args.temperature,
        checkpoint_path=args.out,
        verbose=not args.quiet,
    )

    # Rebuild typed dicts from serializable results
    metrics_map: dict[str, dict[str, float]] = results["metrics"]
    cis_map: dict[str, dict[str, tuple[float, float]]] = {
        arm: {key: tuple(v) for key, v in ci.items()}
        for arm, ci in results["cis"].items()
    }

    print_table(results["arms"], metrics_map, cis_map)

    if "antimode" in results["arms"]:
        print_verdict_block(results["arms"], metrics_map, cis_map, baseline_arm="antimode", k=args.k)
    else:
        print("\n(antimode arm not in run — skipping PASS/FAIL verdict block)")

    print("\n14B POD COMMAND (do NOT run without GPU):")
    print(
        "  uv run python scripts/validate_diversity.py \\\n"
        "      --config creativity_steering/configs/runpod_qwen14b_v2.json \\\n"
        "      --layer 17 --alphas 1.0,1.5,2.0,2.62 --k 8 \\\n"
        "      --out results/validate_14b_L17_sweep.json"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
