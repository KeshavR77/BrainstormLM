"""
Steering presets for local Representation Engine paths: unsteered control, dual-layer purified
schedules, and v7–v12-style variants.

Legacy presets and older experiment variants live in ``legacy/reference/steering_presets_archive.py``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from brainstormlm.conceptial.engine.config import SteeringStrategy
from brainstormlm.conceptial.engine.pipeline import SteeringEngine, apply_layer_injections, load_steering_checkpoint


def default_dual_injection_layer_pair(n_layers: int) -> tuple[int, int]:
    """
    Map two default creative/technical injection depths (Qwen2.5-3B-style: ~layers 14 and 18 on a 36-layer stack).

    Uses ~40% and ~51% of the last layer index, clamped to ``[0, n_layers - 1]``.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    last = n_layers - 1
    creative = int(round(0.40 * last))
    technical = int(round(0.51 * last))
    creative = max(0, min(last, creative))
    technical = max(0, min(last, technical))
    if creative == technical and last > 0:
        technical = min(last, creative + 1)
    return creative, technical


def apply_local_unsteered(engine: SteeringEngine) -> None:
    engine.config.layer_injections = []
    engine.config.strategy = SteeringStrategy.CONTROL
    engine.config.layer_alphas = {}
    engine.config.decay_tokens = 0
    engine.set_steering_vectors({})


def apply_ortho_purified_phase_shift(
    engine: SteeringEngine,
    steering_dir: Path,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    creative_start_alpha: float = 2.5,
    technical_start_alpha: float = 0.75,
    exp_zero_by_step: int = 100,
    sigmoid_mid_step: float = 50.0,
    sigmoid_steepness: float = 0.12,
    steering_math: str = "additive",
) -> None:
    """
    **Local ortho-purified preset:** deep-purified creative with exponential α decay on one decoder layer;
    technical-purified with sigmoid ramp on a second layer.

    Default filenames assume **layers 14 and 18** (Qwen2.5-3B-style). For other depths (e.g. Gemma),
    pass ``creative_vector_path`` / ``technical_vector_path`` and optionally ``creative_layer`` /
    ``technical_layer``. When layer ints are omitted, they are read from each checkpoint's
    ``layer_index`` field.

    ``*_vector_path`` may be ``str`` (e.g. from JSON) or :class:`pathlib.Path`.
    """
    steering_dir = Path(steering_dir)
    if isinstance(creative_vector_path, str):
        creative_vector_path = Path(creative_vector_path)
    if isinstance(technical_vector_path, str):
        technical_vector_path = Path(technical_vector_path)
    c_path = creative_vector_path or (steering_dir / "creative_deep_purified_vector_L14.pt")
    t_path = technical_vector_path or (steering_dir / "technical_purified_vector_L18.pt")
    if not c_path.is_file():
        raise FileNotFoundError(f"Missing creative (purified) checkpoint: {c_path}")
    if not t_path.is_file():
        raise FileNotFoundError(f"Missing technical checkpoint: {t_path}")
    c_bundle = load_steering_checkpoint(c_path, map_location="cpu")
    t_bundle = load_steering_checkpoint(t_path, map_location="cpu")
    c_layer = int(creative_layer) if creative_layer is not None else int(c_bundle["layer_index"])
    t_layer = int(technical_layer) if technical_layer is not None else int(t_bundle["layer_index"])
    if int(c_bundle["layer_index"]) != c_layer:
        raise ValueError(
            f"creative_layer={c_layer} does not match checkpoint layer_index={c_bundle['layer_index']} ({c_path})"
        )
    if int(t_bundle["layer_index"]) != t_layer:
        raise ValueError(
            f"technical_layer={t_layer} does not match checkpoint layer_index={t_bundle['layer_index']} ({t_path})"
        )
    mode = (steering_math or "additive").strip().lower()
    if mode not in ("additive", "spherical"):
        raise ValueError(f"steering_math must be 'additive' or 'spherical', got {steering_math!r}")
    engine.config.layer_injections = [
        {
            "layer": c_layer,
            "vector_path": str(c_path.resolve()),
            "start_alpha": float(creative_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "exp_decay",
            "exp_zero_by_step": int(exp_zero_by_step),
        },
        {
            "layer": t_layer,
            "vector_path": str(t_path.resolve()),
            "start_alpha": float(technical_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "sigmoid_ramp",
            "sigmoid_mid_step": float(sigmoid_mid_step),
            "sigmoid_steepness": float(sigmoid_steepness),
        },
    ]
    engine.config.layer_alphas = {c_layer: float(creative_start_alpha), t_layer: float(technical_start_alpha)}
    engine.config.decay_tokens = 0
    apply_layer_injections(engine)


def apply_v8_idea_forward_phase_shift(
    engine: SteeringEngine,
    steering_dir: Path,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    creative_start_alpha: float = 2.1,
    exp_zero_by_step: int = 180,
    technical_start_alpha: float = 0.48,
    sigmoid_mid_step: float = 100.0,
    sigmoid_steepness: float = 0.078,
    steering_math: str = "spherical",
) -> None:
    """
    **v8:** idea-forward schedule for CreativityPrism-style rubrics.

    Empirically, a **linear** creative fade underperformed v7 on IDEA Prism (see
    ``Cursor/Archive-long-chats-workflow.md``). v8 matches v7's creative exponential schedule
    exactly (same α and ``exp_zero_by_step``) but uses a **later, slightly softer** technical
    sigmoid than v7 so task structure arrives after the main ideation band.
    """
    steering_dir = Path(steering_dir)
    if isinstance(creative_vector_path, str):
        creative_vector_path = Path(creative_vector_path)
    if isinstance(technical_vector_path, str):
        technical_vector_path = Path(technical_vector_path)
    c_path = creative_vector_path or (steering_dir / "creative_deep_purified_vector_L14.pt")
    t_path = technical_vector_path or (steering_dir / "technical_purified_vector_L18.pt")
    if not c_path.is_file():
        raise FileNotFoundError(f"Missing creative checkpoint: {c_path}")
    if not t_path.is_file():
        raise FileNotFoundError(f"Missing technical checkpoint: {t_path}")
    c_bundle = load_steering_checkpoint(c_path, map_location="cpu")
    t_bundle = load_steering_checkpoint(t_path, map_location="cpu")
    c_layer = int(creative_layer) if creative_layer is not None else int(c_bundle["layer_index"])
    t_layer = int(technical_layer) if technical_layer is not None else int(t_bundle["layer_index"])
    if int(c_bundle["layer_index"]) != c_layer:
        raise ValueError(
            f"creative_layer={c_layer} does not match checkpoint layer_index={c_bundle['layer_index']} ({c_path})"
        )
    if int(t_bundle["layer_index"]) != t_layer:
        raise ValueError(
            f"technical_layer={t_layer} does not match checkpoint layer_index={t_bundle['layer_index']} ({t_path})"
        )
    mode = (steering_math or "additive").strip().lower()
    if mode not in ("additive", "spherical"):
        raise ValueError(f"steering_math must be 'additive' or 'spherical', got {steering_math!r}")
    ez = max(1, int(exp_zero_by_step))
    engine.config.layer_injections = [
        {
            "layer": c_layer,
            "vector_path": str(c_path.resolve()),
            "start_alpha": float(creative_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "exp_decay",
            "exp_zero_by_step": ez,
        },
        {
            "layer": t_layer,
            "vector_path": str(t_path.resolve()),
            "start_alpha": float(technical_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "sigmoid_ramp",
            "sigmoid_mid_step": float(sigmoid_mid_step),
            "sigmoid_steepness": float(sigmoid_steepness),
        },
    ]
    engine.config.layer_alphas = {c_layer: float(creative_start_alpha), t_layer: float(technical_start_alpha)}
    engine.config.decay_tokens = 0
    apply_layer_injections(engine)


def apply_v9_hybrid_phase_shift(
    engine: SteeringEngine,
    steering_dir: Path,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    creative_start_alpha: float = 2.1,
    exp_zero_by_step: int = 180,
    technical_start_alpha: float = 0.55,
    sigmoid_mid_step: float = 72.0,
    sigmoid_steepness: float = 0.085,
    steering_math: str = "spherical",
) -> None:
    """
    **v9:** v8-style creative exp decay + v7 technical sigmoid (earlier, stronger anchor).

    Hypothesis: v8’s later/softer technical ramp can leave more room for embedding diversity
    (Vendi); v7’s earlier ``sigmoid_mid_step`` may collapse diversity. v9 is the cross for
    A/B against v8 on Vendi and Prism-IDEA — tune defaults from benchmarks, do not treat as
    a separate vector family (same .pt files as v7/v8 by default).
    """
    apply_v8_idea_forward_phase_shift(
        engine,
        steering_dir,
        creative_vector_path=creative_vector_path,
        technical_vector_path=technical_vector_path,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        creative_start_alpha=creative_start_alpha,
        exp_zero_by_step=exp_zero_by_step,
        technical_start_alpha=technical_start_alpha,
        sigmoid_mid_step=sigmoid_mid_step,
        sigmoid_steepness=sigmoid_steepness,
        steering_math=steering_math,
    )


def apply_v10_logic_lock_phase_shift(
    engine: SteeringEngine,
    steering_dir: Path,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    creative_start_alpha: float = 2.0,
    exp_zero_by_step: int = 132,
    technical_start_alpha: float = 0.68,
    sigmoid_mid_step: float = 62.0,
    sigmoid_steepness: float = 0.1,
    steering_math: str = "spherical",
) -> None:
    """
    **v10 (idea 1):** front-loaded creative burst with earlier technical lock-in.

    Compared to v8/v9 defaults, v10 decays creative pressure earlier and ramps technical pressure
    sooner to preserve logical structure in longer completions while still encouraging novelty.
    """
    apply_v8_idea_forward_phase_shift(
        engine,
        steering_dir,
        creative_vector_path=creative_vector_path,
        technical_vector_path=technical_vector_path,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        creative_start_alpha=creative_start_alpha,
        exp_zero_by_step=exp_zero_by_step,
        technical_start_alpha=technical_start_alpha,
        sigmoid_mid_step=sigmoid_mid_step,
        sigmoid_steepness=sigmoid_steepness,
        steering_math=steering_math,
    )


V10_PARAMETRIC_PRESET_KEYS: tuple[str, ...] = tuple(f"v{n}" for n in range(13, 21))

# Grid on the v10 (logic-lock) spine: same vector bundles as v9/v10; only the schedule moves.
# Tuples: (creative_start_alpha, exp_zero_by_step, technical_start_alpha, sigmoid_mid_step, sigmoid_steepness)
V10_PARAMETRIC_SCHEDULES: dict[str, tuple[float, int, float, float, float]] = {
    "v13": (2.12, 148, 0.64, 59.0, 0.10),
    "v14": (1.88, 120, 0.72, 55.0, 0.12),
    "v15": (2.25, 125, 0.62, 65.0, 0.085),
    "v16": (1.95, 140, 0.58, 70.0, 0.09),
    "v17": (2.05, 132, 0.75, 58.0, 0.11),
    "v18": (2.18, 115, 0.65, 68.0, 0.095),
    "v19": (1.92, 136, 0.70, 60.0, 0.088),
    "v20": (2.08, 155, 0.60, 64.0, 0.082),
}


def apply_v10_parametric_preset(
    engine: SteeringEngine,
    steering_dir: Path,
    preset_key: str,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    steering_math: str = "spherical",
    **schedule_overrides: float | int,
) -> None:
    """
    Apply a v13–v20 exploration preset: v10-style creative decay + technical sigmoid with tuned
    tradeoffs (activation / RepE steering as the diversity lever when decoding is constrained).
    """
    if preset_key not in V10_PARAMETRIC_SCHEDULES:
        allowed = ", ".join(V10_PARAMETRIC_PRESET_KEYS)
        raise ValueError(f"Unknown v10 parametric preset {preset_key!r}; expected one of: {allowed}")

    c0, e0, t0, m0, s0 = V10_PARAMETRIC_SCHEDULES[preset_key]
    sched: dict[str, float | int] = {
        "creative_start_alpha": c0,
        "exp_zero_by_step": int(e0),
        "technical_start_alpha": t0,
        "sigmoid_mid_step": m0,
        "sigmoid_steepness": s0,
    }
    for k, raw in schedule_overrides.items():
        if k not in sched:
            continue
        if k == "exp_zero_by_step":
            sched[k] = int(raw)
        elif k in ("creative_start_alpha", "technical_start_alpha", "sigmoid_mid_step", "sigmoid_steepness"):
            sched[k] = float(raw)
    apply_v10_logic_lock_phase_shift(
        engine,
        steering_dir,
        creative_vector_path=creative_vector_path,
        technical_vector_path=technical_vector_path,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        steering_math=steering_math,
        **sched,
    )


def apply_layer_pair_micro_ensemble(
    engine: SteeringEngine,
    *,
    candidates: list[dict[str, Any]],
    choice_seed: int,
    steering_math: str = "spherical",
    selection_index: int | None = None,
    v10_schedule: Mapping[str, Any] | None = None,
) -> dict[str, int | str]:
    """
    **Idea 4 helper:** deterministically choose one layer/vector pair candidate per sample.

    Each candidate dict must include:
    - ``creative_vector_path``: path to creative checkpoint
    - ``technical_vector_path``: path to technical checkpoint
    - optional ``creative_layer`` / ``technical_layer`` (defaults to checkpoint layer_index)

    If ``selection_index`` is set, ``idx = selection_index % len(candidates)`` (round-robin /
    deterministic policy). Otherwise ``choice_seed`` drives pseudo-random selection.

    ``v10_schedule`` overrides are forwarded to :func:`apply_v10_logic_lock_phase_shift` (idea5
    adaptive clamps, A/B schedule tweaks) after the candidate is chosen.

    Returns metadata about the selected candidate for logging.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one layer pair candidate")
    if selection_index is not None:
        idx = int(selection_index) % len(candidates)
    else:
        rng = random.Random(int(choice_seed))
        idx = rng.randrange(len(candidates))
    chosen = candidates[idx]
    c_path = Path(str(chosen["creative_vector_path"]))
    t_path = Path(str(chosen["technical_vector_path"]))
    c_layer = int(chosen["creative_layer"]) if "creative_layer" in chosen else None
    t_layer = int(chosen["technical_layer"]) if "technical_layer" in chosen else None
    sched: dict[str, Any] = dict(v10_schedule) if v10_schedule else {}
    apply_v10_logic_lock_phase_shift(
        engine,
        steering_dir=c_path.parent,
        creative_vector_path=c_path,
        technical_vector_path=t_path,
        creative_layer=c_layer,
        technical_layer=t_layer,
        steering_math=steering_math,
        **sched,
    )
    return {
        "candidate_index": idx,
        "creative_vector_path": str(c_path.resolve()),
        "technical_vector_path": str(t_path.resolve()),
        "creative_layer": int(engine.config.layer_injections[0]["layer"]),
        "technical_layer": int(engine.config.layer_injections[1]["layer"]),
    }


def apply_v12_stack(
    engine: SteeringEngine,
    *,
    candidates: list[dict[str, Any]],
    choice_seed: int,
    steering_math: str = "spherical",
    selection_index: int | None = None,
    v10_schedule: Mapping[str, Any] | None = None,
) -> dict[str, int | str]:
    """
    **v12 (integrated stack):** micro-ensemble **policy** (candidate / layer-pair choice) fused with
    **v10 logic-lock schedules** on the selected pair—the same spine as ``qwen_v10``, not a separate
    decoding recipe.

    Intended benchmark bundle (``--local-variants v12`` + ``--idea-experiment idea12``):

    - **Micro policy:** ``--micro-selection-mode round_robin`` (or idea12 forcing round-robin index).
    - **v10 body:** exp creative decay + sigmoid technical ramp on the chosen checkpoint pair.
    - **Idea3 / CE rerank:** ``--logic-penalty-weight`` / ``--rerank-objective creativity_evaluator``
      at decode time (benchmark owns candidate selection across ``diversity_candidates``).
    - **Idea5 guardrails:** ``--adaptive-steering`` or idea12 enables adaptive clamps; ``v10_schedule``
      carries the per-sample overrides into this call (parity with ``qwen_v10``).

    Falsification: arm should not beat v10 on coherence while losing diversity unless the micro policy
    or rerank objective—not a different phase curve—is doing the work.
    """
    meta = apply_layer_pair_micro_ensemble(
        engine,
        candidates=candidates,
        choice_seed=choice_seed,
        steering_math=steering_math,
        selection_index=selection_index,
        v10_schedule=v10_schedule,
    )
    return {**meta, "v12_stack": True}


def apply_v11_dual_creative_coordination(
    engine: SteeringEngine,
    *,
    candidates: list[dict[str, Any]],
    choice_seed: int,
    steering_math: str = "spherical",
    creative_start_alpha: float = 2.0,
    exp_zero_by_step: int = 132,
    technical_start_alpha: float = 0.68,
    sigmoid_mid_step: float = 62.0,
    sigmoid_steepness: float = 0.1,
    creative_secondary_alpha_scale: float = 0.55,
    creative_secondary_exp_fraction: float = 0.5,
) -> dict[str, object]:
    """
    **v11 — dual-layer creative coordination + CE-hybrid phase curve**

    **Hypothesis.** Frontiers models collapse toward safe modes when decoding is constrained; we need
    *multi-scale* activation pressure so rerankers (heuristic or **CreativityEvaluator / CE**) see a
    wider pool of coherent candidates. v11 injects two **staggered exp-decay** creatives at distinct
    decoder depths, then a single **v10-class sigmoid** technical anchor on the primary pair. The
    secondary creative uses a shorter horizon ( hybrids “explore → consolidate” in phase space,
    analogous to a diversity-then-coherence objective without a second forward pass.

    **Schedule (CE-hybrid curve).** Primary creative: full ``creative_start_alpha`` with
    ``exp_zero_by_step``. Secondary creative: ``creative_secondary_alpha_scale ×`` primary α on the
    alternate layer, with zero-crossing at ``max(24, int(exp_zero_by_step * creative_secondary_exp_fraction))``.
    Technical: same sigmoid ramp as v10 on the primary technical vector.

    **Requirements.** ≥2 micro-candidates with distinct ``creative_layer`` (from checkpoints or dict).
    Otherwise falls back to :func:`apply_v10_logic_lock_phase_shift` on the primary pair (no dual path).

    **Falsification.** A/B against ``qwen_v10`` with identical vectors: if Vendi / CE scores move
    with no improvement in Prism-IDEA substance, the dual-creative hypothesis fails and we keep v10.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one layer pair candidate")
    n = len(candidates)
    rng = random.Random(int(choice_seed))
    idx_a = rng.randrange(n) if n > 1 else 0
    idx_b = (idx_a + 1) % n
    ca = dict(candidates[idx_a])
    cb = dict(candidates[idx_b])

    def _layers(entry: dict[str, Any]) -> tuple[Path, Path, int | None, int | None]:
        c_path = Path(str(entry["creative_vector_path"]))
        t_path = Path(str(entry["technical_vector_path"]))
        c_lo = int(entry["creative_layer"]) if "creative_layer" in entry else None
        t_lo = int(entry["technical_layer"]) if "technical_layer" in entry else None
        return c_path, t_path, c_lo, t_lo

    ca_c, ca_t, ca_cl, ca_tl = _layers(ca)
    cb_c, _, cb_cl, _ = _layers(cb)

    if n < 2:
        apply_v10_logic_lock_phase_shift(
            engine,
            steering_dir=ca_c.parent,
            creative_vector_path=ca_c,
            technical_vector_path=ca_t,
            creative_layer=ca_cl,
            technical_layer=ca_tl,
            creative_start_alpha=creative_start_alpha,
            exp_zero_by_step=exp_zero_by_step,
            technical_start_alpha=technical_start_alpha,
            sigmoid_mid_step=sigmoid_mid_step,
            sigmoid_steepness=sigmoid_steepness,
            steering_math=steering_math,
        )
        return {"mode": "fallback_v10_single", "primary_index": idx_a}

    bc_a = load_steering_checkpoint(ca_c, map_location="cpu")
    bc_b = load_steering_checkpoint(cb_c, map_location="cpu")
    bt_a = load_steering_checkpoint(ca_t, map_location="cpu")
    lc_a = int(ca_cl) if ca_cl is not None else int(bc_a["layer_index"])
    lt_a = int(ca_tl) if ca_tl is not None else int(bt_a["layer_index"])
    lc_b = int(cb_cl) if cb_cl is not None else int(bc_b["layer_index"])
    if lc_a == lc_b:
        for j in range(n):
            alt = dict(candidates[(idx_a + j) % n])
            _, _, alt_cl, _ = _layers(alt)
            b_alt = load_steering_checkpoint(Path(str(alt["creative_vector_path"])), map_location="cpu")
            lc_alt = int(alt_cl) if alt_cl is not None else int(b_alt["layer_index"])
            if lc_alt != lc_a:
                cb = alt
                cb_c = Path(str(alt["creative_vector_path"]))
                bc_b = b_alt
                lc_b = lc_alt
                idx_b = (idx_a + j) % n
                break
        else:
            apply_v10_logic_lock_phase_shift(
                engine,
                steering_dir=ca_c.parent,
                creative_vector_path=ca_c,
                technical_vector_path=ca_t,
                creative_layer=ca_cl,
                technical_layer=ca_tl,
                creative_start_alpha=creative_start_alpha,
                exp_zero_by_step=exp_zero_by_step,
                technical_start_alpha=technical_start_alpha,
                sigmoid_mid_step=sigmoid_mid_step,
                sigmoid_steepness=sigmoid_steepness,
                steering_math=steering_math,
            )
            return {"mode": "fallback_v10_no_second_layer", "primary_index": idx_a}

    if int(bc_a["layer_index"]) != lc_a or int(bc_b["layer_index"]) != lc_b or int(bt_a["layer_index"]) != lt_a:
        raise ValueError("creative/technical layer_index mismatch against v11 candidate dict")

    mode = (steering_math or "additive").strip().lower()
    if mode not in ("additive", "spherical"):
        raise ValueError(f"steering_math must be 'additive' or 'spherical', got {steering_math!r}")
    ez = max(1, int(exp_zero_by_step))
    frac = float(creative_secondary_exp_fraction)
    if not 0.0 < frac <= 1.0:
        raise ValueError("creative_secondary_exp_fraction must be in (0, 1]")
    ez2 = max(24, int(ez * frac))
    scale2 = float(creative_secondary_alpha_scale)
    if not 0.0 <= scale2 <= 2.0:
        raise ValueError("creative_secondary_alpha_scale must be in [0, 2]")
    a2 = float(creative_start_alpha) * scale2

    engine.config.layer_injections = [
        {
            "layer": lc_a,
            "vector_path": str(ca_c.resolve()),
            "start_alpha": float(creative_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "exp_decay",
            "exp_zero_by_step": ez,
        },
        {
            "layer": lc_b,
            "vector_path": str(cb_c.resolve()),
            "start_alpha": a2,
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "exp_decay",
            "exp_zero_by_step": ez2,
        },
        {
            "layer": lt_a,
            "vector_path": str(ca_t.resolve()),
            "start_alpha": float(technical_start_alpha),
            "decay_tokens": 0,
            "injection_site": "residual",
            "steering_math": mode,
            "alpha_schedule": "sigmoid_ramp",
            "sigmoid_mid_step": float(sigmoid_mid_step),
            "sigmoid_steepness": float(sigmoid_steepness),
        },
    ]
    engine.config.layer_alphas = {
        lc_a: float(creative_start_alpha),
        lc_b: a2,
        lt_a: float(technical_start_alpha),
    }
    engine.config.decay_tokens = 0
    apply_layer_injections(engine)
    return {
        "mode": "v11_dual_creative",
        "primary_index": idx_a,
        "secondary_index": idx_b,
        "creative_layer_primary": lc_a,
        "creative_layer_secondary": lc_b,
        "technical_layer": lt_a,
        "creative_secondary_alpha_scale": scale2,
        "creative_secondary_exp_zero_by_step": ez2,
    }


def apply_v7_balanced_phase_shift(
    engine: SteeringEngine,
    steering_dir: Path,
    *,
    creative_vector_path: Path | None = None,
    technical_vector_path: Path | None = None,
    creative_layer: int | None = None,
    technical_layer: int | None = None,
    creative_start_alpha: float = 2.1,
    technical_start_alpha: float = 0.55,
    exp_zero_by_step: int = 180,
    sigmoid_mid_step: float = 72.0,
    sigmoid_steepness: float = 0.085,
    steering_math: str = "spherical",
) -> None:
    """
    v7 balanced preset tuned for creative spread with controlled technical coherence.

    Defaults are intentionally gentler than the base ortho-purified preset: slightly lower creative alpha but slower
    decay, and a softer technical ramp. This tends to reduce early token collapse while
    preserving task-anchoring structure.
    """
    apply_ortho_purified_phase_shift(
        engine,
        steering_dir,
        creative_vector_path=creative_vector_path,
        technical_vector_path=technical_vector_path,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        creative_start_alpha=creative_start_alpha,
        technical_start_alpha=technical_start_alpha,
        exp_zero_by_step=exp_zero_by_step,
        sigmoid_mid_step=sigmoid_mid_step,
        sigmoid_steepness=sigmoid_steepness,
        steering_math=steering_math,
    )
