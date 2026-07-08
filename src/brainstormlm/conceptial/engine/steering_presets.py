"""
Steering presets for local Representation Engine paths: unsteered control, dual-layer purified
schedules, and v7–v12-style variants.

Legacy presets and older experiment variants live in ``legacy/reference/steering_presets_archive.py``.
"""

from __future__ import annotations

from pathlib import Path

from brainstormlm.conceptial.engine.pipeline import SteeringEngine, apply_layer_injections, load_steering_checkpoint


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
