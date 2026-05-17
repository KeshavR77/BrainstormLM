#!/usr/bin/env python3
"""RunPod-side batch generator for the 14B head-to-head benchmark — FULL STACK.

Reads the prompt suite from ``config/benchmark_creative_briefs.json`` and runs
N outputs per prompt against ``Qwen/Qwen2.5-14B-Instruct`` with the complete
Conceptial stack:

- **CAA steering vectors** at L19 (creative) and L24 (technical), loaded from
  the artifacts produced by ``scripts/extract_14b_vectors.py``.
- **Orthogonal-projection purification** — already baked into the loaded
  vectors via ``hyper_purify_vector``.
- **v17 alpha schedule** — exponential decay on creative, sigmoid ramp on
  technical (same schedule as Corpius 1; may need re-tuning at 14B).
- **Iterative anti-mode L19 residual orthogonalization** — captures each
  sample's first generated token's L19 residual; for sample k>=1, the creative
  vector is re-purified orthogonal to the mean prior residual before generation.
- **Prompt-level anti-mode** — exclusion list of prior concept summaries.
- **Persistent project memory** — SQLite per-project store, reset before each
  prompt's batch.
- **Min-p sampling + temperature jitter** — base_temp 0.8 ± 0.05, min_p 0.01.

Usage on a RunPod A100 80GB pod::

    # 0. Pull vectors back into the repo (run from laptop, one time)
    #    after running scripts/extract_14b_vectors.py on the pod
    #    -> data/steering_14b/*.pt + manifests

    # 1. Clone + setup (see docs/cloud_runpod_pivot.md)
    cd /workspace/BrainstormLM
    uv sync
    export HF_TOKEN=hf_...

    # 2. Extract 14B vectors (one time, ~15 min on A100)
    uv run python scripts/extract_14b_vectors.py

    # 3. Run the batched generation (10 prompts × N=8 ≈ 12-18 min on A100)
    uv run python scripts/runpod_batch_14b.py --n 8

    # 4. scp the results back to local
    #    artifacts/benchmark_v2/conceptial_14b_outputs/*.json

Defaults match Corpius 2 (14B) per ``config/conceptial_14b.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from brainstormlm.conceptial.anti_mode import (  # noqa: E402
    extract_summary,
    format_anti_mode,
)
from brainstormlm.conceptial.engine.config import SteeringConfig  # noqa: E402
from brainstormlm.conceptial.engine.pipeline import (  # noqa: E402
    SteeringEngine,
    load_steering_checkpoint,
)
from brainstormlm.conceptial.engine.steering_presets import (  # noqa: E402
    apply_v10_parametric_preset,
)
from brainstormlm.conceptial.engine.vector_purify import purify_vector  # noqa: E402
from brainstormlm.conceptial.idea_store import (  # noqa: E402
    SOURCE_GENERATED,
    IdeaMemoryStore,
)


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _iter_creative_vector(
    creative_vector_path: Path,
    prior_first_token_hiddens: list[torch.Tensor],
) -> torch.Tensor:
    """Recompute creative vector orthogonal to the mean prior first-token residual.

    Returns the original purified vector when prior_first_token_hiddens is empty.
    """
    bundle = load_steering_checkpoint(creative_vector_path, map_location="cpu")
    v = bundle["vector"].float()
    if not prior_first_token_hiddens:
        return v
    mean_h = torch.stack(
        [h.float().reshape(-1) for h in prior_first_token_hiddens], dim=0
    ).mean(0)
    return purify_vector(v, [mean_h])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_creative_briefs.json"
    )
    parser.add_argument(
        "--corpius-config",
        type=Path,
        default=ROOT / "config" / "conceptial_14b.json",
        help="Corpius 2 (14B) spec — model id, vector paths, layers, schedule.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--base-temp", type=float, default=0.8)
    parser.add_argument("--temp-jitter", type=float, default=0.05)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "benchmark_v2" / "conceptial_14b_outputs",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data" / "benchmark_14b_memory.db",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Process only the first N prompts (for fast smoke tests).",
    )
    parser.add_argument(
        "--no-iterative-anti-mode",
        action="store_true",
        help="Disable iterative L19 first-token residual orthogonalization.",
    )
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable CAA steering entirely (anti-mode + memory only baseline).",
    )
    args = parser.parse_args()

    # ── Load configs ─────────────────────────────────────────────────────────
    cfg = json.loads(args.config.read_text())
    corpius = json.loads(args.corpius_config.read_text())
    prompts = cfg["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    n = args.n or cfg["n_outputs_per_prompt"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = corpius["base_model"]
    creative_vector_path = ROOT / corpius["steering_vectors"]["creative"]
    technical_vector_path = ROOT / corpius["steering_vectors"]["technical"]
    creative_layer = int(corpius["layers"]["creative_layer"])
    technical_layer = int(corpius["layers"]["technical_layer"])
    schedule = corpius["schedule"]

    steering_enabled = not args.no_steering
    if steering_enabled:
        for p in (creative_vector_path, technical_vector_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing steering vector at {p}. Run "
                    f"`uv run python scripts/extract_14b_vectors.py` first."
                )

    iter_anti_mode = steering_enabled and not args.no_iterative_anti_mode

    print("\n=== RunPod 14B batch (full stack) ===")
    print(f"Model:                 {model_id}")
    print(f"Prompts:               {len(prompts)}")
    print(f"N per arm:             {n}")
    print(f"Steering:              {'ON' if steering_enabled else 'OFF'}")
    if steering_enabled:
        print(f"  Creative vector:     {creative_vector_path.relative_to(ROOT)} (L{creative_layer})")
        print(f"  Technical vector:    {technical_vector_path.relative_to(ROOT)} (L{technical_layer})")
        print(f"  v17 schedule:        creative α0={schedule['creative_start_alpha']} exp_zero={schedule['exp_zero_by_step']} · technical α0={schedule['technical_start_alpha']} sigmoid_mid={schedule['sigmoid_mid_step']}")
    print(f"Iterative anti-mode:   {'ON' if iter_anti_mode else 'OFF'} (orthogonalize creative vs prior L{creative_layer} first-token residuals)")
    print(f"Out dir:               {args.out_dir.relative_to(ROOT)}")

    # ── Load model via SteeringEngine ────────────────────────────────────────
    device = _resolve_device()
    print(f"\nLoading {model_id} on {device} … (~120s on A100)")
    cfg_engine = SteeringConfig(model_id=model_id)
    cfg_engine.preferred_device = device
    engine = SteeringEngine(cfg_engine)
    engine.load()
    print("Model loaded.")

    # ── Install steering preset ─────────────────────────────────────────────
    if steering_enabled:
        apply_v10_parametric_preset(
            engine,
            creative_vector_path.parent,
            "v17",
            creative_vector_path=creative_vector_path,
            technical_vector_path=technical_vector_path,
            creative_layer=creative_layer,
            technical_layer=technical_layer,
            steering_math=str(schedule.get("steering_math", "additive")),
            creative_start_alpha=float(schedule["creative_start_alpha"]),
            exp_zero_by_step=int(schedule["exp_zero_by_step"]),
            technical_start_alpha=float(schedule["technical_start_alpha"]),
            sigmoid_mid_step=float(schedule["sigmoid_mid_step"]),
            sigmoid_steepness=float(schedule["sigmoid_steepness"]),
        )
        print("v17 preset applied.")

    store = IdeaMemoryStore(db_path=args.db_path)
    HIDDEN_DIM = 64
    total_outputs = 0
    t_start = time.time()

    # ── Main loop: per prompt × per sample ──────────────────────────────────
    for p in prompts:
        pid = p["id"]
        brief = p["brief"]
        project_id = f"bench14b-{pid}"
        store.delete_project(project_id)

        out_path = args.out_dir / f"{pid}.conceptial-14b.json"
        prior_summaries: list[str] = []
        prior_first_token_hiddens: list[torch.Tensor] = []
        outputs: list[dict] = []
        t_prompt = time.time()

        for k in range(n):
            # 1. Iterative anti-mode: re-purify creative vector against prior residuals.
            if iter_anti_mode and prior_first_token_hiddens:
                v_creative_iter = _iter_creative_vector(
                    creative_vector_path, prior_first_token_hiddens
                )
                p_param = next(engine.model.parameters())
                v_creative_iter = v_creative_iter.to(device=p_param.device, dtype=p_param.dtype)
                current_vectors = dict(engine.steering_vectors or {})
                current_vectors[creative_layer] = v_creative_iter
                engine.set_steering_vectors(current_vectors)

            # 2. Build the anti-mode prompt (exclusion list of prior summaries).
            full_prompt = format_anti_mode(
                user_text=brief,
                prior_summaries=prior_summaries,
                exemplars=None,
                output_mode="generic",
            )

            # 3. Sample with min-p + temperature jitter; capture L19 first-token residual.
            temp = float(
                args.base_temp + (((k * 0.61803) % 1.0 - 0.5) * args.temp_jitter * 2)
            )
            first_token_buf: list[torch.Tensor] = []
            capture_arg = (
                (creative_layer, first_token_buf) if iter_anti_mode else None
            )
            text = engine.complete(
                full_prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=temp,
                do_sample=True,
                do_steering=steering_enabled,
                use_ghost_rejection=False,
                capture_first_decode_residual=capture_arg,
            )
            text = text.strip()
            if iter_anti_mode and first_token_buf:
                prior_first_token_hiddens.append(first_token_buf[0].detach().to("cpu").float())

            # 4. Persist summary + memory.
            summary = extract_summary(text) or text.split("\n")[0][:80]
            prior_summaries.append(summary)
            idea_id = uuid.uuid4().hex
            store.add(
                project_id=project_id,
                idea_id=idea_id,
                hidden=torch.zeros(HIDDEN_DIM, dtype=torch.float32),
                text=text,
                source=SOURCE_GENERATED,
            )
            outputs.append(
                {
                    "model": "conceptial-14b",
                    "k": k,
                    "text": text,
                    "summary": summary,
                    "trajectory_id": idea_id,
                    "temperature": temp,
                    "iterative_anti_mode_applied": iter_anti_mode and k > 0,
                }
            )
            total_outputs += 1

        out_path.write_text(
            json.dumps(
                {
                    "prompt_id": pid,
                    "arm": "conceptial-14b",
                    "n": len(outputs),
                    "outputs": outputs,
                    "stack": {
                        "model_id": model_id,
                        "steering": steering_enabled,
                        "iterative_anti_mode": iter_anti_mode,
                        "creative_layer": creative_layer,
                        "technical_layer": technical_layer,
                    },
                },
                indent=2,
            )
        )
        elapsed = time.time() - t_prompt
        print(
            f"[{pid:35s}] wrote {len(outputs)} outputs in {elapsed:.1f}s "
            f"→ {out_path.name}"
        )

    store.close()
    total_elapsed = time.time() - t_start
    print(f"\nDone. {total_outputs} total outputs in {total_elapsed:.1f}s.")
    print("\nNext step (locally):")
    print(f"  scp -r root@<pod-ip>:/workspace/BrainstormLM/{args.out_dir.relative_to(ROOT)}/* \\")
    print(f"      ~/Desktop/Projects/BrainstormLM/{args.out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
