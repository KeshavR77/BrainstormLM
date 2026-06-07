"""Re-capture per-position activations for an existing scored pool — no regeneration.

When a checkpoint's ``activations/<position>.npz`` is lost (e.g. a partial RunPod volume
migration) but ``scored_pool.jsonl`` survived, this re-derives the missing position(s) with a
single forward pass per saved (prompt, response) — far cheaper than regenerating the responses.

The framed system prompt is reconstructed from the prompt set (creative vs neutral), so
``prompt_len`` — and therefore the ``mean_response`` span — matches the original capture. Output
is row-aligned to ``scored_pool.jsonl``, so the restored ``.npz`` drops straight back into the
checkpoint and ``load_checkpoint_pool`` / resume / recut work unchanged.

Usage (on a GPU pod with the model)::

    python creativity_steering/scripts/recapture_activations.py \\
        --run artifacts/qwen14b_run_v3 \\
        --config creativity_steering/configs/runpod_qwen14b_v3.json \\
        --positions mean_response
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from creativity_steering.derivation.pipeline import _atomic_savez


def recapture(run: Path, backend, positions, system_for: dict[str, str | None]) -> dict[str, tuple]:
    """Forward each saved (prompt, response) and write ``activations/<pos>.npz`` per position.

    ``system_for`` maps a framing (``creative``/``neutral``/``unframed``) to the system prompt
    used at generation time. Returns ``{position: shape}`` of what was written.
    """
    run = Path(run)
    rows = [json.loads(line) for line in (run / "scored_pool.jsonl").read_text().splitlines() if line]
    if not rows:
        raise SystemExit(f"empty/no scored_pool.jsonl at {run}")
    stacks: dict[str, list] = {p: [] for p in positions}
    for i, row in enumerate(rows):
        system = system_for.get(row["framing"])
        acts = backend.capture_activations(
            row["prompt"], row["response"], system=system, positions=list(positions)
        )
        for p in positions:
            stacks[p].append(acts[p])
        if (i + 1) % 50 == 0:
            print(f"  recaptured {i + 1}/{len(rows)}", flush=True)
    acts_dir = run / "activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for p in positions:
        arr = np.stack(stacks[p]).astype(np.float32)
        _atomic_savez(acts_dir / f"{p}.npz", acts=arr)
        written[p] = arr.shape
        print(f"[recapture] wrote {p}.npz {arr.shape}", flush=True)
    return written


def main() -> None:
    from creativity_steering.config import load_config
    from creativity_steering.derivation.prompts import default_prompt_set, load_prompt_set
    from creativity_steering.models.factory import build_backend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run dir with scored_pool.jsonl")
    ap.add_argument("--config", required=True, type=Path, help="ExperimentConfig JSON (backend + topics)")
    ap.add_argument("--positions", nargs="+", required=True, help="positions to re-capture, e.g. mean_response")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ps = load_prompt_set(cfg.topics_file) if cfg.topics_file else default_prompt_set()
    system_for = {"creative": ps.creative_system, "neutral": ps.neutral_system, "unframed": None}
    backend = build_backend(cfg)
    print(f"[recapture] backend={cfg.backend} model={backend.model_id} positions={args.positions}",
          flush=True)
    recapture(args.run, backend, args.positions, system_for)
    print("[recapture] done", flush=True)


if __name__ == "__main__":
    main()
