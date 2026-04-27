"""Modal entrypoint for offline DPO training of a LoRA adapter on Qwen2.5-7B.

Pre-flight cost estimate is informational + a guardrail (`--max-modal-usd`).
Actual cost is enforced by the function timeout and Modal's per-second billing.

Usage:

    uv run python scripts/train_dpo.py --pairs-path data/preferences.jsonl \\
        --max-pairs 2000 --max-modal-usd 10.0

    uv run python scripts/train_dpo.py --local --dry-run

Architecture:
1. The local CLI reads the pairs JSONL into memory and passes it to the
   Modal function as a string argument (pairs are small — ~1MB at 2k pairs).
2. The Modal function trains and saves the adapter to a Modal Volume.
3. The CLI then downloads the adapter directory from the Volume to
   `data/adapters/<run_id>/` via `modal volume get`.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import modal
import typer

from brainstormlm.config import get_settings
from brainstormlm.feedback import pricing
from brainstormlm.training.dpo import (
    DPOConfig,
    build_dpo_trainer,
    dataset_from_pairs,
)
from brainstormlm.training.preferences import PreferencePair, read_all

# ---------------------------------------------------------------------------
# Modal app + image + adapter volume
# ---------------------------------------------------------------------------

ADAPTERS_REMOTE_DIR = "/root/adapters"
ADAPTER_VOLUME_NAME = "brainstormlm-adapters"

app = modal.App("brainstormlm-dpo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45",
        "torch",
        "peft>=0.13",
        "trl>=1.0",
        "accelerate>=1.0",
        "datasets",
        "pydantic>=2.0",
        "pydantic-settings",
        # Transitive imports: training/preferences.py → feedback/evaluator.py
        # → clients/openai.py and clients/together.py. Even though we never
        # call those clients during a training run, Python still imports them
        # at module load. Cheaper to ship the packages than to refactor.
        "openai",
        "together",
        "python-dotenv",
    )
    .add_local_dir("src/brainstormlm", remote_path="/root/brainstormlm")
)

adapter_volume = modal.Volume.from_name(ADAPTER_VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Core training routine (callable both locally and inside Modal)
# ---------------------------------------------------------------------------


def _split_train_val(
    pairs: list[PreferencePair],
    *,
    seed: int,
    val_fraction: float = 0.05,
) -> tuple[list[PreferencePair], list[PreferencePair]]:
    """Deterministic 95/5 train/val split."""
    if not pairs:
        return [], []
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) >= 20 else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def _train_metrics_rows(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project DPO-relevant fields out of `trainer.state.log_history`."""
    keys = {"step", "epoch", "loss", "rewards/chosen", "rewards/rejected", "rewards/margins"}
    rows = []
    for entry in log_history:
        if not any(k in entry for k in ("loss", "rewards/margins")):
            continue
        rows.append({k: entry.get(k) for k in keys})
    return rows


def _run_training(
    pairs_source: Path | str,
    *,
    run_id: str,
    max_pairs: int,
    seed: int,
    adapter_out_dir: Path,
    base_model_id: str,
    hf_token: str | None,
) -> dict[str, Any]:
    """Runs the actual DPO training. Heavy imports inside.

    ``pairs_source`` is either a local Path or raw JSONL text (used inside the
    Modal container to avoid a separate file-shipping step).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(pairs_source, Path):
        pairs = list(read_all(pairs_source))
    else:
        pairs = [
            PreferencePair.model_validate_json(line)
            for line in pairs_source.splitlines()
            if line.strip()
        ]
    if not pairs:
        raise RuntimeError(f"No preference pairs found from {pairs_source!r}")
    pairs = pairs[:max_pairs]
    train_pairs, val_pairs = _split_train_val(pairs, seed=seed)

    cfg = DPOConfig(
        base_model_id=base_model_id,
        adapter_out_dir=adapter_out_dir,
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset_from_pairs(train_pairs, tokenizer)
    # val_dataset is built but DPOTrainer's eval pass is optional; we leave it
    # for a future iteration. Track size for the manifest.
    _ = dataset_from_pairs(val_pairs, tokenizer) if val_pairs else None

    import torch  # heavy, deferred

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        token=hf_token,
    )
    # TRL ≥ 1.x: pass `ref_model=None` and let the trainer handle the reference
    # (with a PEFT adapter on the policy model, the base weights serve as ref).
    ref_model = None

    trainer = build_dpo_trainer(model, ref_model, tokenizer, train_dataset, cfg)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    adapter_out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_out_dir))
    tokenizer.save_pretrained(str(adapter_out_dir))

    # train_metrics.csv
    metrics_rows = _train_metrics_rows(trainer.state.log_history)
    metrics_path = adapter_out_dir / "train_metrics.csv"
    if metrics_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    manifest = {
        "run_id": run_id,
        "base_model_id": cfg.base_model_id,
        "adapter_out_dir": str(adapter_out_dir),
        "config": cfg.model_dump(mode="json"),
        "n_pairs_total": len(pairs),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "training_seconds": elapsed,
        "estimated_modal_cost_usd": pricing.estimate_modal_cost(
            elapsed / 3600.0, gpu="A100-80GB"
        ),
    }
    manifest_path = adapter_out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@app.function(
    gpu="A100-80GB",
    timeout=2 * 60 * 60,
    image=image,
    volumes={ADAPTERS_REMOTE_DIR: adapter_volume},
)
def train(
    pairs_jsonl: str,
    run_id: str = "",
    max_pairs: int = 2000,
    seed: int = 42,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict[str, Any]:
    """Modal-hosted DPO training. Returns the manifest dict.

    Adapter artifacts land in the ``brainstormlm-adapters`` Modal Volume at
    ``/<run_id>/`` and are committed before the function returns. The local
    CLI then pulls them down via ``modal volume get``.
    """
    if not run_id:
        run_id = uuid.uuid4().hex[:16]
    adapter_out_dir = Path(ADAPTERS_REMOTE_DIR) / run_id
    manifest = _run_training(
        pairs_jsonl,
        run_id=run_id,
        max_pairs=max_pairs,
        seed=seed,
        adapter_out_dir=adapter_out_dir,
        base_model_id=base_model_id,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    adapter_volume.commit()
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

cli = typer.Typer(add_completion=False, no_args_is_help=False)


@cli.command()
def main(
    pairs_path: Annotated[Path, typer.Option(help="Path to PreferencePair JSONL.")] = Path(
        "data/preferences.jsonl"
    ),
    run_id: Annotated[
        str | None, typer.Option(help="Run id; auto-generated if omitted.")
    ] = None,
    max_pairs: Annotated[int, typer.Option(help="Cap on number of pairs to train on.")] = 2000,
    max_modal_usd: Annotated[
        float, typer.Option(help="Abort pre-flight if Modal estimate exceeds this.")
    ] = 25.0,
    local: Annotated[
        bool, typer.Option("--local", help="Run training in-process (no Modal).")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Stop after pre-flight cost + pair check; do not train."),
    ] = False,
    seed: Annotated[int, typer.Option(help="Seed for train/val split.")] = 42,
) -> None:
    settings = get_settings()
    rid = run_id or uuid.uuid4().hex[:16]

    estimate = pricing.estimate_modal_cost(2.0, gpu="A100-80GB")
    typer.echo(f"Run id: {rid}")
    typer.echo(f"Pre-flight Modal cost estimate (2h on A100-80GB): ${estimate:.2f}")
    if estimate > max_modal_usd:
        typer.echo(
            f"Aborting: estimate ${estimate:.2f} exceeds --max-modal-usd ${max_modal_usd:.2f}.",
            err=True,
        )
        raise typer.Exit(code=2)

    if not pairs_path.exists():
        typer.echo(f"Pairs file not found: {pairs_path}", err=True)
        raise typer.Exit(code=2)

    n_pairs_available = sum(1 for _ in read_all(pairs_path))
    typer.echo(f"Available pairs in {pairs_path}: {n_pairs_available} (will use up to {max_pairs})")
    if n_pairs_available == 0:
        typer.echo("No pairs to train on; aborting.", err=True)
        raise typer.Exit(code=2)

    if dry_run:
        typer.echo("Dry run: pre-flight checks passed; no training launched.")
        return

    adapter_out_dir = settings.adapters_dir / rid

    if local:
        typer.echo("Running training locally (no Modal)...")
        manifest = _run_training(
            pairs_path,
            run_id=rid,
            max_pairs=max_pairs,
            seed=seed,
            adapter_out_dir=adapter_out_dir,
            base_model_id=settings.hf_base_model,
            hf_token=(
                settings.hf_token.get_secret_value() if settings.hf_token else None
            ),
        )
    else:
        typer.echo("Launching Modal training job (detached: survives local disconnects)...")
        pairs_jsonl_text = pairs_path.read_text(encoding="utf-8")
        # detach=True: app + function keep running on Modal even if the local
        # client loses heartbeat (WiFi blip, display sleep, etc.). Without
        # this, a single dropped heartbeat kills training mid-run. The local
        # call still blocks for the return value when the connection is
        # healthy; if the client dies, you can pull the adapter from the
        # `brainstormlm-adapters` Volume after the job finishes.
        with app.run(detach=True):
            manifest = train.remote(
                pairs_jsonl=pairs_jsonl_text,
                run_id=rid,
                max_pairs=max_pairs,
                seed=seed,
                base_model_id=settings.hf_base_model,
            )

        # Pull adapter artifacts from the Modal Volume to the local disk.
        adapter_out_dir.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Downloading adapter from Modal Volume → {adapter_out_dir} ...")
        result = subprocess.run(
            [
                "modal", "volume", "get",
                ADAPTER_VOLUME_NAME,
                rid,
                str(adapter_out_dir.parent),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(f"modal volume get failed: {result.stderr}", err=True)
            typer.echo(
                f"Adapter is still on the Modal Volume. Pull it manually:\n"
                f"  modal volume get {ADAPTER_VOLUME_NAME} {rid} {adapter_out_dir.parent}",
                err=True,
            )
        manifest["adapter_out_dir"] = str(adapter_out_dir)

    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    typer.echo(f"Adapter saved to: {manifest['adapter_out_dir']}")


if __name__ == "__main__":
    sys.exit(cli() or 0)
