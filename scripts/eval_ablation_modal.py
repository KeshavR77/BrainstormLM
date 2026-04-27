"""Modal-hosted variant of eval_ablation.py.

Local Apple Silicon MPS cannot run Qwen-7B inference (4GB tensor limit), so
the ablation harness needs to run somewhere with a real GPU. This script
mirrors the architecture of `scripts/train_dpo.py`:

1. Local CLI reads HELD_OUT + (optional) lessons.jsonl, packages them, and
   calls a Modal function.
2. Modal function loads HFGenerator from the adapter on the existing
   `brainstormlm-adapters` Volume, runs `run_ablation` against HELD_OUT,
   and returns the rows + summary as a JSON-friendly dict.
3. Local CLI persists the rows + summary to `data/eval/ablation_<run_id>*`
   and prints the summary table.

API keys (OPENAI/TOGETHER) flow into the container via
`modal.Secret.from_dotenv()`.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import modal
import typer

# ---------------------------------------------------------------------------
# Modal app + image + volume (shared with train_dpo.py — same volume name)
# ---------------------------------------------------------------------------

ADAPTERS_REMOTE_DIR = "/root/adapters"
ADAPTER_VOLUME_NAME = "brainstormlm-adapters"

app = modal.App("brainstormlm-eval-ablation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.45",
        "torch",
        "peft>=0.13",
        "accelerate>=1.0",
        "datasets",
        "pydantic>=2.0",
        "pydantic-settings",
        "openai",
        "together",
        "python-dotenv",
        "numpy",
        "tqdm",
    )
    .add_local_dir("src/brainstormlm", remote_path="/root/brainstormlm")
)

adapter_volume = modal.Volume.from_name(ADAPTER_VOLUME_NAME, create_if_missing=False)


# ---------------------------------------------------------------------------
# Modal function: runs the ablation
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-40GB",  # plenty for Qwen-7B + LoRA inference; cheaper than 80GB
    timeout=60 * 60,  # 60 min ceiling; worst-case ~$2.78
    image=image,
    volumes={ADAPTERS_REMOTE_DIR: adapter_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def evaluate(
    adapter_run_id: str,
    held_out: list[list[str]],  # JSON-serializable list of [domain, prompt]
    *,
    eval_run_id: str,
    lessons_jsonl: str = "",
    k_success: int = 3,
    k_failure: int = 2,
    temperature: float = 1.0,
    max_tokens: int = 400,
    bootstrap_iterations: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the 4-arm (or 2-arm if no lessons) ablation entirely on Modal."""
    import time as _time

    from brainstormlm.eval.harness import (
        ARMS,
        AblationRow,
        format_summary_table,
        row_to_dict,
        summarize,
    )
    from brainstormlm.feedback import evaluator
    from brainstormlm.memory import (
        LessonStore,
        augmented_system,
        lessons as lessons_mod,
        retrieve,
    )
    from brainstormlm.model import HFGenerator, TogetherGenerator

    adapter_path = Path(ADAPTERS_REMOTE_DIR) / adapter_run_id
    if not adapter_path.exists():
        raise RuntimeError(
            f"Adapter dir not found in volume: {adapter_path}. "
            f"Available: {list(Path(ADAPTERS_REMOTE_DIR).iterdir())}"
        )

    base_generator = TogetherGenerator()
    trained_generator = HFGenerator(adapter_path=adapter_path)

    store: LessonStore | None = None
    if lessons_jsonl.strip():
        lessons_tmp = Path("/tmp/lessons_for_eval.jsonl")
        lessons_tmp.write_text(lessons_jsonl, encoding="utf-8")
        store = LessonStore(lessons_tmp)
        t0 = _time.time()
        store.load()
        print(
            f"[eval] LessonStore loaded {len(store.lessons)} lessons in {_time.time()-t0:.1f}s"
        )
        if not store.lessons:
            raise RuntimeError("LessonStore loaded but contains no lessons.")
        _ = lessons_mod.EXTRACTION_VERSION

    held_out_tuples = [(d, p) for d, p in held_out]
    arms_to_run = [a for a in ARMS if not (a.endswith("+mem") and store is None)]
    total_calls = len(held_out_tuples) * len(arms_to_run)
    print(f"[eval] Running {len(arms_to_run)} arms × {len(held_out_tuples)} prompts = {total_calls} calls")

    # Inline ablation loop with progress prints (replaces run_ablation so we can
    # see live progress; behavior is otherwise identical).
    rows: list[AblationRow] = []
    adapter_str = str(adapter_path)
    t_loop = _time.time()
    for arm in arms_to_run:
        gen = trained_generator if arm.startswith("trained") else base_generator
        for i, (domain, prompt) in enumerate(held_out_tuples):
            if arm.endswith("+mem") and store is not None:
                lessons = retrieve(store, prompt, k_success=k_success, k_failure=k_failure)
                system = augmented_system(None, lessons)
                mem_ids = [lesson.id for lesson in lessons]
            else:
                system = None
                mem_ids = []
            response, cached = gen.generate_with_meta(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens
            )
            result, _ = evaluator.score(prompt, response)
            rows.append(
                AblationRow(
                    arm=arm,
                    domain=domain,
                    prompt=prompt,
                    response=response,
                    creativity=result.creativity,
                    logic=result.logic,
                    total=result.creativity + result.logic,
                    memory_context_ids=mem_ids,
                    adapter_path=adapter_str if arm.startswith("trained") else None,
                    generator_cached=cached,
                )
            )
            done = len(rows)
            if done == 1 or done % 5 == 0 or done == total_calls:
                elapsed = _time.time() - t_loop
                rate = done / max(elapsed, 1e-6)
                eta = (total_calls - done) / max(rate, 1e-6)
                print(
                    f"[eval] {done}/{total_calls} ({arm} #{i+1}) "
                    f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
    summaries = summarize(rows, n_iterations=bootstrap_iterations, seed=seed)

    payload = {
        "rows": [row_to_dict(r) for r in rows],
        "summaries": [
            {
                "arm": s.arm,
                "n": s.n,
                "mean_creativity": s.mean_creativity,
                "mean_logic": s.mean_logic,
                "mean_total": s.mean_total,
                "mean_delta_vs_base": s.mean_delta_vs_base,
                "p_value": s.p_value,
            }
            for s in summaries
        ],
        "table": format_summary_table(summaries),
    }

    # Persist to volume so the result survives a local disconnect. The local
    # CLI may still be alive to receive the return value below; if not, recover
    # via `modal volume get brainstormlm-adapters _eval_results data/eval/`.
    results_dir = Path(ADAPTERS_REMOTE_DIR) / "_eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{eval_run_id}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    adapter_volume.commit()
    print(f"[eval] persisted result to volume: _eval_results/{eval_run_id}.json", flush=True)

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

cli = typer.Typer(add_completion=False, no_args_is_help=False)


@cli.command()
def main(
    adapter: Annotated[
        Path,
        typer.Option(help="Local adapter directory; the directory name is the run_id stored in the Modal Volume."),
    ],
    out_dir: Annotated[
        Path | None, typer.Option(help="Eval output dir (default: data/eval).")
    ] = None,
    skip_memory: Annotated[
        bool, typer.Option("--skip-memory", help="Run only base + trained arms (no LessonStore needed).")
    ] = False,
    k_success: Annotated[int, typer.Option(help="Success lessons per memory-arm prompt.")] = 3,
    k_failure: Annotated[int, typer.Option(help="Failure lessons per memory-arm prompt.")] = 2,
    temperature: Annotated[float, typer.Option(help="Generator sampling temperature.")] = 1.0,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed.")] = 42,
    bootstrap_iterations: Annotated[int, typer.Option(help="Paired-bootstrap resamples.")] = 10000,
) -> None:
    """Score every HELD_OUT prompt under each arm via Modal; print + persist."""
    from brainstormlm.config import get_settings
    from brainstormlm.eval.fixtures import HELD_OUT

    settings = get_settings()
    adapter_run_id = adapter.name
    rid = uuid.uuid4().hex[:12]
    out = out_dir or (settings.data_dir / "eval")
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"ablation_{rid}.jsonl"
    summary_path = out / f"ablation_{rid}_summary.json"

    lessons_jsonl = ""
    if not skip_memory:
        if not settings.lessons_path.exists():
            typer.echo(
                f"Memory store missing ({settings.lessons_path}). "
                "Run scripts/rebuild_memory.py first or pass --skip-memory.",
                err=True,
            )
            raise typer.Exit(code=1)
        lessons_jsonl = settings.lessons_path.read_text(encoding="utf-8")

    typer.echo(f"Eval run id: {rid}")
    typer.echo(f"Adapter run id: {adapter_run_id}")
    typer.echo(f"Held-out prompts: {len(HELD_OUT)}")
    typer.echo(
        f"Arms: {'base, trained' if skip_memory else 'base, base+mem, trained, trained+mem'}"
    )
    typer.echo("Launching Modal ablation job (detached: survives local disconnects)...")

    held_out_payload = [list(pair) for pair in HELD_OUT]

    with app.run(detach=True):
        result = evaluate.remote(
            adapter_run_id=adapter_run_id,
            held_out=held_out_payload,
            eval_run_id=rid,
            lessons_jsonl=lessons_jsonl,
            k_success=k_success,
            k_failure=k_failure,
            temperature=temperature,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )

    with rows_path.open("w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": rid,
                "adapter_run_id": adapter_run_id,
                "n_prompts": len(HELD_OUT),
                "summaries": result["summaries"],
            },
            f,
            indent=2,
        )

    typer.echo("")
    typer.echo(result["table"])
    typer.echo("")
    typer.echo(f"Rows: {rows_path}")
    typer.echo(f"Summary: {summary_path}")


if __name__ == "__main__":
    sys.exit(cli() or 0)
