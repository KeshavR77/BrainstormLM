#!/usr/bin/env python3
"""RunPod-side batch generator for the 14B head-to-head benchmark.

Reads the prompt suite from ``config/benchmark_creative_briefs.json``, runs
N outputs per prompt against ``Qwen/Qwen2.5-14B-Instruct`` with the locked
Conceptial stack (cognitive anti-mode + per-prompt persistent memory in
SQLite + brief-aware output mode + min-p sampling), and writes one JSON
file per prompt in the shape the benchmark aggregator expects.

Usage on a RunPod A100 80GB pod::

    # 1. Clone + setup (see docs/cloud_runpod_pivot.md)
    cd /workspace/BrainstormLM
    uv sync
    export HF_TOKEN=hf_...

    # 2. Run the batch (10 prompts × N=8 ≈ 8-12 minutes on A100)
    uv run python scripts/runpod_batch_14b.py --n 8

    # 3. scp the results back to local
    #    rsync -av artifacts/benchmark_v2/conceptial_14b_outputs/ \\
    #      <local>:~/Desktop/Projects/BrainstormLM/artifacts/benchmark_v2/conceptial_14b_outputs/

    # 4. Locally, the harness picks them up and computes metrics:
    uv run python scripts/benchmark_creative_briefs.py \\
        --arms gpt-4o,gpt-4o-mini,conceptial-14b --n 8 \\
        --out artifacts/benchmark_v2/run_14b_$(date +%Y%m%d)

Configuration:
- Model: ``Qwen/Qwen2.5-14B-Instruct``  (override with --model-id)
- Steering: NOT applied (3B-extracted vectors don't transfer)
- Anti-mode: applied via brainstormlm.conceptial.anti_mode.format_anti_mode
- Memory: SQLite project, reset before each prompt's batch
- Sampling: temperature 0.8 with ±0.05 jitter; min_p 0.01
- Output mode: "generic" (mode-agnostic to the brief category)
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
from brainstormlm.conceptial.idea_store import (  # noqa: E402
    SOURCE_GENERATED,
    IdeaMemoryStore,
)


def _apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-P truncation: zero probs below ``min_p * max(probs)``, renormalize."""
    if min_p <= 0.0 or probs.numel() == 0:
        return probs
    pmax = float(probs.max().item())
    if pmax <= 0.0:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    thr = min_p * pmax
    out = probs.clone()
    out[out < thr] = 0.0
    s = float(out.sum().item())
    if s <= 1e-12:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    return out / out.sum()


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _generate_one(
    *,
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
) -> str:
    """One sampling pass with min-p + temperature."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    new_tokens: list[int] = []
    past = None
    cur = inputs["input_ids"]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :].squeeze(0) / max(temperature, 1e-3)
        probs = torch.softmax(logits, dim=-1)
        probs = _apply_min_p(probs, min_p)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        if next_id == tokenizer.eos_token_id:
            break
        new_tokens.append(next_id)
        cur = torch.tensor([[next_id]], device=model.device, dtype=torch.long)
        past = out.past_key_values
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_creative_briefs.json"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--base-temp", type=float, default=0.8)
    parser.add_argument("--temp-jitter", type=float, default=0.05)
    parser.add_argument("--min-p", type=float, default=0.01)
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
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    prompts = cfg["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    n = args.n or cfg["n_outputs_per_prompt"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== RunPod 14B batch ===")
    print(f"Model:     {args.model_id}")
    print(f"Prompts:   {len(prompts)}")
    print(f"N per arm: {n}")
    print(f"Out dir:   {args.out_dir}")

    # Load model
    print(f"\nLoading {args.model_id} … (this takes 60-120s on A100)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = _resolve_device()
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    store = IdeaMemoryStore(db_path=args.db_path)
    HIDDEN_DIM = 64
    total_outputs = 0
    t_start = time.time()

    for p in prompts:
        pid = p["id"]
        brief = p["brief"]
        project_id = f"bench14b-{pid}"
        store.delete_project(project_id)

        out_path = args.out_dir / f"{pid}.conceptial-14b.json"
        prior_summaries: list[str] = []
        outputs: list[dict] = []
        t_prompt = time.time()
        for k in range(n):
            full_prompt = format_anti_mode(
                user_text=brief,
                prior_summaries=prior_summaries,
                exemplars=None,
                output_mode="generic",
            )
            temp = float(args.base_temp + (((k * 0.61803) % 1.0 - 0.5) * args.temp_jitter * 2))
            text = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=full_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                min_p=args.min_p,
            )
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
                }
            )
            total_outputs += 1

        out_path.write_text(
            json.dumps(
                {"prompt_id": pid, "arm": "conceptial-14b", "n": len(outputs), "outputs": outputs},
                indent=2,
            )
        )
        elapsed = time.time() - t_prompt
        print(f"[{pid:35s}] wrote {len(outputs)} outputs in {elapsed:.1f}s → {out_path.name}")

    store.close()
    total_elapsed = time.time() - t_start
    print(f"\nDone. {total_outputs} total outputs in {total_elapsed:.1f}s.")
    print("\nNext step (locally):")
    print(f"  scp -r root@<pod-ip>:/workspace/BrainstormLM/{args.out_dir.relative_to(ROOT)}/* \\")
    print(f"      ~/Desktop/Projects/BrainstormLM/{args.out_dir.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
