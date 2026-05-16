#!/usr/bin/env python3
"""Cloud 14B smoke — raw Qwen2.5-14B-Instruct + prompt-level anti-mode + persistence.

The cloud variant of ``smoke_cognitive_anti_mode.py``. Critical differences from the
local-3B smoke:

- **No v17 steering**: the L14+L18 purified vectors were extracted from Qwen2.5-3B
  (hidden_size=2048). Qwen2.5-14B has hidden_size=5120; the vectors are dimensionally
  incompatible. Re-extracting Harmonic vectors on 14B is a separate research task.
- **No SteeringEngine**: we load the model directly via ``transformers`` since steering
  isn't applied. KV-cache + min_p logic is plain HF generate.
- **Same anti-mode / persistence / brief-aware output mode**: every prompt-level
  innovation carries over unchanged from the local stack.

The expectation (per ``docs/cloud_runpod_pivot.md``): comparable Vendi (~4.0) with
substantially higher individual concept quality (composite 4.3+, less factual
hallucination, more specific imagery).

Run on the RunPod pod after the setup steps in ``docs/cloud_runpod_pivot.md``::

    python3 scripts/smoke_14b_antimode.py \\
        --prompt "Give me a creative logo for an engineering club at Stanford." \\
        --n 5 --base-temp 0.8 --temp-jitter 0.05 --min-p 0.01 \\
        --project-id "cloud-14b-logo-S1" --reset-project \\
        --out-md artifacts/conceptial_14b/CLOUD_14B_S1.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Reuse the same prompt formatter + persistence layer the local stack uses. No model code.
from brainstormlm.conceptial import (
    DEFAULT_PROMPT,
    OUTPUT_MODE_INSTRUCTIONS,
    IdeaMemoryStore,
    SOURCE_GENERATED,
    extract_summary,
    format_anti_mode,
)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-P truncation: zero out probs below ``min_p × max(probs)``, renormalize."""
    if min_p <= 0.0 or probs.numel() == 0:
        return probs
    pmax = float(probs.max().item())
    if pmax <= 0.0:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    thr = float(min_p) * pmax
    out = probs.clone()
    out[out < thr] = 0.0
    s = float(out.sum().item())
    if s <= 1e-12:
        out = torch.zeros_like(probs)
        out[int(probs.argmax().item())] = 1.0
        return out
    return out / out.sum()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float, min_p: float) -> str:
    """Plain HF generate with min_p sampling. No steering, no ghost decode."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    eos_id = tokenizer.eos_token_id
    out_ids: list[int] = []

    pre = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = pre.past_key_values
    logits = pre.logits[:, -1, :]

    for _ in range(int(max_new_tokens)):
        if temperature > 0:
            scaled = logits[0].float() / float(temperature)
            probs = torch.softmax(scaled, dim=-1)
            if min_p > 0:
                probs = _apply_min_p(probs, float(min_p))
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(logits[0].argmax().item())
        if eos_id is not None and next_id == int(eos_id):
            break
        out_ids.append(next_id)
        step = torch.tensor([[next_id]], device=model.device, dtype=torch.long)
        m = model(input_ids=step, past_key_values=past, use_cache=True, return_dict=True)
        past = m.past_key_values
        logits = m.logits[:, -1, :]

    return tokenizer.decode(out_ids, skip_special_tokens=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--base-temp", type=float, default=0.8)
    p.add_argument("--temp-jitter", type=float, default=0.05)
    p.add_argument("--min-p", type=float, default=0.01)
    p.add_argument(
        "--output-mode",
        choices=tuple(OUTPUT_MODE_INSTRUCTIONS.keys()),
        default="image-prompt",
    )
    p.add_argument("--project-id", default=None)
    p.add_argument("--reset-project", action="store_true", default=False)
    p.add_argument("--exemplars-json", type=Path, default=None)
    p.add_argument("--db-path", type=Path, default=None)
    p.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "artifacts" / "conceptial_14b" / "CLOUD_14B_SMOKE.md",
    )
    args = p.parse_args()

    exemplars: list[str] = []
    if args.exemplars_json:
        if not args.exemplars_json.is_file():
            print(f"--exemplars-json not found: {args.exemplars_json}", file=sys.stderr)
            return 1
        data = json.loads(args.exemplars_json.read_text(encoding="utf-8"))
        exemplars = [str(x) for x in data.get("exemplars", []) if str(x).strip()]
        print(f"[exemplars] loaded {len(exemplars)} from {args.exemplars_json}", flush=True)

    device = _resolve_device()
    print(f"[load] {args.model_id} on {device}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[load] ready · params={sum(p.numel() for p in model.parameters())/1e9:.1f}B", flush=True)

    # Persistent project memory (same SQLite store as local 3B path).
    store: IdeaMemoryStore | None = None
    prior_summaries: list[str] = []
    project_id = (args.project_id or "").strip() or None
    if project_id:
        store = IdeaMemoryStore(args.db_path)
        if args.reset_project:
            n_removed = store.delete_project(project_id)
            print(f"[anti ] cleared {n_removed} prior rows", flush=True)
        prior_rows = store.list_recent(project_id, limit=64)
        prior_summaries = [r.text for r in prior_rows if r.text][::-1]
        print(f"[anti ] hydrated {len(prior_summaries)} prior summaries from {store.path}", flush=True)
    n_seeded = len(prior_summaries)

    outs: list[dict] = []
    t0 = time.time()
    for i in range(args.n):
        temp = float(args.base_temp + args.temp_jitter * i)
        prompt = format_anti_mode(args.prompt, prior_summaries, exemplars=exemplars, output_mode=args.output_mode)
        text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=temp,
            min_p=float(args.min_p),
        )
        summary = extract_summary(text)
        outs.append({"i": i + 1, "temp": temp, "text": text, "summary": summary, "exclusions": list(prior_summaries)})
        if summary:
            prior_summaries.append(summary)
            if store and project_id:
                try:
                    # We don't have a SteeringEngine hidden-capture path on 14B; persist a
                    # zero-vector placeholder so the row exists for text-level anti-mode
                    # without claiming a hidden-state match. Future: capture L20 hidden if
                    # we extract Harmonic vectors on 14B.
                    hidden_size = int(model.config.hidden_size)
                    zero_h = torch.zeros(hidden_size, dtype=torch.float32)
                    idea_id = f"cloud14b-{int(time.time())}-{i+1:03d}"
                    store.add(project_id, idea_id, zero_h, summary, source=SOURCE_GENERATED)
                except Exception as exc:
                    print(f"[anti  {i+1}/{args.n}] store.add failed: {exc!r}", flush=True)
        print(
            f"[anti  {i+1}/{args.n}] T={temp:.2f} {len(text)} chars · "
            f"summary='{(summary or '(no-summary)')[:60]}' · "
            f"now_excluding={len(prior_summaries)} (seeded={n_seeded})",
            flush=True,
        )
    elapsed = time.time() - t0
    print(f"[anti ] elapsed={elapsed:.1f}s ({elapsed/max(1,len(outs)):.1f}s/idea)", flush=True)

    # Write the smoke report in the same format as the local stack so score_outputs.py
    # can ingest it unchanged.
    lines = [
        "# Conceptial cloud-14B anti-mode smoke",
        "",
        f"**Prompt:** {args.prompt}",
        f"**Model:** {args.model_id} on {device}",
        f"**Samples:** {args.n} · base_temp={args.base_temp} jitter={args.temp_jitter} min_p={args.min_p} mode={args.output_mode}",
    ]
    if exemplars:
        lines.append(f"**Few-shot exemplars:** {len(exemplars)}")
    if project_id:
        lines.append(f"**Project:** {project_id} (seeded={n_seeded} from disk)")
    lines.extend(["", "---", "", "## v17-antimode outputs", ""])  # arm header matches local-stack parser
    for o in outs:
        excl_str = "\n".join(f"  - {e}" for e in o["exclusions"]) if o["exclusions"] else "  (none — first sample)"
        lines.extend(
            [
                f"### Idea {o['i']} (T={o['temp']:.2f})",
                "",
                "_Exclusions presented in prompt:_",
                f"```\n{excl_str}\n```",
                "_Output:_",
                "```",
                o["text"].strip(),
                "```",
                "",
            ]
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
