# Conceptial Benchmark Runbook

Head-to-head Conceptial 14B vs frontier OpenAI models (GPT-4o, GPT-4o-mini)
across 10 creative-operations prompts at N=8 outputs per arm.

## Architecture

- **GPT side**: pre-generated locally via OpenAI API (~$0.30, ~60s for all 10 prompts).
- **Conceptial 14B side**: generated on a RunPod A100 80GB pod (~10 min, ~$0.50).
  Outputs scp'd back to local.
- **Aggregation + scoring**: run locally on the combined results (~$0.30 in
  GPT-4o-mini scoring calls).

## Pre-flight (local, one-time)

1. The prompt suite lives at `config/benchmark_creative_briefs.json`. Edit
   if you want different prompts.
2. The harness script is `scripts/benchmark_creative_briefs.py`. It supports
   three arms: `gpt-4o`, `gpt-4o-mini`, `conceptial-14b`. (There's also a
   `conceptial-3b` arm for local fast iteration; skip for the real benchmark.)
3. Verify your `.env` has `OPENAI_API_KEY` set.

## Step 1 — Generate the OpenAI side (local, ~60s)

```bash
cd ~/Desktop/Projects/BrainstormLM
set -a; source .env; set +a
uv run python scripts/benchmark_creative_briefs.py \
  --arms gpt-4o,gpt-4o-mini --n 8 \
  --skip-existing --skip-scoring \
  --out artifacts/benchmark_v2/run_$(date +%Y%m%d)
```

Produces 20 JSON files (10 prompts × 2 arms) at `artifacts/benchmark_v2/run_<date>/`.
Cost: ~$0.30.

## Step 2 — Generate the Conceptial 14B side (on RunPod A100, ~10 min)

Spin up an A100 80GB pod per `docs/cloud_runpod_pivot.md`, then on the pod:

```bash
cd /workspace
git clone https://github.com/KeshavR77/BrainstormLM.git
cd BrainstormLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync

export HF_TOKEN='hf_PASTE_YOURS'
uv run huggingface-cli login --token "$HF_TOKEN"

uv run python scripts/runpod_batch_14b.py --n 8
# 10 prompts × 8 outputs ≈ 8-12 min wall-clock on A100 80GB.
```

This writes 10 JSON files to `artifacts/benchmark_v2/conceptial_14b_outputs/`.

## Step 3 — Pull 14B outputs back to local (~10s)

On your laptop:

```bash
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
  'root@<pod-ip>:/workspace/BrainstormLM/artifacts/benchmark_v2/conceptial_14b_outputs/*' \
  ~/Desktop/Projects/BrainstormLM/artifacts/benchmark_v2/conceptial_14b_outputs/
```

Then **terminate the pod** (RunPod console → your pod → Terminate). Critical.

## Step 4 — Aggregate, score, summarize (local, ~5 min)

```bash
cd ~/Desktop/Projects/BrainstormLM
set -a; source .env; set +a

# Re-run the harness with all 3 arms; the cached GPT files + cached 14B files
# both get reused. Only the scoring round actually runs new GPT-4o-mini calls.
uv run python scripts/benchmark_creative_briefs.py \
  --arms gpt-4o,gpt-4o-mini,conceptial-14b --n 8 \
  --skip-existing \
  --out artifacts/benchmark_v2/run_$(date +%Y%m%d)

# Produces SUMMARY.json + RESULTS.md in that run dir.

# Then the headline-numbers report:
uv run python scripts/summarize_benchmark.py \
  --run artifacts/benchmark_v2/run_$(date +%Y%m%d)
# Writes HEADLINE_NUMBERS.md with paste-ready aggregate sentences.
```

## Expected outputs

- `SUMMARY.json` — raw per-prompt + aggregate metrics + quality scores
- `RESULTS.md` — markdown table version
- `HEADLINE_NUMBERS.md` — pitch-ready aggregate sentences with
  Vendi/near-dup/quality/archetype-count comparisons across arms

## Total cost + time

| Phase | Time | Cost |
|---|---:|---:|
| OpenAI side generation | ~60s | $0.30 |
| RunPod pod time (boot + load + generate + idle) | ~15 min | $0.50 |
| Scoring (GPT-4o-mini judge) | ~5 min | $0.30 |
| **Total** | **~25 min** | **~$1.10** |

## Reusability

After the first full run, the JSON outputs are cached. Subsequent edits
(e.g., adding a new prompt, swapping a model) only need to regenerate the
changed pieces — `--skip-existing` reuses everything that's already on disk.
