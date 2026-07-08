# Cloud 14B pivot — RunPod setup guide

You confirmed Vendi parity with documented cloud 14B Harmonic numbers on local 3B (4.03 vs
4.02). The remaining ~2-3× lift expected on 14B comes from raw per-idea polish, not
diversity. This guide spins up Qwen2.5-14B-Instruct on RunPod and reuses our entire
anti-mode + persistent-memory stack at the higher model scale.

## GPU + provisioning

**Recommended**: A100 80GB on RunPod Community Cloud (~$1.89/hr at time of writing).

- Qwen2.5-14B-Instruct fp16 weights: ~28GB. A100 80GB has comfortable headroom for
  KV cache + activations.
- H100 80GB is faster (~3-4×) but $3-4/hr. Only worth it if you plan to run >30 min of
  inference; for a 1-2hr validation run, A100 is the sweet spot.
- **Don't** pick A100 40GB — Qwen2.5-14B fp16 will OOM with even a small batch.

**Disk**: 100GB volume (room for model weights + Python venv + smoke artifacts).

**Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.1.0-devel-ubuntu22.04` (or whichever
torch≥2.1 + py3.11 image is current).

## One-shot deploy script

**The Day-2 consolidation collapsed `shrooms-llm` into `BrainstormLM`.** Only one repo to clone.

After you SSH into the pod (use the RunPod web-SSH or `ssh root@<pod-ip> -p <pod-ssh-port>`):

```bash
# Pod-side: clone only BrainstormLM. All steering vectors, config, and the legacy engine
# subpackage are now in-tree.
cd /workspace
git clone <your BrainstormLM remote> BrainstormLM
cd BrainstormLM

# Install deps. uv is fastest; falls back to pip.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # uv on PATH
uv sync

# Hugging Face auth (Qwen2.5-14B is gated for some accounts; this auth covers both cases)
export HF_TOKEN=<your hf token>
huggingface-cli login --token "$HF_TOKEN"

# Run a smoke generation. The standalone scripts/smoke_14b_antimode.py helper has been
# retired; use the current batch driver instead (see `scripts/runpod_batch_14b.py --help`
# and docs/runpod_14b_runbook.md for the up-to-date invocation and flags).
mkdir -p artifacts/conceptial_14b data
.venv/bin/python3 scripts/runpod_batch_14b.py --help
```

Reuse the same `--project-id` across successive runs (dropping any reset flag after the
first) so the IdeaMemoryStore accumulates exclusions across sessions, just like the local
3B path.

## Cost estimate

| Operation | Wall-clock on A100 80GB | $ |
|---|---|---|
| Model load (first time) | ~60s | — |
| 1 generation @ 320 tokens | ~3-5s (vs 30-60s on M-series MPS) | — |
| 1 session of 5 ideas | ~25s | $0.013 |
| Full 3-shot demo flow (15 ideas) | ~75s | $0.04 |
| 1 hour of iteration | 60min | **$1.89** |

You should expect to spend $5-10 total for the validation run + a few iterations + a
spare buffer.

## Bring results back to local

```bash
# On the pod
rsync -av artifacts/conceptial_14b/ root@<your-local-ip>:~/Desktop/Projects/BrainstormLM/artifacts/conceptial_14b/

# Or just copy the JSON via scp
scp -P <pod-ssh-port> root@<pod-ip>:/workspace/BrainstormLM/artifacts/conceptial_14b/*.md .

# Score locally with the same tooling we've been using
python3 scripts/score_outputs.py --md artifacts/conceptial_14b/CLOUD_14B_S1.md
```

## What to expect on 14B vs local 3B

| Axis | Local 3B (current) | Cloud 14B (expected) | Mechanism |
|---|---|---|---|
| Vendi (aggregate, 15 outputs) | 4.03 | 4.0-4.3 | Bounded by anti-mode/persistence; we already match docs |
| Composite quality | 3.83 | **4.3+** | Larger model = better individual ideas |
| Creativity | 3.60 | **4.0+** | Less mode collapse on hard briefs |
| Specificity | 3.40 | **4.2+** | More precise visual / strategic vocabulary |
| Factual accuracy | 3B hallucinates Stanford mascot/colors | 14B knows the Cardinal | Better world knowledge |

The single biggest demo-relevant lift on 14B is **factual specificity** — the 3B's
incorrect "bulldog mascot" and "yellow and brick red colors" issues should largely
disappear.

## What's NOT carried over from 3B

- **v17 purified L14+L18 steering** — the vectors were extracted from 3B (hidden dim 2048).
  14B has hidden dim 5120; the vectors are dimensionally incompatible. Re-extracting
  Harmonic vectors on 14B is a separate research task (legacy script lived at
  `shrooms-llm/scripts/extract_harmonic_steering_vectors.py`; the pre-Day-2 archive repo
  still has it). Initial cloud smoke is "raw 14B + prompt-level anti-mode + persistence."
- **min_p=0.01** — same default; carries over.
- **Brief-aware output modes** — carry over unchanged.

## Decision points after the cloud smoke

1. If 14B raw + anti-mode meets the wow bar → lock cloud config, deploy demo on the same
   pod or migrate to a managed inference service (Anyscale, Together, Modal).
2. If quality is good but diversity is lower than 4.0 → extract Harmonic vectors fresh on
   14B and re-run. This is a multi-hour task; ask before committing.
3. If raw 14B is wow → consider whether the 14B model cost is acceptable for production.
   Per-call cost on A100 is ~$0.001 per concept; cloud-managed inference is similar.
