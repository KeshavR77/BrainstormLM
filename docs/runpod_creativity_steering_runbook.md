# RunPod Runbook — `creativity_steering` + `creativity_evaluators` (verified-contrast stack)

End-to-end runbook for the **new** verified-contrast pipeline:

> **vector derivation** (generate → evaluator-verified filter → DiffMeans at every layer)
> → **analysis** (norm / cosine / oracle → candidate layers)
> → **layer sweep** (pick the best vector, scored by the two-stage evaluator).

This is **not** the older Conceptial script pipeline in `docs/runpod_14b_runbook.md`
(`extract_14b_vectors.py` / `sweep_14b_steering.py` / `runpod_batch_14b.py`) — that's the
prompt-framing approach this package was built to replace. Use this doc for the
`creativity-steering` CLI against Qwen2.5-14B.

**Target:** A100 80GB pod. The 14B (bf16, ≈28 GB) co-loads comfortably with the Stage-1
evaluator models (BERT ≈0.4 GB, GPT-2-medium ≈1.5 GB). **Keep the Skywork reward model OFF**
(`run_reward=false`, the default) — a 27B/8B reward model alongside the 14B would not fit.

---

## What runs where

| Phase | Compute | Cost driver |
|---|---|---|
| 1. Derive | 14B generation (pod) + Stage-1 (pod) + **Stage-2 judges (API)** | judge calls dominate |
| 2. Analyze | numpy only (pod or laptop) | negligible |
| 3. Sweep   | 14B generation (pod) + Stage-1 (pod) + **Stage-2 judges on top-k cells (API)** | generation time + judges |

Judge calls are **disk-cached** (`.cache/creativity_evaluators`), so re-runs are free. Stage-2
is gated twice: by `stage1_percentile_floor` (skip the weakest responses) and, in the sweep,
by `judge_top_k` (only the k best cells get judged).

---

## Phase 0 — Local pre-flight (laptop)

```bash
cd ~/Documents/BrainstormLM
git pull origin <branch>            # the branch with this stack (feat/creativity-evaluators)

# Prove the whole pipeline composes offline before spending a cent — no models, no API:
PYTHONPATH=creativity_steering python creativity_steering/scripts/smoke_end_to_end.py
PYTHONPATH=creativity_steering python creativity_steering/scripts/smoke_end_to_end.py --with-judge
# Each prints "OK — end-to-end pipeline composed ... best vector beats baseline."

# Have your judge keys ready (the run reads them from the environment):
grep -E '^(ANTHROPIC|OPENAI|GOOGLE)_API_KEY=' .env   # confirm all three are set
```

---

## Phase 1 — Spin up the pod + install both packages

1. RunPod console → **A100 80GB** pod. Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` (or
   any recent torch+CUDA). Disk: ≥ 80 GB (14B weights + BERT/GPT-2 + activations + outputs).
   Expose SSH (port 22). Deploy, wait ~60s.

Once SSH'd in:

```bash
cd /workspace
git clone https://github.com/KeshavR77/BrainstormLM.git
cd BrainstormLM
git checkout feat/creativity-evaluators     # the branch carrying this stack
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# ONE virtualenv, BOTH packages installed editable into it.
uv venv && source .venv/bin/activate
uv pip install -e "./creativity_evaluators[stage1,nlp,judges]"   # BERT+GPT-2, spaCy, judge SDKs
uv pip install -e "./creativity_steering[hf]"                     # Qwen backend (torch+transformers)
python -m spacy download en_core_web_sm                           # POS-template overlap (optional)
```

> With both installed in the same venv, `import creativity_evaluators` resolves directly and
> the in-repo path bridge is a no-op. (If you instead run from source without installing,
> the bridge adds `creativity_evaluators/src` to `sys.path` automatically.)

**Export the judge API keys + HF token** (paste your values):

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GOOGLE_API_KEY='...'
export HF_TOKEN='hf_...'; uv run huggingface-cli login --token "$HF_TOKEN"   # Qwen2.5 is public; harmless
```

**Sanity checks:**

```bash
# Both packages import in one process:
python -c "import creativity_steering, creativity_evaluators; print('both import OK')"
# Model is reachable:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct'); print('tokenizer ok')"
# Judges are wired (keys present):
python -c "import os; assert all(os.getenv(k) for k in ['ANTHROPIC_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY']); print('keys ok')"
```

---

## Phase 1b — Cheap dry-run on the real 14B (recommended, ~judge-free)

Validate the full pipeline on the real model **without** judge spend first. This turns Stage 2
off everywhere (derivation filter falls back to the Stage-1 composite; the two-stage sweep
scores Stage-1-only):

```bash
jq '.filtering.enable_judge=false' creativity_steering/configs/runpod_qwen14b.json > /tmp/dryrun.json

creativity-steering derive  --config /tmp/dryrun.json --out artifacts/qwen14b_dryrun
creativity-steering analyze --run artifacts/qwen14b_dryrun --position last_response_token
LAYERS=$(python -c "import json;print(','.join(map(str,json.load(open('artifacts/qwen14b_dryrun/analysis/last_response_token/report.json'))['candidate_layers'])))")
creativity-steering sweep   --run artifacts/qwen14b_dryrun --config /tmp/dryrun.json --position last_response_token --layers "$LAYERS"
```

If derive prints sane `n_positive`/`n_negative` (≥ `min_per_class`), analyze localizes a
plausible candidate band, and the sweep prints a `best cell` with `delta > 0`, the wiring is
good — proceed to the judged run.

> ⚠️ `min_per_class=100` (paper N). If derive raises *"verified subset too small"*, you have
> too few responses per class after the quantile cut — raise `generation.samples_per_prompt`,
> add topics, or relax `creative_quantile`/`neutral_quantile` in the config.

---

## Phase 2 — The real judged run

```bash
# 1) Derive: generate → evaluator_stack filter (DSI + 3-judge ensemble) → DiffMeans/layer
creativity-steering derive --config creativity_steering/configs/runpod_qwen14b.json \
                           --out artifacts/qwen14b_run
#   Watch: the JSON summary (n_examples / n_positive / n_negative / n_dropped / n_layers).

# 2) Analyze → candidate layers
#    Pass --position explicitly: the CLI otherwise defaults to the alphabetically-first
#    captured position (assistant_start), not the primary last_response_token.
creativity-steering analyze --run artifacts/qwen14b_run --position last_response_token
LAYERS=$(python -c "import json;print(','.join(map(str,json.load(open('artifacts/qwen14b_run/analysis/last_response_token/report.json'))['candidate_layers'])))")
echo "candidate layers: $LAYERS"

# 3) Sweep the candidate layers, scored by the two-stage stack (judges on top-k cells)
creativity-steering sweep --run artifacts/qwen14b_run \
                          --config creativity_steering/configs/runpod_qwen14b.json \
                          --position last_response_token --layers "$LAYERS"
#   Watch: "best cell: layer=L alpha=a composite=... (delta=+..., judged=True)"
```

The config (`creativity_steering/configs/runpod_qwen14b.json`) already sets:
`filtering.method="evaluator_stack"`, `enable_judge=true`, `generator_family="qwen"`,
`patching.sweep_evaluator="two_stage"`, `judge_top_k=3`. Tune `judge_top_k` up for a finer
final comparison (more judge cost) or down to save money.

**Artifacts produced (under `artifacts/qwen14b_run/`):**
- `vectors/<position>/{vectors.npz,manifest.json}` — DiffMeans vectors at every layer.
- `derivation_summary.json` — contrast-set sizes.
- `analysis/<position>/{report.json,report.md}` — candidate layers + norm/cosine/oracle.
- `patching/<position>/sweep_results.json` — per-cell two-stage scores + the chosen `best`.

---

## Phase 3 — Pull results back + terminate

```bash
# On your laptop (pod IP/port from RunPod → Connect → SSH):
cd ~/Documents/BrainstormLM
scp -P <port> -i ~/.ssh/id_ed25519 -r \
  'root@<ip>:/workspace/BrainstormLM/artifacts/qwen14b_run' artifacts/

# Inspect the winner:
python -c "import json;b=json.load(open('artifacts/qwen14b_run/patching/last_response_token/sweep_results.json'))['best'];print(b)"
```

**Then terminate the pod** (RunPod → Terminate). Pod time is the dominant cost.

---

## Cost / scale notes

- **Derivation** is the larger judge bill: with `samples_per_prompt=6` over 48 topics, the
  creative pool is ~288 responses; Stage-2 judges run on everything above the Stage-1 floor
  (~80%), ×3 judges. The neutral pool decodes **greedily** (deterministic) → ~1 distinct
  response/topic regardless of `samples_per_prompt`; size it via topic count.
- **Sweep** generation = `len(layers) × len(alphas) × briefs × eval_samples` decodes on the
  14B; judges run only on `judge_top_k+1` cells (top-k + baseline).
- Re-runs are cheap: judge results are disk-cached by `(prompt, response, config_hash)`.
- Start with the Phase-1b dry-run; only flip judges on once the shapes look right.

---

## Quick reference

```bash
# === POD ===
cd /workspace/BrainstormLM && source .venv/bin/activate
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GOOGLE_API_KEY=...
creativity-steering derive  --config creativity_steering/configs/runpod_qwen14b.json --out artifacts/qwen14b_run
creativity-steering analyze --run artifacts/qwen14b_run --position last_response_token
LAYERS=$(python -c "import json;print(','.join(map(str,json.load(open('artifacts/qwen14b_run/analysis/last_response_token/report.json'))['candidate_layers'])))")
creativity-steering sweep   --run artifacts/qwen14b_run --config creativity_steering/configs/runpod_qwen14b.json --position last_response_token --layers "$LAYERS"

# === LAPTOP ===
scp -P <port> -i ~/.ssh/id_ed25519 -r 'root@<ip>:/workspace/BrainstormLM/artifacts/qwen14b_run' artifacts/
```
