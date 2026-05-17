# RunPod 14B Run — Step-by-Step

End-to-end runbook for the next Conceptial 14B benchmark session. Covers
**vector extraction → tuning sweep → benchmark generation → local aggregation**,
with troubleshooting checks at each step.

**Target:** A100 80GB pod. **Estimated total:** ~60-75 min on the pod
(~$2.00), ~5 min local (~$0.30 scoring).

---

## What's different this run vs the previous one

1. The Stanford-engineering brief is now: *"Give me a creative logo for an
   engineering club at Stanford. I want a three-sentence visual description
   suitable for an image-generation prompt."* (no concept name, three
   sentences instead of one paragraph). Already updated in
   `config/benchmark_creative_briefs.json`.
2. We sweep `creative_layer × creative_alpha` BEFORE generating the real
   benchmark, picking the best operating point empirically rather than
   trusting 3B-inherited defaults.
3. `scripts/runpod_batch_14b.py` auto-reads the sweep's chosen config from
   `data/steering_14b/sweep/chosen_config.json` if present — no manual config
   edit between sweep and benchmark.

---

## Phase 0 — Local pre-flight (do this once, on your laptop)

```bash
cd ~/Desktop/Projects/BrainstormLM
git pull origin main                # ensure scripts are current
set -a; source .env; set +a         # OPENAI_API_KEY needs to be in shell
```

**Check the OpenAI side is current:**

```bash
ls artifacts/benchmark_v2/run_14b_pending/*.gpt-4o.json | wc -l
# Expect: 10
```

If it's not 10, regenerate:

```bash
uv run python scripts/benchmark_creative_briefs.py \
  --arms gpt-4o,gpt-4o-mini --n 8 \
  --skip-existing --skip-scoring \
  --out artifacts/benchmark_v2/run_14b_pending
```

This costs ~$0.30 and takes ~60s. Since we just changed the
Stanford-engineering brief, **delete that one cached file first** so it
regenerates with the new brief:

```bash
rm -f artifacts/benchmark_v2/run_14b_pending/logo-stanford-engineering.*.json
```

Then run the regen command above.

---

## Phase 1 — Spin up the pod

1. RunPod console → **A100 80GB** community-cloud pod.
2. Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` (or any recent torch+CUDA).
3. Disk: at least 80GB persistent volume (Qwen2.5-14B weights ≈ 28GB on disk
   plus tokenizer + activations + outputs).
4. Expose SSH (port 22) — you'll scp results back at the end.
5. Click **Deploy**. Wait ~60s for boot.

Once SSHd in:

```bash
cd /workspace
git clone https://github.com/KeshavR77/BrainstormLM.git
cd BrainstormLM
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync       # ~3-5 min first time

export HF_TOKEN='hf_PASTE_YOURS'
uv run huggingface-cli login --token "$HF_TOKEN"
```

**Sanity check that the model is reachable:**

```bash
uv run python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')
print('tokenizer ok, vocab', t.vocab_size)
"
```

If this errors with `401 Unauthorized`: re-run the `huggingface-cli login`
step — the token didn't persist.

If it errors with `OutOfDiskError`: pod's disk is too small. Terminate and
spin a new one with at least 80GB volume.

---

## Phase 2 — Extract base 14B steering vectors (~10-15 min, ~$0.45)

```bash
uv run python scripts/extract_14b_vectors.py
```

**What's happening:** 32 contrast prompts × 8 system-prompt framings = 256
forward passes through Qwen2.5-14B at L19 and L24, plus mean-difference + SVD
orthogonal-projection purification.

**Watch for in stdout:**

```
[1/3] Deriving creative CAA vector (positive vs generic) …
    [creative] L19: 8/32 pairs
    [creative] L19: 16/32 pairs
    ...
Hyper-purifying creative vector …
  retained=0.XX  neg_before=0.YY  neg_after<0.08  rank_removed=5
```

**Edge cases:**

- ❌ `ValueError: hyper purification over-scrubbed source direction:
  retained_source_cosine=0.0XX < min_source_cosine=0.15` — the negative bank
  ate too much of the creative direction. The framings in
  `config/contrast_pairs_14b.json` may be too aggressive. Rerun with
  `--no-purify` to save the raw CAA vectors and proceed; the sweep will still
  work.
- ❌ `CUDA out of memory` — the 80GB pod should not hit this for 14B at fp16.
  If it does, restart the pod (memory is fragmented) and retry. If still
  failing, run with smaller `--limit-prompts 16`.
- ✅ End-state: `data/steering_14b/hyper_creative_purified_vector_L19.pt` and
  `hyper_technical_purified_vector_L24.pt` plus matching `_manifest.json`
  files. Confirm with `ls data/steering_14b/`.

**Skip Phase 2 if you've already extracted on a previous pod** and committed
the artifacts: `ls data/steering_14b/*.pt` should show 2 files. If they're
already on this pod from `git clone`, skip.

---

## Phase 3 — Sweep creative_layer × creative_alpha (~25-35 min, ~$0.90)

This is the new step that mitigates the two open caveats (alpha tuning + layer
specificity).

```bash
uv run python scripts/sweep_14b_steering.py
```

**What's happening:**

- **Step A** — Re-extracts creative vectors at L17, L19, L21, L23 in one
  forward-pass batch (same 32 contrast prompts as Phase 2; just captures more
  layers).
- **Step B + C** — For each of 4 layers × 4 alphas = 16 cells, generates N=5
  samples on the new Stanford-engineering brief with iterative anti-mode +
  memory + min-p sampling. Scores each cell by Vendi + near-dup + a coherence
  filter (length ≥ 80 chars, no degenerate token runs, ≥ 70% unique lines).
- **Step D** — Picks the (layer, alpha) cell with the highest Vendi that
  passes the coherence filter for ≥ 4/5 samples.

**Watch for in stdout:**

```
[A] Deriving creative + negative vectors at all candidate layers …
    multi-layer extract: 8/32 prompts
    ...
  Purifying creative @ L17 …
    retained=0.XX neg_after=0.0X rank_removed=Y
  Purifying creative @ L19 …
  ...

[B/C] Sweeping cells on the brief …
  L17 α=1.50: Vendi=2.XXX  near_dup=0.0XX  coherent=5/5  PASS  (XXs)
  L17 α=2.00: Vendi=2.XXX  near_dup=0.0XX  coherent=5/5  PASS
  L17 α=2.62: Vendi=2.XXX  near_dup=0.0XX  coherent=4/5  PASS
  L17 α=3.00: Vendi=1.XXX  near_dup=0.0XX  coherent=2/5  FAIL  ← too hot
  L19 α=1.50: ...
  ...

Winner: L19 α=2.00  Vendi=3.XXX  near_dup=0.0XX  coherent=5/5
```

**Edge cases:**

- ❌ **All cells FAIL the coherence filter** — outputs are gibberish across
  the board. Most likely: alpha range is way too high, or a vector got
  corrupted in extraction. The script falls back to highest-Vendi cell
  regardless, but **inspect outputs manually** before trusting it:
  ```bash
  uv run python -c "
  import json
  r = json.load(open('data/steering_14b/sweep/sweep_results.json'))
  for c in r['cells']:
      print(f\"L{c['creative_layer']} α={c['creative_alpha']:.2f}:\")
      for o in c['outputs'][:1]:
          print('  ', o['text'][:200])
  "
  ```
  If they're all gibberish, retry with halved alphas:
  `--creative-alphas 0.5,1.0,1.5,2.0`.

- ❌ **The "winner" has Vendi < 2.0** — the steering isn't actually adding
  diversity. Either the vectors are too weak (try purify retained-cosine
  diagnostic in the manifest) or alphas are too low. Try
  `--creative-alphas 2.0,3.0,4.0,5.0`.

- ❌ **All cells produce essentially identical Vendi (within ±0.05)** —
  steering may not be wired correctly. Quick diagnostic:
  ```bash
  uv run python scripts/runpod_batch_14b.py \
    --n 3 --limit-prompts 1 --no-steering \
    --out-dir /tmp/baseline_no_steering
  uv run python scripts/runpod_batch_14b.py \
    --n 3 --limit-prompts 1 \
    --out-dir /tmp/baseline_steering
  diff <(jq -r '.outputs[].text' /tmp/baseline_no_steering/*.json | head -c 500) \
       <(jq -r '.outputs[].text' /tmp/baseline_steering/*.json | head -c 500)
  ```
  If the two are byte-identical, steering hooks aren't firing.

- ✅ End-state: `data/steering_14b/sweep/chosen_config.json` exists with
  fields `creative_layer`, `creative_alpha`, `creative_vector_path`.
  Confirm:
  ```bash
  cat data/steering_14b/sweep/chosen_config.json
  ```

---

## Phase 4 — Generate the real 14B benchmark (~12-18 min, ~$0.55)

```bash
uv run python scripts/runpod_batch_14b.py --n 8
```

**The script auto-detects** `data/steering_14b/sweep/chosen_config.json` and
applies the winning (layer, alpha) without manual config edits. Output
includes a line like:

```
Sweep override: data/steering_14b/sweep/chosen_config.json → L19 α=2.00
```

**To override** and use the corpius defaults instead of the sweep's pick:
add `--no-sweep-override`.

**Watch for:**

```
[logo-stanford-engineering        ] wrote 8 outputs in XXs → logo-stanford-engineering.conceptial-14b.json
[logo-dtc-skincare                 ] wrote 8 outputs in XXs → logo-dtc-skincare.conceptial-14b.json
...
```

**Edge cases:**

- ❌ One prompt takes >60s per sample — the model is dropping into long
  rambles. Look at one output:
  ```bash
  jq -r '.outputs[0].text' artifacts/benchmark_v2/conceptial_14b_outputs/<prompt-id>.conceptial-14b.json | head
  ```
  If it's a repeating wall of text, increase `--max-new-tokens 250` (default
  350) or check that the tokenizer's `eos_token_id` is being respected. The
  v17 schedule's decay should normally land EOS within 200-300 tokens.

- ❌ A single prompt's outputs all duplicate (Vendi ≈ 1.0) — anti-mode
  exclusion list may not be threading through. Inspect:
  ```bash
  jq -r '.outputs[].summary' artifacts/benchmark_v2/conceptial_14b_outputs/<prompt-id>.conceptial-14b.json
  ```
  If summaries are all identical, iterative-anti-mode isn't working. Try
  `--no-iterative-anti-mode` to fall back to prompt-level anti-mode only,
  and compare.

- ✅ End-state: 10 JSON files in `artifacts/benchmark_v2/conceptial_14b_outputs/`,
  each with 8 outputs. Confirm:
  ```bash
  ls artifacts/benchmark_v2/conceptial_14b_outputs/*.json | wc -l    # 10
  jq -r '.n' artifacts/benchmark_v2/conceptial_14b_outputs/*.json | sort -u   # 8
  ```

---

## Phase 5 — Pull results back to laptop (~30s)

Note the pod's IP and SSH port (RunPod console → pod → Connect → SSH).

```bash
# On your laptop:
cd ~/Desktop/Projects/BrainstormLM

# Pull the sweep artifacts + vectors so we keep them in the repo
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
  -r 'root@<pod-ip>:/workspace/BrainstormLM/data/steering_14b' \
  data/

# Pull the benchmark outputs
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
  'root@<pod-ip>:/workspace/BrainstormLM/artifacts/benchmark_v2/conceptial_14b_outputs/*' \
  artifacts/benchmark_v2/conceptial_14b_outputs/
```

**Sanity check:**

```bash
ls artifacts/benchmark_v2/conceptial_14b_outputs/*.json | wc -l    # 10
ls data/steering_14b/*.pt | wc -l                                  # 2 (creative + technical)
ls data/steering_14b/sweep/*.pt | wc -l                            # 4 (one per swept layer)
```

**Then terminate the pod.** RunPod console → your pod → Terminate. Critical
— pod cost continues until terminated.

---

## Phase 6 — Aggregate, score, summarize (local, ~5 min, ~$0.30)

```bash
set -a; source .env; set +a

uv run python scripts/benchmark_creative_briefs.py \
  --arms gpt-4o,gpt-4o-mini,conceptial-14b --n 8 \
  --skip-existing \
  --out artifacts/benchmark_v2/run_14b_final

uv run python scripts/summarize_benchmark.py \
  --run artifacts/benchmark_v2/run_14b_final
```

**Watch for:**

- The harness uses `--skip-existing` so cached GPT files and cached 14B files
  are reused; only GPT-4o-mini scoring calls run fresh.
- `RESULTS.md` and `HEADLINE_NUMBERS.md` land in
  `artifacts/benchmark_v2/run_14b_final/`.

**Edge cases:**

- ❌ Scoring fails with rate limit — wait 60s and re-run. `--skip-existing`
  reuses everything already on disk.
- ❌ `HEADLINE_NUMBERS.md` is missing — check that
  `artifacts/benchmark_v2/run_14b_final/SUMMARY.json` exists. If not, the
  benchmark step didn't complete; re-run it.

---

## Phase 7 — Commit + push

```bash
git add data/steering_14b artifacts/benchmark_v2/conceptial_14b_outputs \
        artifacts/benchmark_v2/run_14b_final
git status

git checkout -b results/14b-benchmark-$(date +%Y%m%d)
git commit -m "$(cat <<'EOF'
add 14B benchmark results — sweep-picked operating point + final outputs

Vectors: data/steering_14b/{creative_L19, technical_L24} + sweep dir
Outputs: 10 prompts × 8 samples in conceptial_14b_outputs/
Summary: artifacts/benchmark_v2/run_14b_final/{SUMMARY.json, RESULTS.md, HEADLINE_NUMBERS.md}

Sweep winner: L<X> α=<Y> selected from {17,19,21,23} × {1.5, 2.0, 2.62, 3.0}.
EOF
)"

git push -u origin results/14b-benchmark-$(date +%Y%m%d)
# Open PR via the URL printed above; merge.
```

---

## Total cost + time

| Phase | Time | Cost |
|---|---:|---:|
| 0. Local pre-flight | ~3 min | ~$0.30 (OpenAI side) |
| 1. Pod boot + setup | ~6 min | ~$0.20 |
| 2. Extract base vectors | ~12 min | ~$0.40 |
| 3. Sweep | ~30 min | ~$0.95 |
| 4. Generate benchmark | ~15 min | ~$0.50 |
| 5. scp back + terminate | ~1 min | ~$0.05 |
| 6. Local aggregation + scoring | ~5 min | ~$0.30 |
| **Total** | **~75 min** | **~$2.70** |

If you skip the sweep (only do this if you've sweep-confirmed once already
and are re-running the same benchmark): subtract 30 min and ~$0.95.

---

## What to do if you have to abort partway through

The work is checkpointed at every phase:

- After Phase 2: `data/steering_14b/*.pt` exists → next pod can skip Phase 2.
- After Phase 3: `data/steering_14b/sweep/chosen_config.json` exists → next
  pod can skip both Phase 2 and Phase 3.
- After Phase 4: outputs are already on the pod; `scp` and you're done.
- After Phase 5: laptop has everything; you can terminate the pod with no
  data loss.

**Always terminate the pod when you stop working.** Pod time is the dominant
cost.

---

## Quick-reference command list (copy-paste-friendly)

```bash
# === ON THE POD ===
cd /workspace/BrainstormLM
export HF_TOKEN='hf_PASTE_YOURS'
uv run huggingface-cli login --token "$HF_TOKEN"
uv run python scripts/extract_14b_vectors.py           # Phase 2
uv run python scripts/sweep_14b_steering.py            # Phase 3
uv run python scripts/runpod_batch_14b.py --n 8        # Phase 4

# === ON THE LAPTOP ===
scp -P <port> -i ~/.ssh/id_ed25519 -r \
  'root@<ip>:/workspace/BrainstormLM/data/steering_14b' data/
scp -P <port> -i ~/.ssh/id_ed25519 \
  'root@<ip>:/workspace/BrainstormLM/artifacts/benchmark_v2/conceptial_14b_outputs/*' \
  artifacts/benchmark_v2/conceptial_14b_outputs/
set -a; source .env; set +a
uv run python scripts/benchmark_creative_briefs.py \
  --arms gpt-4o,gpt-4o-mini,conceptial-14b --n 8 --skip-existing \
  --out artifacts/benchmark_v2/run_14b_final
uv run python scripts/summarize_benchmark.py \
  --run artifacts/benchmark_v2/run_14b_final
```
