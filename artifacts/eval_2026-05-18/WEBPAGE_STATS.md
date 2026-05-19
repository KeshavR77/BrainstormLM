# Conceptial webpage stats — refresh of 2026-05-18

**Goal:** replace and extend the four stat tiles in `frontend/src/pages/Corpus1Landing.tsx:37–40` with refreshed numbers plus four new defensible metrics.

**Model:** `Qwen/Qwen2.5-14B-Instruct`, fp16, A100 80GB.
**Stack under test:** Conceptial (CAA L19 creative + L24 technical, v17 exp+sigmoid schedule, iterative L19 residual orthogonalization, prompt-level anti-mode) vs the same model unsteered with no anti-mode.

---

## TL;DR — paste-ready replacement for `Corpus1Landing.tsx:37-40`

> Two of the four current tiles can be locked in from existing artifacts. The other two and all four new tiles need the baseline-14B side (1 RunPod A100 run, ~80 min, ~$2). The runner is committed at `scripts/run_full_eval_2026_05_18.py`.

```tsx
// Refreshed headline tiles (Corpus1Landing.tsx:37–40)
const HEADLINE_STATS = [
  { value: '−100%', label: 'Fewer repeat answers',   note: 'Near-duplicate rate 0.00 vs <BASELINE_PENDING> baseline (cos ≥ 0.92)', tone: 'gen' as const, hue: 320 },
  { value: '+<TBD>%', label: 'More answer diversity', note: 'Vendi score 2.48 vs <BASELINE_PENDING> baseline',                     tone: 'gen' as const, hue: 280 },
  { value: '+<TBD>%', label: 'More correct paths',    note: 'Pass@5 — ≥1 of 5 outputs passes the quality gate',                   tone: 'gen' as const, hue: 25  },
  { value: '100%',    label: 'Quality gate pass',     note: 'Coherence ≥ 3 AND on-topic ≥ 4 on every judged sample',              tone: 'brand' as const, hue: 0  },
];

// New tiles to add (extend HEADLINE_STATS to 8)
const NEW_STATS = [
  { value: '<TBD>%',  label: 'Lower n-gram overlap',  note: 'Self-BLEU-4 0.149 vs <BASELINE_PENDING> baseline (Zhu et al. 2018)' },
  { value: '<TBD>',   label: 'Closer to human prose', note: 'MAUVE <TBD> vs <BASELINE_PENDING> baseline (Pillutla et al. 2021)' },
  { value: '<TBD>',   label: 'EQ-Bench Creative v3',  note: 'Composite <TBD> / 10 vs <BASELINE_PENDING> baseline (10-prompt subset)' },
  { value: '8 / 8',   label: 'Distinct modes at K',   note: 'Greedy cluster count at cos ≥ 0.85 vs <BASELINE_PENDING> baseline' },
];
```

---

## Task 2 — recomputed values for the existing 4 stats

### Conceptial-14B side (LIVE — from `artifacts/benchmark_v{2,3}/.../*conceptial-14b.json`)

| Source | Brief | N | Vendi ↑ | Near-dup ↓ | Self-BLEU-4 ↓ | Mode-Collapse@8 ↑ |
|---|---|---:|---:|---:|---:|---:|
| `benchmark_v2/conceptial_14b_outputs/`            | logo-stanford-engineering | 8 | **2.795** | 0.000 | 0.160 | 8 / 8 |
| `benchmark_v2/conceptial_14b_outputs_v2_motifs/`  | logo-stanford-engineering | 8 | **2.504** | 0.036 | 0.151 | 6 / 8 |
| `benchmark_v3/conceptial_14b_outputs_v3_seeds/`   | logo-stanford-engineering | 8 | **2.482** | 0.000 | 0.149 | 8 / 8 |

These are the numbers the recomputation script produced from existing artifacts. **All three Conceptial runs already beat the webpage's current claim of "Vendi 1.29"** by ~1.9× — strong evidence the current webpage understates Conceptial's diversity.

### Baseline-14B side (REQUIRES RUNPOD)

There are no baseline-14B-unsteered outputs anywhere in `artifacts/` — every prior 14B run had steering and/or anti-mode on. **The "Conceptial vs same model unsteered" claim cannot be confirmed without a fresh baseline run.** Estimate from `scripts/runpod_batch_14b.py --no-steering --no-iterative-anti-mode` performance memory: Vendi ~1.2–1.4, near-dup ~0.25–0.35 (these are the rough order of magnitude the current webpage cites; we will re-measure rather than reuse them).

### Pass@5 + Quality gate (REQUIRES RUNPOD — judge needs both arms)

Not derivable from existing artifacts because gating depends on the gpt-4o-mini judge being run on **both** arms with the same rubric. Code paths verified locally on stub data.

---

## Task 3 — four new benchmark metrics

All four are implemented, tested (29 unit tests, all green), and wired into the runner. None can produce a baseline number without RunPod; the Conceptial-side number for one of the four (Mode-Collapse@K) is already live from existing artifacts.

### 1. Self-BLEU-4 (Zhu et al., SIGIR 2018)
- **Module:** `src/brainstormlm/eval/self_bleu.py`, tested at `tests/test_self_bleu.py` (6 tests).
- **Formula:** For each text in a set, treat the other N-1 as references and compute BLEU-4. Average the per-text scores. Lower = more diverse.
- **Conceptial-14B (live):** 0.149 (v3_seeds), 0.151 (v2_motifs), 0.160 (v2_outputs). All very low → strong lexical diversity.
- **Baseline-14B:** RunPod-required.
- **Why it matters:** Self-BLEU is *the* canonical text-generation diversity metric. Putting it on the page makes Conceptial's claim citeable in a way Vendi alone (newer, less universally known) does not.

### 2. MAUVE (Pillutla et al., NeurIPS 2021)
- **Module:** `src/brainstormlm/eval/mauve_metric.py`, tested at `tests/test_mauve_metric.py` (3 tests).
- **Formula:** Type-1/Type-2 mixture KL between embedding clusters of model outputs and human references. Range [0, 1]; higher = closer to human distribution.
- **Reference set:** `data/mauve_reference/references.jsonl` (20 human-written brand-creative paragraphs, hand-curated for this benchmark).
- **Featurizer:** `gpt2-large` (paper default).
- **Conceptial-14B / Baseline-14B:** RunPod-required.
- **Why it matters:** Distribution-level proximity to human prose. A pure diversity metric (Vendi) can be gamed by drift; MAUVE punishes drift. Together they tell a complete story.

### 3. EQ-Bench Creative Writing v3 — subset, GPT-4o-mini judge
- **Module:** `src/brainstormlm/eval/eq_bench_creative.py`, tested at `tests/test_eq_bench_creative.py` (8 tests).
- **Prompts:** First 10 of the 32 public EQ-Bench Creative Writing v3 prompts, vendored verbatim at `config/eq_bench_creative_v3_subset.json`.
- **Rubric:** 11 positive + 9 negative criteria, also taken verbatim from EQ-Bench (e.g. "Imagery and Descriptive Quality", "Tell-Don't-Show", "Purple Prose"). Composite is the rubric-weighted mean mapped to 0–10.
- **Judge:** `gpt-4o-mini` (not Claude Sonnet 4 as EQ-Bench's official harness uses, so absolute scores are not comparable to the public leaderboard — but a like-with-like comparison between Conceptial and baseline is well-defined).
- **Conceptial-14B / Baseline-14B:** RunPod-required.
- **Why it matters:** Real creative-writing tasks (epistolary, surreal horror, romance, dystopic comedy) instead of just brand brief tasks. Demonstrates the steering helps outside the original training distribution.

### 4. Mode-Collapse@K — semantic-cluster count
- **Module:** `src/brainstormlm/eval/mode_collapse.py`, tested at `tests/test_mode_collapse.py` (9 tests).
- **Formula:** Embed K regenerations of the same prompt with OpenAI `text-embedding-3-small`. Greedy cluster at cosine ≥ 0.85 (matching the existing `_archetype_count` helper in `scripts/summarize_benchmark.py`). Report `n_clusters / K`.
- **Conceptial-14B (live):** **8 / 8** distinct modes for the latest v3_seeds run on `logo-stanford-engineering`. The iterative L19 residual orthogonalization is doing exactly what it's designed for.
- **Baseline-14B (RunPod-required):** Expected to be ~2–4 modes / 10 based on the original webpage claim that near-duplicate rate is 0.33 baseline.
- **Why it matters:** Most direct demonstration of the anti-mode loop. Easy to visualize on the page ("Conceptial: 8 distinct ideas / Baseline: 3 distinct ideas").

---

## Task 4 — what's missing and the exact RunPod commands

### What needs RunPod

The four baseline-14B numbers, the EQ-Bench composite for both arms, the MAUVE for both arms, and the Pass@5 + quality-gate scoring (since the judge needs both arms' outputs to be run against the same rubric with the same prompts).

### Estimated wall-clock on an A100 80GB pod

| Phase | Time | Cost |
|---|---|---|
| Model load (Qwen2.5-14B-Instruct, fp16) — twice (once per arm) | 2 × 2 min | — |
| Creative-brief generation: 2 arms × 10 prompts × 5 outputs × ~22s each | ~37 min | A100 |
| Repeated-prompt stress test: 2 arms × 10 regenerations × ~22s each | ~7 min | A100 |
| EQ-Bench Creative writing (longer outputs ~1100 tokens): 2 arms × 10 prompts × ~75s each | ~25 min | A100 |
| OpenAI judging (quality gate, EQ-Bench) | ~10 min | ~$0.40 |
| **Total** | **~80 min** | **~$2.50** |

### Exact commands to run

```bash
# On the RunPod pod (after the usual setup in docs/runpod_14b_runbook.md):
cd /workspace/BrainstormLM
git pull && uv sync                                   # picks up the new eval modules

# Re-extract vectors if not already cached on this pod:
test -f data/steering_14b/hyper_creative_purified_vector_L19.pt \
  || uv run python scripts/extract_14b_vectors.py

# Generate both arms:
uv run python scripts/run_full_eval_2026_05_18.py --phase generate

# Compute all 8 metrics:
uv run python scripts/run_full_eval_2026_05_18.py --phase metrics

# SCP results back:
# (run from laptop)
scp -r root@<pod-ip>:/workspace/BrainstormLM/artifacts/eval_2026-05-18 \
    ~/Documents/BrainstormLM/artifacts/
```

The runner writes:
- `artifacts/eval_2026-05-18/{baseline-14b,conceptial-14b}/<prompt-id>.json` — raw outputs
- `artifacts/eval_2026-05-18/eq_bench/{baseline-14b,conceptial-14b}/<prompt-id>.json` — EQ-Bench outputs
- `artifacts/eval_2026-05-18/stress/{baseline-14b,conceptial-14b}.json` — K=10 stress test
- `artifacts/eval_2026-05-18/SUMMARY.json` — every metric, per-prompt + aggregate
- `artifacts/eval_2026-05-18/WEBPAGE_STATS.md` — overwrites this file with final paste-ready numbers

### What I still need from you

1. **RunPod pod IP + SSH key** (or paste the ssh command into the chat as `! ssh ...` and I'll handshake).
2. **Confirmation that ~$2.50 of pod time + ~$0.40 of OpenAI API spend is approved.**

After that the workflow is fully autonomous — the runner phases drive themselves, the metrics phase regenerates this file with locked-in numbers, and I commit the result for you to drop straight into `Corpus1Landing.tsx:37–40`.

---

## Methodology notes (locked, would not change between runs)

- **Sampling:** temperature 0.75–0.85 jittered by a golden-ratio sequence (matches `scripts/runpod_batch_14b.py`), top-p 1.0, do_sample=True, identical for both arms.
- **Conceptial stack:** v17 parametric preset, sweep-picked creative L17 α=1.5 (per `data/steering_14b/sweep/chosen_config.json` if present, else manifest L19 α=2.62), technical L24 α=0.86.
- **Baseline stack:** identical model, identical sampling temps, no steering, no anti-mode formatter, no SQLite memory store.
- **Quality gate:** `coherence ≥ 3 AND on_topic ≥ 4` on the 1–5 gpt-4o-mini rubric — the same rubric `scripts/score_outputs.py` already uses.
- **Pass@5:** for each of 10 prompts, generate N=5; per-prompt pass@5 = 1 if ≥1 of 5 passes the gate, else 0. Aggregate = mean across prompts.
- **Vendi / Near-duplicate / Mode-Collapse:** OpenAI `text-embedding-3-small` (already cached in `data/cache/`), L2-normalized, cosine kernel. Thresholds: near-dup ≥ 0.92 (existing `eval/diversity.py` default), mode-collapse cluster ≥ 0.85 (matches existing `_archetype_count` helper).
- **Self-BLEU-4:** nltk `sentence_bleu` with smoothing method 1.
- **MAUVE:** gpt2-large featurizer, max_text_length 256, device_id -1 (CPU) so the metric is reproducible off-pod.
- **EQ-Bench:** prompts 1–10 of v3, one seed_modifier per prompt (the first), 1100 max_new_tokens, gpt-4o-mini judge with the 20-criterion rubric, composite = mean across all positive-direction-flipped criteria.
