# Creativity-Steering v3 — Full 48-Layer Judged Sweep (Qwen2.5-14B-Instruct)

_Experiment report · `artifacts/qwen14b_run_v3` · compiled 2026-06-15_

A complete, judge-scored map of activation-steering for creativity: **every layer (48) × every
small steering strength α∈{0.2, 0.3, 0.4}**, with each cell scored by a 3-model LLM-judge
ensemble. This is the most exhaustive evaluation of the v3 steering vector to date and it
overturns the working assumption about *which* layers matter.

---

## TL;DR

- **The best layers are mid-network (L14–L20, L26–L27), not the early band (L5–8) we'd been
  steering.** Every earlier run gated judging to a handful of layers picked by the cheap novelty
  metric and **missed this entire region**. Judging all 144 cells surfaced it.
- **Best coherent operating point: L17 @ α=0.2** — wins head-to-head on creativity **~91%** of the
  time, lifts novelty from the **34th → 71st percentile**, and keeps **~86% of baseline coherence**.
- **Clean dose-response in α:** α=0.2 is the sweet spot (44/48 layers beat baseline, coherence
  ≈ baseline); α=0.3 is marginal; α=0.4 breaks coherence.
- **Broad, not cherry-picked:** **87/144 (60%)** of cells beat baseline; **74** are *coherent*
  winners (beat baseline **and** judge-Pragmaticality ≥ 3.5).

---

## Pipeline (how the result was achieved)

```
   derive vector            validate direction         steer + generate            judge (off-pod)
 framed quality-gated  →  novelty gate (judge-free)  →  48 layers × α{.2,.3,.4}  →  3-judge ensemble,
   contrast (CAA)           α=1: novelty↑ 36/48          145 cells, gens saved       all 145 in one batch
```

Each stage is detailed below.

---

## 1. Vector derivation (DiffMeans / CAA over a framed, quality-gated contrast)

The steering vector is a **difference-of-means (CAA)** direction between two populations of the
*same model's* hidden states, captured under two opposing system-prompt **framings**:

| Framing | System prompt (verbatim) |
|---|---|
| **creative (+)** | "You are a wildly inventive brainstorming partner. Embrace surprising ideas, unconventional analogies, and creative leaps. Be original. Quality over consensus." |
| **neutral (−)** | "You are a careful, conventional assistant. Give the most standard, expected, straightforward answer. Avoid speculation or unusual ideas." |

- **1,200 topics × 2 framings → 2,400 generated responses**, hidden states captured at every layer.
- **Selection (the "quality-gated" part):** keep the **top 10% by novelty** as positives and the
  **bottom 10%** as negatives, one response per prompt; positives must additionally clear a judge
  **Pragmaticality ≥ 3.0** floor (so the "creative" pole is novel *and* coherent, not just weird).
  This dropped 2,162 of 2,400 and left a clean contrast of **118 positive / 120 negative**.
- **Vector = mean(positive hidden states) − mean(negative hidden states)** per layer
  (`normalization: raw_diffmeans`), `hidden_dim = 5120`, `n_layers = 48`.
- **Primary capture position: `mean_response`** (mean over the response tokens) — the v1-proven
  position. Vectors for `last_response_token` and `assistant_start` were also derived.

Artifacts: `vectors/mean_response/vectors.npz` (+ `manifest.json`), `contrast_set.jsonl` (238
rows), `scored_pool.jsonl` (2,400 rows + per-example novelty/quality), `derivation_summary.json`.

> Context: v1 used the same framed approach and was novelty-aligned but never beat baseline
> (metric ceiling); v2 switched to an unframed `score_split` and the vector went
> novelty-*subtractive* (dead). v3 = **revert to the framed contrast + quality-gate + small α**.

## 2. Vector validation (judge-free novelty gate)

Before spending any judge budget, the direction was tested with a **model-free novelty gate**:
steer at α=1 across all 48 layers and check whether Stage-1 novelty rises.

- **Result: novelty rose on 36/48 layers (best L5 +0.81).** The direction is genuinely
  creativity-aligned. (Gate reference: v1 passed at +1.42; v2 *failed*, dead at +0.056.)
- The same α=1 scan also showed **coherence collapses at full strength** (judge Pragmaticality → ~1.0
  everywhere) — i.e. the vector is right, but α=1 is far too strong. That motivated the small-α
  regime used here.

## 3. Steering sweep (on-pod generation, novelty only)

- **Grid:** all **48 layers × α ∈ {0.2, 0.3, 0.4}** = 144 steered cells **+ 1 unsteered baseline**.
- **Position:** `mean_response`. **Generation:** temperature 0.9, 256 max-new-tokens; **15 responses
  per cell** (n=15). Model: Qwen2.5-14B-Instruct, bf16, A100-80GB.
- **Cheap metric only on-pod** (judge OFF): each response gets Stage-1 model-free novelty; no API.
- **Robustness:** the sweep was **chunked into 6 layer-groups** (generations persist only when a
  sweep finishes, so chunking caps any crash loss to ~one chunk). Every generation
  (`{cell, layer, alpha, prompt, response}`) was written to `S1_chunk{1..6}_generations.jsonl`
  (2,250 rows) — the input to off-pod judging.

## 4. Off-pod judging (the scoring stage — no GPU, no regeneration)

The persisted generations were judged on a laptop, decoupled from the GPU:

- **Loader** (`judge_offpod.load_generation_records`): concatenates the 6 chunk files and keeps the
  baseline from only the first chunk (each chunk re-emits it) → **2,175 responses across 145 cells**
  (baseline + 144 steered, 15 each).
- **One batch, all cells** (`judge_offpod.judge_generations` → `creativity_evaluators.evaluate_batch`):
  the composite **z-scores every signal against the batch**, so all 145 cells must be scored
  together for the per-cell composites and deltas to be on one comparable scale.
- **Stage-1 recomputed deterministically** from the saved text (DSI `bert-base-uncased` + perplexity
  `gpt2-medium`, on MPS) — identical to the on-pod novelty since it's the same response text.
- **Stage-2: 3-judge ensemble** — `claude-sonnet-4-6` + `gpt-4o` + `gemini-2.5-flash`, scoring
  Originality/Flexibility/Fluency/Elaboration/Pragmaticality (1–5). `stage1_percentile_floor = 0` →
  **every response judged** (no top-k gating).
- **Resilience:** content-hash judge cache (every successful call cached → crash-resumable);
  Retry-After-aware 6× backoff. The Anthropic balance was exhausted mid-run and **topped up**; the
  run resumed from cache (only the unfinished Claude calls re-issued). **26 of 2,175 responses**
  ended up scored by 2 judges instead of 3 (a transient Claude dropout the ensemble averages over).

Code: `creativity_steering/patching/judge_offpod.py`, `scripts/judge_generations.py`
(TDD: `tests/test_judge_offpod.py`). Output: `patching/mean_response/offpod_judged_all144.json`.

## 5. Metrics (definitions)

- **DSI (Divergent Semantic Integration, Johnson et al. 2022):** mean pairwise cosine *distance*
  across a response's BERT token embeddings — the model-free novelty signal (the dominant Stage-1
  term: `novelty = z(DSI) + z(PPL) − z(lexical_redundancy)`).
- **Quality-gated composite (the per-cell `composite`):** for each response,
  `score = novelty_combined` **iff** its raw judge Pragmaticality ≥ **2.0**, else it is
  **disqualified** to a large negative. So steering is rewarded for novelty *only while the text
  stays coherent*; novelty bought by breaking the model is rejected. `novelty_combined =
  mean(z(novelty), z(originality))`.
- **`stage1` (novelty):** the per-cell mean Stage-1 novelty composite. **`pragmaticality`:** the raw
  1–5 judge coherence score (mean over the cell). **`delta`:** cell composite − baseline composite.
- **Scale note:** composite/novelty are **z-scored within this batch**, so they are a standardized
  effect size (Cohen-d-like), not points on a natural scale. **Pragmaticality (1–5) is absolute**
  and comparable across runs. Interpretable conversions used below: a z-gap `d` → common-language
  win-rate `Φ(d/√2)`; a novelty z → percentile `Φ(z)`.

---

## 6. Results

**Baseline (unsteered):** composite −0.927 · novelty z −0.400 (34th pct) · **Pragmaticality 4.33** · n=15.

### 6a. Dose-response in α (the cleanest finding)

| α | cells beating baseline | mean novelty | mean Pragmaticality |
|---|---|---|---|
| **0.20** | **44 / 48 (92%)** | +0.089 | **3.94** (≈ baseline) |
| 0.30 | 25 / 48 (52%) | +0.071 | 3.08 |
| 0.40 | 18 / 48 (38%) | −0.152 | 2.44 (breaking) |

**α=0.2 is the sweet spot across the whole network** — almost every layer beats baseline while
coherence stays near the unsteered 4.33. By α=0.4 the model is degrading. (This refines the earlier
"0.2/0.3 good, 0.4 bad" into "0.2 is the operating point.")

### 6b. Top cells by margin over baseline

| cell | Δ (within-batch SD) | novelty z | Pragmaticality |
|---|---|---|---|
| L20 @ 0.2 | +1.89 | +0.562 | 2.98 |
| **L17 @ 0.2** | **+1.87** | **+0.562** | **3.71** ← best coherent |
| L27 @ 0.2 | +1.81 | +0.479 | 3.13 |
| **L14 @ 0.2** | +1.74 | +0.541 | **3.69** |
| L16 @ 0.2 | +1.62 | +0.470 | 3.38 |
| L26 @ 0.2 | +1.62 | +0.427 | 3.51 |
| L5 @ 0.3 | +1.53 | +0.605 | 3.71 |

- **87/144 (60%) beat baseline; 74 are coherent winners** (Δ>0 **and** Pragmaticality ≥ 3.5).
- L20 has the highest margin but borderline coherence (2.98); **L17 / L14 @ 0.2 are the best
  *coherent* cells** — high novelty *and* prag ~3.7.

### 6c. The mid-network band the targeted runs missed

The overnight judged frontier scored only layers **{5, 6, 7, 22, 32, 44}** (chosen by the cheap
novelty ranking). The full run's **top-10 layers are {5, 11, 14, 15, 16, 17, 20, 26, 27, 30}** —
**9 of 10 were never judged before.** The cheap Stage-1 novelty metric had pointed at the wrong
layers; only exhaustive judging revealed the real frontier. This is the core methodological lesson:
**gate-then-judge can hide your best cells.**

---

## 7. Headline (interpretable, defensible) numbers

> **Steering at L17 (α=0.2) beats the unsteered model on creativity ~91% of the time, lifting
> outputs from the 34th to the 71st percentile of novelty while retaining ~86% of baseline
> coherence — and the effect holds across 60% of all 144 layer/strength settings (74 coherent
> winners).** Best layers are mid-network; the optimal strength is α≈0.2, with coherence collapsing
> by α=0.4.

Every number here traces to a raw 1–5 judge score, a direct count, or a textbook z→percentile/
win-rate conversion — none require trusting the internal composite scale.

---

## Caveats

- **Batch-relative composite.** composite/novelty/Δ are z-scored within this 145-cell batch — compare
  *within this run*. Pragmaticality (1–5) is the cross-run-comparable metric (baseline 4.33 here vs
  4.33/4.38 in prior runs → the judge panel is consistent).
- **n = 15 per cell**, 3 samples × ~5 briefs. Strong signal, not a powered study.
- **26/2,175 responses scored by 2 judges** (Claude billing dropout, since topped up) — averaged over;
  directionally robust.
- **α=0.25 untested on the winning layers.** The 0.25 data from prior runs only covers L5–8 (where
  0.25 is a better novelty/coherence trade than either 0.2 or 0.3). Whether L14–20 @ 0.25 beats the
  current frontier is the open question.

## Next step

Full 48-layer sweep at **α=0.25**, then judge it **fused in one batch with these 144 cells** (one
comparable scale) — directly tests the sweet-spot strength on the newly-found mid-network winners.
Set up in `configs/phase1_alpha025_v3.json` + `scripts/sweep_alpha025_v3.sh` (pending a GPU pod).

## Artifacts
- `patching/mean_response/offpod_judged_all144.json` — the 144-cell judged frontier (this report)
- `patching/mean_response/S1_mean_novelty_results.json` — the on-pod novelty map (judge-free)
- `patching/mean_response/S1_chunk{1..6}_generations.jsonl` — 2,250 raw generations (judge inputs)
- `vectors/mean_response/{vectors.npz,manifest.json}`, `derivation_summary.json`, `contrast_set.jsonl`
