# Creativity-Steering v3 — Results (Qwen2.5-14B-Instruct)

_Run: `artifacts/qwen14b_run_v3` · compiled 2026-06-09 · updated 2026-06-11 (overnight replication)_

## TL;DR — steering beats the unsteered baseline (a first across v1/v2/v3)

After deriving a clean **quality-gated, framed contrast vector (118 positive / 120 negative)**,
a small-alpha activation-steering sweep found a genuine **beat-baseline** result:

> **L6 @ α=0.25** → quality-gated composite **+0.434**, **Δ +1.52 over baseline**, while keeping
> coherence essentially intact (judge Pragmaticality **3.92** vs baseline 4.38).

**10 of 15 judged cells beat baseline _and_ clear the coherence bar.** The key was the
injection strength: α=1 destroys coherence everywhere (Pragmaticality 1.0); **α=0.25 is the
sweet spot** where novelty rises while the text stays coherent.

> **Replicated (2026-06-11):** a clean overnight re-run with a full 3-judge ensemble reproduced
> the headline — **L6 @ α=0.25 Pragmaticality 3.93** (vs 3.92 prior), Δ composite **+1.44**. See
> the overnight section below.

---

## Update — overnight full-map + clean replication (2026-06-11)

A 7.0-hour A100 run (`night_v3.sh`) did two things the earlier sweeps couldn't: a **complete
48-layer × α{0.2,0.3,0.4} novelty map** (144 cells, judge-free) and a **judged frontier with a
full, un-degraded 3-judge ensemble** on the top-novelty layers.

### 1. The headline result replicates cleanly
The prior beat-baseline cell carried a caveat — 34 Claude-judge calls failed mid-run, so some
cells were scored by 2 judges, not 3. This run re-judged with all three judges intact:

| | Prior (2026-06-09) | This run (2026-06-11) |
|---|---|---|
| L6 @ α=0.25 Pragmaticality | 3.92 | **3.93** |
| Δ composite over baseline | +1.52 SD | **+1.44 SD** |

The near-identical Pragmaticality turns the result from a caveated pilot into a **replicated**
finding. Interpretable framing: novelty percentile **35 → 64**, **~90 % of baseline coherence
retained**, common-language win-rate **~85 %**.

### 2. Full 48-layer novelty landscape (144 cells)
- **118 / 144 (82 %) of steered cells beat baseline novelty.**
- Mean novelty by α: **0.2 → +0.06, 0.3 → +0.08, 0.4 → −0.08** (baseline −0.22). Relative to
  baseline, every α is net-positive on average (Δ +0.28 / +0.30 / +0.14); α=0.4 gives the smallest
  gain. Novelty only, judge-free; coherence was not scored in this map.
- **New high-novelty layers the targeted sweeps never tested:** L22, L27, L28, L32, L44, L45
  (mid-to-late band). Top raw novelty is **L32 @ α=0.4 (+0.771)**.

### 3. Judged frontier (top-6 novelty layers × fine α grid) — 7/10 beat baseline
| layer | α | Δ (SD) | novelty pct | Pragmaticality | note |
|---|---|---|---|---|---|
| L22 | 0.20 | **+1.54** | 35→67 | 2.90 | new layer; top composite but coherence cost |
| **L6** | **0.25** | **+1.44** | 35→64 | **3.93** | **cleanest coherent win — replicated** |
| L7 | 0.25 | +1.36 | 35→**72** | 3.77 | **most novelty while staying coherent** |
| L6 | 0.30 | +1.23 | — | 3.21 | |
| L7 | 0.20 | +1.07 | — | 4.28 | very coherent, modest novelty |
| L5 | 0.25 | +0.88 | — | 4.12 | |

Beyond α≈0.3 on these layers coherence collapses (L22@0.25 → Pragmaticality 1.77, L5@0.35 → 2.92),
re-confirming the α curve. **For a steer→refine pipeline, L7 @ α=0.25 is the pick** (highest
novelty among coherent cells); **for a single coherent win, L6 @ α=0.25.**

Artifacts: `patching/mean_response/S1_mean_novelty_results.json` (full map) and
`S2_mean_judged_results.json` (frontier) are committed alongside this report; the
`S1_chunk*_/S2_*_generations.jsonl` raw generations (2 870) stay local.

---

## Setup

| | |
|---|---|
| Model | Qwen/Qwen2.5-14B-Instruct (bf16) |
| Method | DiffMeans (CAA) over a framed creative-vs-neutral contrast |
| Selection | top/bottom 10% by **novelty**, one response per prompt, positives **quality-gated** |
| Metric | **quality-gated novelty**: score = novelty if judge Pragmaticality ≥ floor, else disqualified |
| Quality floor | derive 3.0 ("clearly creative") · sweep 2.0 ("coherent/refinable") |
| Capture position | `mean_response` (primary; v1-proven) |
| Contrast size | **118 positive / 120 negative** (of 2400 scored) |
| Judges | claude-sonnet-4-6 + gpt-4o + gemini-2.5-flash (Pragmaticality 1–5) |

---

## Result 1 — the derived vector is creativity-aligned

The 118/120 vector **passes the novelty gate**: at α=1 it raised the model-free Stage-1 novelty
on 36/48 layers (best L5 +0.81). (v1 also passed at +1.42; v2 failed, dead at +0.056.) So the
direction is real — the open question was always whether steering with it can beat baseline on a
quality-aware metric.

## Result 2 — α=1 is far too strong (coherence collapse)

The 48-layer scan at α=1 showed **every judged layer collapses to Pragmaticality 1.0** (fully
degenerate text) — baseline coherence is 4.36. Novelty is present but the text is broken:

| layer | novelty (Δ) | Pragmaticality | composite |
|---|---|---|---|
| L5 | +0.691 | 1.00 | -10.000 |
| L6 | +0.417 | 1.00 | -10.000 |
| L7 | +0.413 | 1.00 | -10.000 |
| L37 | +0.390 | 1.00 | -10.000 |
| L33 | +0.383 | 1.00 | -10.000 |
| L32 | +0.354 | 1.00 | -10.000 |
| L39 | +0.320 | 1.00 | -10.000 |
| L14 | +0.316 | 1.00 | -10.000 |
| L34 | +0.306 | 1.00 | -10.000 |
| L12 | +0.281 | 1.00 | -10.000 |
| L30 | +0.279 | 1.00 | -10.000 |
| L8 | +0.258 | 1.00 | -10.000 |

→ the blocker is the **alpha**, not the quality bar.

## Result 3 — small alpha finds the novelty↔coherence sweet spot ✅

Sweeping the high-novelty layers at **α ∈ {0.1, 0.25, 0.5}**
(baseline: novelty -0.328, Pragmaticality 4.38, composite -1.085):

| layer | α | novelty | Pragmaticality | composite | Δ vs base | clears 2.0 | |
|---|---|---|---|---|---|---|---|
| L6 | 0.25 | +0.506 | 3.92 | +0.434 | +1.520 | yes | **WIN** |
| L8 | 0.25 | +0.697 | 3.97 | +0.321 | +1.407 | yes | **WIN** |
| L7 | 0.25 | +0.710 | 3.61 | +0.316 | +1.401 | yes | **WIN** |
| L5 | 0.25 | +0.256 | 4.10 | -0.197 | +0.889 | yes | **WIN** |
| L32 | 0.25 | -0.033 | 4.32 | -0.440 | +0.646 | yes | **WIN** |
| L12 | 0.1 | +0.101 | 4.43 | -0.462 | +0.623 | yes | **WIN** |
| L33 | 0.25 | -0.111 | 4.27 | -0.642 | +0.444 | yes | **WIN** |
| L6 | 0.1 | -0.131 | 4.55 | -0.649 | +0.436 | yes | **WIN** |
| L7 | 0.1 | -0.124 | 4.33 | -0.785 | +0.300 | yes | **WIN** |
| L5 | 0.1 | -0.110 | 4.43 | -0.840 | +0.246 | yes | **WIN** |
| L12 | 0.25 | +0.586 | 2.53 | -1.681 | -0.595 | yes |  |
| L39 | 0.5 | +0.167 | 2.58 | -1.949 | -0.864 | yes |  |
| L37 | 0.5 | +0.212 | 2.30 | -3.862 | -2.776 | yes |  |
| L33 | 0.5 | +0.372 | 1.60 | -7.326 | -6.240 | no |  |
| L32 | 0.5 | +0.354 | 1.35 | -8.846 | -7.760 | no |  |

**10/15 judged cells beat baseline and clear the 2.0 coherence bar.**

### The alpha curve (the core finding)
| α | effect | Pragmaticality |
|---|---|---|
| 1.0 | degenerate everywhere | ~1.0 |
| 0.5 | breaking (some layers collapse) | 1.3–2.6 |
| **0.25** | **the win — novelty up, coherent** | **~3.9** |
| 0.1 | coherent but too weak (novelty barely moves) | 4.3–4.5 |

### Recommended configs
- **Best composite:** `mean_response`, **L6 @ α=0.25** (composite +0.43, Δ +1.52, prag 3.92)
- **Most novelty + coherent** (for a steer→refine pipeline): **L7 or L8 @ α=0.25** (novelty +0.70/+0.71, prag 3.6/4.0)

---

## Caveats
- **Judge degradation:** _(resolved 2026-06-11)_ the original run lost 34 Claude-judge calls to an
  API outage, so some cells were scored by 2 judges not 3. The overnight re-run re-judged with a
  full 3-judge ensemble and reproduced the headline (Pragmaticality 3.92 → 3.93), so the win is no
  longer caveated on this point.
- **Batch-relative scores:** the composite is z-scored within each sweep; compare Δ-vs-baseline
  within a run, not absolute composites across runs.
- `last_response_token` sweep not yet run (A/B pending).

## Context (v1 / v2)
- **v1:** framed contrast, mean_response, novelty-aligned (+1.42) but **never beat baseline** —
  blocked by a multiplicative novelty×quality metric and α≥1 breaking coherence.
- **v2:** unframed score_split, last_response_token — vector went **novelty-subtractive** (dead).
- **v3 (this run):** quality-gated framed contrast + **small alpha** → **first beat-baseline.**

## Next steps
1. ~~Clean re-run (cache-assisted) to lock exact numbers.~~ **Done (2026-06-11)** — headline replicated.
2. ~~Refine α ∈ {0.2, 0.3} on L6/L7/L8 for the precise frontier.~~ **Done** — full 48L × {0.2,0.3,0.4}
   map + fine-α judged frontier; α∈{0.2,0.3} give the largest mean novelty gain network-wide.
3. **Probe the newly-surfaced layers** — L22/L27/L32/L44 were never in the targeted sweeps; judge
   them at α∈{0.15,0.2} to see if any beats L6/L7 on the novelty↔coherence frontier.
4. `last_response_token` A/B (deferred from the overnight run for time; generations are persisted).
5. Validate the steer→Opus-refine pipeline on the L7 @ α=0.25 outputs (highest coherent novelty).
