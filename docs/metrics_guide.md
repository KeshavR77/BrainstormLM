# Metrics guide — direction & interpretation

A single legend for every diversity/quality metric in the repo, so a reader never mistakes a
regression for a win. **Always quote a metric with its `(model, prompt-set, N, embedder, method)`
context** — a bare number is not comparable across runs.

| Metric | Better | What it measures | Use as |
|---|---|---|---|
| **Vendi** | ↑ higher | Effective number of distinct modes in a set (embedding-based) | Headline diversity |
| **NoveltyBench-Distinct** | ↑ higher | Count of semantic equivalence classes in N outputs | Headline diversity (north star) |
| **Conditional-Vendi** | ↑ higher | Diversity with prompt-driven variance conditioned out | Headline diversity (genuine) |
| **Pragmaticality** | ↑ higher | Judge coherence / on-briefness (1–5) | Quality gate |
| **composite (z-scored)** | ↑ higher | Quality-gated novelty, batch-relative | Steering-sweep ranking |
| **near-duplicate rate** | ↓ lower | Fraction of output pairs above a cosine threshold | Diversity (repeat rate) |
| **Self-BLEU** | ↓ lower | N-gram overlap across outputs (lexical only) | **Canary only — never a target** |
| **MAUVE** | ↑ higher | Distributional closeness to a human reference | **Canary only** (a brainstormer diverging from "typical" prose is fine) |

## Known label traps (do not repeat)

- `artifacts/eval_2026-05-18/` (the older **v17 CAA** eval) has **Self-BLEU 0.116 vs 0.080** and
  **MAUVE 0.042 vs 0.156** for steered-vs-baseline. Both moved the **wrong** way (steered is
  *worse* on both). They are NOT wins — Self-BLEU/MAUVE are semantically blind canaries, and for a
  brainstorming model low MAUVE is expected. The real win in that eval is **near-dup −86%
  (0.010 vs 0.070)** and **Vendi +4.1% (2.331 vs 2.240)**.
- The current shipping steering is the **v3 DiffMeans small-alpha** line (`artifacts/qwen14b_run_v3/`),
  whose validation uses composite / Pragmaticality / novelty-percentile — see
  [steering_lineage.md](steering_lineage.md). Prefer those numbers over the v17 eval.
- Any frontend/mocks metric must trace to an artifact under `artifacts/`. Do not display a number
  that exists in no eval.
