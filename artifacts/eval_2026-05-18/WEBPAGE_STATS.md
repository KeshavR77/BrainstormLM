# Webpage stats — refreshed 2026-05-18

Conceptial vs the same model (Qwen2.5-14B-Instruct) unsteered, on a repeated-prompt stress test (5 regenerations × 10 prompts).

## Results (from `SUMMARY.json`)

Small sample: 10 prompts × 5 regenerations, one run, no significance tests. Treat as directional.

| Metric | Conceptial | Baseline | Direction |
|---|---|---|---|
| Near-duplicate rate (cos ≥ 0.92) | 0.010 | 0.070 | better (1 vs 7 near-dup pairs) |
| Vendi score | 2.331 | 2.240 | better (+4.1%) |
| Pass@5 (≥1 of 5 passes gate) | 1.000 | 1.000 | no change |
| Quality gate pass rate | 1.000 | 1.000 | no change |
| Self-BLEU-4 (lower = less n-gram overlap) | 0.116 | 0.080 | **worse** (more overlap) |
| MAUVE vs human reference (higher = closer) | 0.042 | 0.156 | **worse** (further from human prose) |
| EQ-Bench Creative v3 composite | 5.52 | 5.08 | better (10-prompt subset, single judge) |
| Distinct modes at K=10 (one prompt) | 10 / 10 | 8 / 10 | better |

## Methodology

- **Model**: Qwen/Qwen2.5-14B-Instruct, fp16, A100 80GB.
- **Conceptial stack**: CAA L19 (creative, α₀=2.62) + L24 (technical, α₀=0.86), v17 exp+sigmoid schedule, iterative L19 residual orthogonalization, prompt-level anti-mode. **(Note: this eval pre-dates the 2026-05-18 sweep that selected L17/α₀=1.5 as the production config. These numbers reflect the old L19 operating point.)**
- **Baseline**: Same model, no steering, no anti-mode, identical sampling temps (0.75–0.85 jitter).
- **Prompt suite**: `config/benchmark_creative_briefs.json` (10 creative-operations briefs).
- **N per prompt**: 5 regenerations.
- **Repeated-prompt stress K**: 10 regenerations of `logo-stanford-engineering`.
- **Vendi / Near-dup**: OpenAI text-embedding-3-small, cosine kernel, Friedman & Dieng 2022.
- **Quality gate / Pass@5**: gpt-4o-mini, coherence ≥ 3 AND on_topic ≥ 4 (1–5 scale).
- **Self-BLEU-4**: nltk sentence_bleu, smoothing method 1 (Zhu et al. 2018).
- **MAUVE**: gpt2-large featurizer, vs `data/mauve_reference/references.jsonl` human samples.
- **EQ-Bench Creative**: 10-prompt subset of v3, gpt-4o-mini judge, 11 positive + 9 negative criteria, composite mean.
- **Mode-Collapse@K**: greedy cosine clustering at threshold 0.85.
