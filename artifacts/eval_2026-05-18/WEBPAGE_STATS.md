# Webpage stats — refreshed 2026-05-18

Conceptial vs the same model (Qwen2.5-14B-Instruct) unsteered, on a repeated-prompt stress test (5 regenerations × 10 prompts).

## Headline tile values (paste into Corpus1Landing.tsx)

| Value | Label | Note |
|---|---|---|
| -85.7% | Fewer repeat answers | Near-duplicate rate 0.010 vs 0.070 baseline (cos ≥ 0.92) |
| +4.1% | More answer diversity | Vendi score 2.331 vs 2.240 baseline |
| +0.0% | More correct paths | Pass@5 1.000 vs 1.000 baseline (≥1 of 5 passes gate) |
| 100% | Quality gate pass | Coherence ≥ 3 AND on-topic ≥ 4 on every judged sample |

## New stats (additional credibility tiles)

| Value | Label | Note |
|---|---|---|
| +44.4% | Lower n-gram overlap | Self-BLEU-4 0.116 vs 0.080 baseline (Zhu et al. 2018) |
| 0.042 | Closer to human prose | MAUVE 0.042 vs 0.156 baseline (Pillutla et al. 2021) |
| 5.52 | EQ-Bench Creative v3 | composite 5.52 vs 5.08 baseline (10-prompt subset, gpt-4o-mini judge) |
| 10 / 10 | Distinct modes at K=10 | greedy cluster count at cos ≥ 0.85 vs baseline 8/10 |

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
