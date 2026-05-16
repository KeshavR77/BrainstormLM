# Conceptial / Corpius 1 — locked demo configuration

This subpackage is the production home for the Conceptial diverse-ideation engine. It
combines representation engineering (purified L14 + L18 steering, v17 schedule) with three
prompt-level innovations to push local 3B Qwen2.5 to **diversity parity with documented cloud
14B Harmonic numbers** (Vendi 4.03 vs cloud's 4.02) on the logo-brief class.

## Locked stack

| Component | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` (MPS / fp16) | Local-runnable, fast iteration |
| Steering | v17 purified L14 (α=2.62→0 by step 168) + L18 (α=0.86, sigmoid mid=52) | Documented "winning trajectory" from `Corpius-1-Technical-Thesis.md`. **No** IOR-at-L28, ghost decode, or cognitive-route layer — all four broke grammar in V8.x |
| Sampling | `min_p=0.01`, `base_temp=0.8`, `temp-jitter=0.05` per sample | Wide tail without high-PPL salad |
| Diversity-1 | Cognitive anti-mode at PROMPT level | Frontier models do this internally; we do it explicitly across samples |
| Diversity-2 | Cross-session persistence via `IdeaMemoryStore` (SQLite) | Day-3 infrastructure repurposed for text-level memory |
| Diversity-3 | Few-shot exemplars (auto-extracted top-K from prior scored runs) | Quality lift +12% creativity, +13% specificity; anti-copy guardrail prevents verbatim |
| Brief routing | `--output-mode {image-prompt, strategy, generic}` | Visual prompts vs strategic prompts need different output instructions |

## Measured numbers (15 outputs per arm, scored by GPT-4o-mini)

| Brief class | Vendi | Composite quality | On-topic | Notes |
|---|---:|---:|---:|---|
| **Logo** ("creative logo for engineering club at Stanford") | **4.03** | 3.83 | 4.6 | Matches cloud 14B 4.02; 15 distinct visual archetypes |
| **GTM** ("distinct go-to-market strategies for diverse-AI startup") | 2.52 | **3.98** | **4.93** | 14/15 on-brief with all 4 sections (customer wedge, proof metric, pricing, major risk) |

Comparison points:
- Baseline (unsteered Qwen2.5-3B): Vendi 1.40, composite 3.60
- Cloud Corpius-1 Harmonic on Qwen2.5-14B (documented): Vendi 4.02

## Usage

From `BrainstormLM/`:

```bash
# Generate 5 logo concepts on a fresh project (image-prompt mode is default)
python3 scripts/smoke_cognitive_anti_mode.py \
    --prompt "Give me a creative logo for an engineering club at Stanford." \
    --n 5 --project-id "my-project" --reset-project \
    --exemplars-json artifacts/conceptial/exemplars_auto_logo.json

# Continue the same project with a second session (3-shot demo flow)
python3 scripts/smoke_cognitive_anti_mode.py \
    --prompt "Give me a creative logo for an engineering club at Stanford." \
    --n 5 --project-id "my-project" \
    --exemplars-json artifacts/conceptial/exemplars_auto_logo.json

# Strategy mode for non-visual briefs (GTM, R&D, etc.)
python3 scripts/smoke_cognitive_anti_mode.py \
    --prompt "Generate GTM strategies with customer wedge, proof metric, pricing, major risk." \
    --n 5 --output-mode strategy --project-id "my-gtm-project" --reset-project

# Score quality of any smoke artifact
python3 scripts/score_outputs.py --md artifacts/conceptial/ANTIMODE_SMOKE.md

# Auto-extract new top-scoring exemplars to feed back as future few-shot demos
python3 scripts/extract_exemplars.py \
    --scores artifacts/conceptial/*.scores.json \
    --top-k 3 --min-composite 4.0 \
    --out artifacts/conceptial/exemplars_auto_logo.json
```

## Module map

- `idea_store.py` — SQLite-backed per-project memory; zero non-stdlib non-tensor deps (numpy + torch only)
- `anti_mode.py` — prompt formatter, exemplar plumbing, brief-aware output instructions, concept-name extractor
- `engine/` — **(Day-2 consolidation)** the full steering / inference subsystem migrated from
  the legacy `shrooms-llm` repo. `engine/corpius.py`, `engine/pipeline.py`,
  `engine/steering_presets.py`, `engine/hooks.py`, `engine/config.py`, `engine/core/`,
  etc. All imports were renamed `shrooms_llm.* → brainstormlm.conceptial.engine.*`.
- `__init__.py` — re-exports the public API (`IdeaMemoryStore`, `format_anti_mode`, `format_pristine`, `extract_summary`, `OUTPUT_MODE_INSTRUCTIONS`)

The repo is now self-contained: steering vectors at `BrainstormLM/data/steering/`, the
locked Corpius 1 spec at `BrainstormLM/config/conceptial.json`, Vendi/Prism metrics at
`BrainstormLM/creativity_evaluator/` (slim copy — the LiveBench fork stayed in the
legacy archive). The `shrooms-llm` repo is now reference-only / outdated.

## What's NOT here (and why)

- **V8.x IOR-at-L28**: broke grammar in 4 distinct ways (Conditions A/B/C/D + verbatim Chinese-character bleed). Replaced with cognitive anti-mode at prompt level.
- **Ghost decode**: contributed to grammatical destabilization at α=0.5 unit-norm. Dropped.
- **Cognitive prompt rewrite (`format_cognitive_steering_prompt`)**: the V6-era rewrite combined with IOR caused additional drift. Replaced with the simpler instruction-aware prompt in `anti_mode.py`.
- **Manifold circuit breakers**: built for the V8.x failure modes that no longer occur on this stack. Kept on the FastAPI server side (shrooms-llm) for safety.
