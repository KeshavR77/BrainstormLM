# Conceptial / Corpius 1 — locked demo configuration

This subpackage is the production home for the Conceptial diverse-ideation engine. It
combines representation engineering (purified L14 + L18 steering, v17 schedule) with three
prompt-level diversity mechanisms on local 3B Qwen2.5.

## Locked stack

| Component | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` (MPS / fp16) | Local-runnable, fast iteration |
| Steering | v17 purified L14 (α=2.62→0 by step 168) + L18 (α=0.86, sigmoid mid=52) | Schedule carried over from the legacy repo. **No** IOR-at-L28, ghost decode, or cognitive-route layer (dropped in the legacy repo) |
| Sampling | `min_p=0.01`, `base_temp=0.8`, `temp-jitter=0.05` per sample | Sampling defaults |
| Diversity-1 | Cognitive anti-mode at PROMPT level | Each sample sees the earlier samples as an exclusion list |
| Diversity-2 | Cross-session persistence via `IdeaMemoryStore` (SQLite) | Day-3 infrastructure repurposed for text-level memory |
| Diversity-3 | Few-shot exemplars (auto-extracted top-K from prior scored runs) | Anti-copy guardrail rejects verbatim copies of exemplars |
| Brief routing | `--output-mode {image-prompt, strategy, generic}` | Visual prompts vs strategic prompts need different output instructions |

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

## What's NOT here

V8.x IOR-at-L28, ghost decode, the V6-era cognitive prompt rewrite
(`format_cognitive_steering_prompt`), and manifold circuit breakers were tried in the legacy
`shrooms-llm` repo and dropped in favour of prompt-level anti-mode. The experiments behind those
decisions are not in this repo.
