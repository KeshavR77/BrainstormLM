# creativity_evaluators

A two-stage creativity evaluator that turns `(prompt, response)` into a scalar creativity
score. Built to **filter raw model outputs into a creative / not-creative contrast set** for
DiffMeans steering-vector derivation — but it works as a standalone creativity scorer too.

- **Stage 1 — model-free signals** (cheap, deterministic, no API): DSI, lexical/slop
  redundancy, length-normalized perplexity, and a quality reward model.
- **Stage 2 — LLM-as-judge ensemble** (expensive, more reliable): a 3-judge ensemble scored
  on a coarse 1–5 Torrance-style rubric. Run only on responses that clear a Stage-1 floor.
- **Composite**: fuses both stages into one scalar (`novelty × quality`), batch-z-scored so
  thresholds are relative within a run.

Grounded in: Chakrabarty et al. CHI 2024 (TTCW), Johnson et al. 2022 (DSI), Lu et al. ICLR
2025 (Creativity Index), Zhang et al. COLM 2025 (NoveltyBench), Saakyan et al. ICLR 2026,
Fein et al. 2025 (LitBench), Padmakumar et al. ICLR 2026, and EQ-Bench Creative Writing v3.

## Install

```bash
# core (pure-Python math only): numpy, scipy, pydantic
pip install -e .

# Stage-1 model scorers (BERT for DSI, GPT-2 for PPL, Skywork reward model)
pip install -e ".[stage1]"

# spaCy for POS-template overlap (optional; degrades gracefully if absent)
pip install -e ".[nlp]" && python -m spacy download en_core_web_sm

# Stage-2 judge API clients
pip install -e ".[judges]"

# everything + dev tooling
pip install -e ".[all,dev]"
```

**Design note.** Every heavy/model/API dependency is *optional* and *lazily imported*: the
pure cores (DSI math, slop overlap, z-scoring, ensemble aggregation, composite) and the whole
test suite run on numpy/scipy alone. Set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
`GOOGLE_API_KEY` for the Stage-2 judges.

## Quickstart

```python
from creativity_evaluators import evaluate_batch, EvalConfig, PromptCategory

records = [
    {"prompt": "Invent an unusual use for a paperclip.",
     "response": "A tiny lightning rod for a dollhouse, grounding static into the carpet.",
     "category": "creative"},
    {"prompt": "What is the capital of France?",
     "response": "The capital of France is Paris.",
     "category": "neutral"},
]
config = EvalConfig(enable_stage2=False)        # Stage 1 only — no API keys needed
results = evaluate_batch(records, config)       # composite is batch-relative, so score in batches
for r in results:
    print(r.category.value, round(r.creativity_score, 3), r.components["novelty"])
```

`evaluate(prompt, response, category, config)` scores a single record, but the composite
z-scores against the batch — a batch of one yields a degenerate score, so prefer
`evaluate_batch`.

## The composite formula

All `z(·)` are standardized against the **current batch**.

```
novelty           = z(DSI) + z(PPL) − z(lexical_redundancy)
novelty_combined  = mean(z(novelty), z(originality_s2))
quality_combined  = mean(z(reward), z(pragmaticality_s2))
creativity        = sqrt(max(0, novelty_combined) · max(0, quality_combined))
final             = 0.85 · creativity + 0.15 · z(elaboration_s2)
```

The geometric mean enforces *novelty AND quality*: jumbled gibberish is novel but low-quality,
so its `quality_combined` is small (or clipped to 0) and `creativity` collapses — this is the
`novelty × appropriateness` framing. Every raw and intermediate term is stored in
`EvalResult.components`, so you can re-tune the weights (`composite_creativity_weight`,
`composite_elaboration_weight`) **without re-running the judges**. Missing signals (too-short
DSI, Stage-1-only records) are imputed to the batch mean (z = 0) so they contribute neutrally.

## Tuning thresholds for the DiffMeans contrast set

1. Score a batch with `evaluate_batch`.
2. Run `scripts/calibrate_thresholds.py --gold <set>` to check Spearman vs. human ratings
   (acceptance bar **Spearman ≥ 0.5**) and print suggested quintile cut-points.
3. Assign classes by quintile: keep top-quintile `creative` responses as positives and
   bottom-quintile `neutral` responses as negatives.
4. **Length-match** the two classes with
   `creativity_evaluators.utils.length.length_matched_partition(creative_pool, neutral_pool)`
   before deriving DiffMeans — otherwise the steering vector encodes verbosity, not creativity.

The Stage-2 gating floor (`stage1_percentile_floor`, default 20) skips the expensive judges on
responses in the bottom Stage-1 quintile.

## Scripts

```bash
# batch-score a JSONL of {prompt, response[, category]} into EvalResults
python scripts/score_dataset.py --input raw.jsonl --output scored.jsonl --no-stage2

# correlate the composite against a hand-rated gold set
python scripts/calibrate_thresholds.py --gold tests/fixtures/sample_responses.json --no-stage2
```

`notebooks/01_validate_on_gold.ipynb` does the same validation interactively with a scatter plot.

## Tests

```bash
pytest                          # from this folder; runs on the minimal (numpy/scipy) install
```

## Bias-mitigation checklist (all reflected in code)

- [x] Judge inputs truncated to 4,000 chars (`judge.py`, length bias).
- [x] Explicit "do not reward length" instruction (`rubric.py`).
- [x] Explicit "do not reward flowery vocabulary" instruction (`rubric.py`).
- [x] 3-judge ensemble with mean aggregation (`ensemble.py`).
- [x] Judge-family exclusion (`ensemble.exclude_same_family`, gated by `generator_family`).
- [x] Coarse 1–5 Likert, not fine-grained binary (`rubric.py`).
- [x] Slop-list penalty in Stage 1 (`stage1/lexical.py` + `stage2/slop_list.py`).
- [x] Length-matched contrast partition (`utils/length.length_matched_partition`).

## What this does NOT do

No fine-tuned reward model, no prompt generation, no DiffMeans logic, no web UI — this module
only scores responses. The upstream generation and downstream steering-vector code live
elsewhere (`creativity_steering/`).
```
