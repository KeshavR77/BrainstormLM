# creativity_steering

Derive, analyze, and activation-patch **DiffMeans creativity steering vectors at every
layer** of a Qwen model — a clean, standalone re-implementation of the methodology in
*Mechanistic Interpretability of Conformity in LLMs* (`research/`), retargeted from
conformity to **creativity**.

The whole pipeline runs offline on a deterministic **mock backend** (no model download).
Switching to the real **Qwen2.5-14B-Instruct** on a RunPod GPU is a one-line config change
(`backend`, `model_id`) — no code changes.

---

## Why this exists

The current Conceptial pipeline derives its steering vector from a *prompt-framing*
contrast (creative system prompt vs generic) at the prompt's last token, and never checks
whether the output was actually creative. This project fixes the data DiffMeans is applied
to: we generate **responses**, keep only the ones a creativity **evaluator verifies**
(confidently-creative vs confidently-not), and DiffMeans those — at every layer.

## The three phases

### 1. Vector derivation (`derivation/`)
1. **Generate** responses to a creativity-invoking framing and a neutral framing over a
   shared set of topics, caching each response's per-layer residual activations
   (`generate.py`). The contrast is driven by **decoding as well as framing**: each pool has
   its own `SamplingPreset` (`config.py`). The **creative** pool samples at high temperature
   (1.2) with a 0.95 nucleus, top-k off, a 0.02 min-p coherence floor, and a mild 1.1
   repetition penalty; the **neutral** pool decodes **greedily** (the most standard, expected
   output). A knob left at its no-op value is not forwarded, so the model's own generation
   default applies (`build_generate_kwargs`). Default capture position is the **last generated
   token**; `mean_response` and `assistant_start` are cached too for comparison.

   *Note:* greedy is deterministic, so on the real model the neutral pool yields one response
   per topic regardless of `samples_per_prompt` — size that pool via the topic count (and mind
   `min_per_class`), not by sampling more.
2. **Filter** to an evaluator-verified subset (`filter.py`): a creative-framed response is a
   *positive* only if its creativity score ≥ `creative_min`; a neutral-framed response is a
   *negative* only if ≤ `neutral_max`. The ambiguous middle is dropped, so the two pools are
   cleanly separated. A `min_per_class` guard (default 100, the paper's N) prevents deriving
   on too little data.
3. **DiffMeans at every layer** (`diffmeans.py`): `d_L = mean(positive) − mean(negative)`.

### 2. Vector analysis (`analysis/`)
- **Norm** (`norm.py`) — which layers have the largest difference vector.
- **Cosine** (`cosine.py`) — the per-layer top-k neighbour cosine `s_L` (paper's Fig 5) to
  find the mutually-aligned band, plus the layer pairs whose directions are *furthest apart*.
- **Activation oracle** (`oracle.py`) — probe each layer's vector for the trait of
  creativity. `StubOracle` runs offline (norm-percentile placeholder); `LatentQAOracle` is a
  documented slot to wire the real LatentQA adapter on a GPU.
- **Report** (`report.py`) fuses all three into a ranked list of **candidate layers** — a
  cheap, generation-free pre-filter for which layers to sweep first in Experiment 1. The
  layers carried into Experiment 2 come from the sweep itself (see below), not this analysis.

### 3. Activation patching (`patching/`)
- **Experiment 1 — per-layer alpha sweep** (`sweep.py`): for every layer `L` and every
  `alpha` (both signs), inject `alpha · d_L` at layer `L` (matched extract-and-inject) and
  score creativity. Localises where/which-sign has the largest effect.
- **Experiment 2 — multi-layer** (`multilayer.py`), run on the **promising layers Experiment 1
  surfaced** — the top layers by behavioural creativity in `sweep_results.json`
  (`select_promising_layers`; judged cells first, then by Stage-1 score), not the cheaper
  vector analysis:
  - `all_layers_single_vector`: inject one chosen vector at *every* layer with a small alpha.
  - `multi_vector_combo`: inject several layer-specific vectors at their own layers at once.

Every patched generation is scored by a **creativity scorer** (`scoring.py`) behind the
swappable `CreativityScorer` interface:
- `"real"` / `"embedding"` — a self-contained embedding + lexical scorer (prompt distance +
  distinct-2 + low repetition); needs the `embedding` extra (sentence-transformers). The
  former vendored `creativity_evaluator` wrapper has been retired (parked at the repo root as
  `old_creativity_evaluators/`); the richer two-stage signal now lives in the sibling
  `creativity_evaluators` package — see *derivation filtering* below.
- `"heuristic"` — dependency-free lexical proxy; the default for offline/mock runs and the
  test suite.

**Derivation filtering** can optionally use the full two-stage `creativity_evaluators` stack
(Stage 1 model-free signals — DSI etc. — plus a Stage 2 LLM-as-judge ensemble) instead of the
per-item scorer: set `FilterConfig.method = "evaluator_stack"`. It scores every generated
response, keeps the highest-scoring creative-framed and lowest-scoring neutral-framed
responses (by quantile), and feeds only those to DiffMeans. See `derivation/evaluator_filter.py`.

## Layer convention

"Layer `L`" is the **output of decoder block `L`** — `hidden_states[L+1]` at capture time,
and the residual a forward hook on `model.model.layers[L]` modifies at injection time. So
extraction and injection hit the same residual location (fixing the off-by-one in the
repo's current `extract_14b_vectors`, which reads `hidden_states[L]` but injects at block L).

## Quickstart

```bash
# Offline, no downloads — full pipeline on the mock backend:
python -m creativity_steering.cli demo --out artifacts/demo

# Or step by step with a config:
python -m creativity_steering.cli derive  --config configs/local_mock.json --out artifacts/run
python -m creativity_steering.cli analyze --run artifacts/run
python -m creativity_steering.cli sweep   --run artifacts/run --config configs/local_mock.json
python -m creativity_steering.cli multi   --run artifacts/run --config configs/local_mock.json \
                                          --source-layer 12 --layers 9,12,15
```

## Running the real model

`configs/qwen_0_5b_local.json` runs the real (small) Qwen2.5-0.5B locally; switch to
`configs/runpod_qwen14b.json` on a RunPod A100/H100 for the 14B target. Both set
`backend="hf"`; the only differences from the mock config are `backend`, `model_id`,
`device`, and `scorer="real"`. Install the extras with `pip install -e ".[hf,real]"`
(`hf` = the Qwen backend; `real` = the full vendored creativity evaluator).

## Tests

```bash
pytest            # from this directory (uses pythonpath=src)
```

The mock backend carries a *planted* per-layer creativity direction (peaked at a chosen
layer, correlated across a band), so the tests assert real end-to-end behaviour: DiffMeans
recovers the planted direction, the norm/cosine/oracle analysis localises the planted layer
and band, and the patching sweep recovers the planted layer/sign as the winner. The HF
backend is not unit-tested (it needs a model download); exercise it with the `hf` configs.

## Status / TODO

- [x] Full scaffolding: derivation, analysis, patching, CLI, configs, mock + HF backends.
- [ ] Wire the real `LatentQAOracle` adapter on a GPU pod (documented slot in `oracle.py`).
- [ ] Run on Qwen2.5-14B on RunPod and re-tune thresholds/alphas against real evaluator scores.
- [x] Verifiable two-stage filtering: `FilterConfig.method = "evaluator_stack"` runs the
      sibling `creativity_evaluators` stack (DSI + LLM-as-judge ensemble) over the generated
      responses and keeps only the top creative / bottom neutral examples for DiffMeans. The
      Stage-2 judge ensemble needs Anthropic/OpenAI/Google keys.
