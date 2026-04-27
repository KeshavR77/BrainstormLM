# Plan 04 — RLHF Training (DPO + LoRA)

_Parent plan: `docs/plans/master-plan.md` (Phase 2, Sub-plan 2.1)._
_Builds on: `plan-03-reasoningbank.md` (complete as of commit `af1aaea`)._
_Authored: 2026-04-25._

## Context

Phase 1 is closed. The synthetic feedback loop (`gpt-4o-mini` prompter +
Qwen2.5-7B-Instruct-Turbo on Together + `gpt-4o-mini` evaluator) produces scored
trajectories at `data/trajectories.jsonl`, and ReasoningBank retrieval injects
abstracted lessons into the system prompt at inference. We have a held-out
50-prompt eval and a paired-bootstrap harness that already shows memory > base.

Plan-04 turns those scored trajectories — and on-demand sibling samples — into a
LoRA adapter trained with DPO over the same Qwen2.5-7B base. After this plan
lands we satisfy **Phase 2 Exit Criterion 2**: trained-with-retrieval beats
plan-03's retrieval-only row on mean `creativity + logic`, and we ship the full
ablation table (base, base+mem, trained, trained+mem).

Two structural realities the master plan didn't fully reconcile, that plan-04
must:

1. **Inference today is API-only.** `TogetherGenerator` (`model/base.py`) hits
   the Together chat API; there is no local HF runtime. To *serve* a LoRA we
   trained ourselves, we need a second `Generator` implementation that loads
   `Qwen2.5-7B-Instruct` from HuggingFace and applies a PEFT adapter. The
   master plan's footnote ("vLLM-on-Unicorn at Phase 3") is about the steered
   generator backend; plan-04 introduces a transformers-based serving path
   alongside Together, gated by `--adapter`. Phase 3 can later swap that
   transformers path out for vLLM without touching training code.
2. **Training never runs locally on the dev box.** Per the master plan's
   footnote on avoiding "a day of CUDA/SSH setup," and to keep the dev loop
   API-driven, the DPO job runs on **Modal** (rented A100). Local Unicorn
   stays reserved for plan-05 steering work.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Pair-construction rule | **Best-vs-worst per prompt with a minimum score-gap of 2** on `creativity + logic` (range 2–10) | Top-vs-bottom is the lowest-knob option and produces the cleanest signal at MVP scale. Minimum gap filters out near-ties where the evaluator is noisy. Configurable via `--min-gap`. |
| Siblings per prompt (N) | **6** (between the master plan's 4 and 8) | At 4 siblings ~25% of prompts produce no pair after the gap filter; 8 doubles cost. 6 lands ~80% pair yield empirically (verified by short pilot before full run). |
| Sibling temperature schedule | **`[0.7, 0.9, 1.0, 1.1, 1.2, 1.4]`** | Spans low-variance to high-variance regimes so the evaluator has real spread to pick from. Salts cache per (prompt, temperature). |
| Source of training prompts | **Re-sample from `prompter.generate_prompt` over `DOMAINS`**, NOT reuse `data/trajectories.jsonl` prompts | Trajectory prompts already have one response each; we'd need to draw new siblings anyway, and reusing prompts risks overfitting the eval. Held-out prompts (`HELD_OUT`) are NEVER sampled. |
| Pair count target | **1.5k–2k preference pairs** | Master plan's "1k+" lower bound, with headroom. Empirically the smallest pair count where TRL DPO produces a measurable shift on a 7B base with rank-16 LoRA. |
| Base model for training | **`Qwen/Qwen2.5-7B-Instruct`** (HuggingFace) | Same base as the Together-served `Qwen/Qwen2.5-7B-Instruct-Turbo` (Turbo = Together's quantized serving variant; same weights / chat template). Adapter trained on HF base transfers cleanly. |
| LoRA config | **rank=16, alpha=32, dropout=0.05, target_modules=`["q_proj","k_proj","v_proj","o_proj"]`** | Standard "safe" attention-only LoRA for 7B; ~10M trainable params; fits comfortably on a single 80GB A100 with bf16 + gradient checkpointing. Higher rank deferred to plan-04 iteration if signal is weak. |
| Training framework | **`trl.DPOTrainer` + `peft.LoraConfig` + `transformers`** | TRL is the canonical DPO implementation; auto-handles ref-model logprobs, β scheduling, paired-batch loss. No custom trainer code. |
| DPO β | **0.1** (TRL default) | Standard starting point. Open item if the trained model regresses to base. |
| Training compute | **Modal · single A100-80GB · bf16 · ~2h budget · cap $25** | On-demand, Python-native (no SSH/CUDA setup), pay-per-second. RunPod is the fallback if Modal credits run out. |
| Trained-adapter inference path | **New `HFGenerator`** in `model/hf.py` loading `Qwen2.5-7B-Instruct` + adapter via `PeftModel.from_pretrained` | Together does not load arbitrary local adapters without going through their Custom Models flow; transformers + PEFT is the simpler bridge. CLI flag `--adapter <path>` switches the generator from Together to HF. |
| Adapter format on disk | **PEFT-saved directory** at `data/adapters/<run_id>/` (no merging into base weights) | Keeps adapter swappable at inference for clean ablations; merging is a one-liner if Phase 3/4 need it. |
| Cost guardrails | **`--max-calls` and `--max-spend-usd`** on both new scripts; a separate `--max-modal-usd` on `train_dpo.py` (training cost is non-API and must be tracked separately from prompter/evaluator cost) | Mirrors `run_feedback_loop.py`. Aborts pre-flight when the estimate exceeds the cap, and mid-run when the running tally crosses it. |

## Deliverables

Files to **create**:

```
src/brainstormlm/training/__init__.py
src/brainstormlm/training/preferences.py    # Pair schema, mining, JSONL persistence
src/brainstormlm/training/dpo.py            # TRL DPOTrainer + PEFT LoraConfig wrapper
src/brainstormlm/model/hf.py                # HFGenerator: HF base + optional PEFT adapter
src/brainstormlm/model/factory.py           # make_generator(adapter_path) -> Generator
src/brainstormlm/eval/harness.py            # 4-row ablation table over HELD_OUT
src/brainstormlm/eval/stats.py              # Paired-bootstrap helper (lifted from eval_memory.py)
scripts/sample_siblings.py                  # Generate N siblings/prompt → score → emit pairs
scripts/train_dpo.py                        # Modal entrypoint for DPO training
scripts/eval_ablation.py                    # Run the 4-arm ablation; print + persist results
tests/test_preferences_pairs.py             # Pure-logic tests for pair construction
tests/test_preferences_io.py                # JSONL round-trip
tests/test_dpo_config.py                    # LoRA + DPO config builder (no training)
tests/test_hf_generator.py                  # Generator-protocol conformance, adapter toggle
tests/test_eval_harness.py                  # 4-row plumbing with mocked generators/scorer
```

Files to **extend**:

```
src/brainstormlm/config.py                  # add hf_base_model, adapters_dir, default_adapter_path, modal_*
src/brainstormlm/cli.py                     # add --adapter / --no-adapter; switch to HFGenerator when adapter set
src/brainstormlm/model/__init__.py          # re-export HFGenerator alongside TogetherGenerator
src/brainstormlm/feedback/trajectories.py   # add adapter_path: str | None = None (additive)
src/brainstormlm/feedback/pricing.py        # add estimate_sibling_run_cost; add Modal $/hr line item (informational)
scripts/run_feedback_loop.py                # add --adapter flag; route through make_generator
scripts/eval_memory.py                      # import bootstrap helper from eval/stats.py
pyproject.toml                              # add transformers, torch, peft, trl, accelerate, datasets, modal
```

No changes to `feedback/evaluator.py`, `feedback/prompter.py`, or `memory/*` —
they're consumed unchanged.

## Preference Pair Schema (v1)

```python
# src/brainstormlm/training/preferences.py
class PreferencePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str                          # sha256(prompt + chosen + rejected)
    timestamp: str                   # ISO 8601 UTC
    domain: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float              # creativity + logic, 2..10
    rejected_score: float
    score_gap: float                 # chosen_score - rejected_score
    chosen_temperature: float
    rejected_temperature: float
    generator_model: str             # base generator id (e.g. Qwen/Qwen2.5-7B-Instruct-Turbo)
    evaluator_model: str
    rubric_version: int
    source: Literal["sibling_sampling"] = "sibling_sampling"
    sibling_run_id: str              # groups pairs from one sample_siblings.py invocation
```

Persisted as append-only JSONL at `data/preferences.jsonl`. Same discipline as
`Trajectory`: schema changes are additive only; bump `schema_version` for
incompatible changes.

`PreferencePair.id` is a deterministic content hash so re-running sibling
sampling on the same prompt with cached responses produces no duplicate rows.

## Sibling Sampling & Pair Construction Contract

`training/preferences.py::sample_and_score_siblings`
- Inputs: `(prompt: str, domain: str, n: int, temperatures: list[float],
  generator: Generator, run_id: str)`.
- For each of `n` temperatures (cycle the list if `n > len(temperatures)`),
  call `generator.generate_with_meta(prompt, system=None, temperature=t)` —
  cache key naturally varies because temperature is part of it.
- For each response, call `evaluator.score(prompt, response)`.
- Returns `list[SiblingSample]` where `SiblingSample = (response, temperature,
  creativity, logic, rationale, failure_modes, evaluator_cached, generator_cached)`.

`training/preferences.py::mine_pairs`
- Inputs: `(prompt: str, domain: str, siblings: list[SiblingSample],
  *, min_gap: float = 2.0)`.
- Compute `score = creativity + logic` (range 2..10) per sibling.
- Pick `chosen = argmax(score)`, `rejected = argmin(score)`.
- If `chosen.score - rejected.score < min_gap` → return `[]` (skip prompt).
- If `chosen.response == rejected.response` (caching collisions across temps) →
  return `[]`.
- Otherwise return a single `PreferencePair`. Best-vs-worst, one pair per
  prompt, by design.

`scripts/sample_siblings.py` orchestrates:
- CLI: `--n-prompts 300 --siblings 6 --min-gap 2.0 --max-spend-usd 4.00
  --max-calls 5000 --seed 42 --out data/preferences.jsonl --resume`.
- Flow: shuffle DOMAINS by seed → for each prompt slot, call
  `prompter.generate_prompt(domain, salt=i)` → `sample_and_score_siblings(...)`
  → `mine_pairs(...)` → append pair (if any). Pre-flight cost estimate via
  `pricing.estimate_sibling_run_cost`; abort if estimate > `--max-spend-usd`.
- Cost-tracked components: `prompter`, `generator`, `evaluator` (mirrors
  `run_feedback_loop.py`'s tally pattern at `scripts/run_feedback_loop.py:104-110`).
- `--resume` reads existing JSONL, skips prompts whose hash is already
  represented (idempotent re-runs).
- Per-prompt yield logged at INFO; aggregate at end (`pairs_emitted /
  prompts_attempted`, mean gap, cost split).

Reuses unchanged: `prompter.generate_prompt` (`feedback/prompter.py:51`),
`evaluator.score` (`feedback/evaluator.py:64`), `pricing.cost_of`
(`feedback/pricing.py:17`), `TogetherGenerator.generate_with_meta`.

## DPO Training Contract

`training/dpo.py::DPOConfig`
- Pydantic dataclass holding: `base_model_id`, `adapter_out_dir`,
  `lora_rank=16`, `lora_alpha=32`, `lora_dropout=0.05`,
  `target_modules=("q_proj","k_proj","v_proj","o_proj")`, `beta=0.1`,
  `learning_rate=5e-5`, `num_train_epochs=3`, `per_device_batch_size=2`,
  `gradient_accumulation_steps=8`, `max_length=1024`, `max_prompt_length=384`,
  `bf16=True`, `gradient_checkpointing=True`, `seed=42`.

`training/dpo.py::build_lora_config(cfg) -> peft.LoraConfig` — pure-Python,
testable without a GPU.

`training/dpo.py::build_dpo_trainer(model, ref_model, tokenizer, dataset, cfg)
  -> trl.DPOTrainer` — wires up TRL with the LoRA config, dataset, and TRL's
  `DPOConfig` (TRL ≥ 0.7).

`training/dpo.py::dataset_from_pairs(pairs: list[PreferencePair],
    tokenizer) -> datasets.Dataset` — converts our schema to TRL's expected
columns: `prompt`, `chosen`, `rejected`. Applies the Qwen chat template so
both arms include the same system+user framing.

`scripts/train_dpo.py`
- Modal-decorated entrypoint:
  ```python
  app = modal.App("brainstormlm-dpo")
  image = modal.Image.debian_slim().pip_install("transformers", "trl", "peft", ...)

  @app.function(gpu="A100-80GB", timeout=2 * 60 * 60, image=image)
  def train(...): ...
  ```
- Inputs (CLI via `modal run scripts/train_dpo.py`):
  `--pairs-path data/preferences.jsonl --run-id <ulid>
   --max-pairs 2000 --max-modal-usd 25.00 --hf-token <secret>`.
- Pre-flight: cost estimate from `pricing.estimate_modal_cost(hours=2,
  gpu="A100-80GB")` (informational; actual cap is `--max-modal-usd` enforced
  by the function timeout + Modal's per-second billing).
- Steps: read pairs JSONL → split 95/5 train/val → load
  `Qwen/Qwen2.5-7B-Instruct` in bf16 → build LoRA config → build TRL trainer
  → `.train()` → `model.save_pretrained(adapter_out_dir)` →
  `tokenizer.save_pretrained(adapter_out_dir)`.
- Outputs to `data/adapters/<run_id>/`:
  `adapter_model.safetensors`, `adapter_config.json`, `tokenizer.json`,
  `train_metrics.csv` (per-step DPO loss + reward margins),
  `manifest.json` (config, pair count, training time, cost estimate).
- Local-run fallback: `python scripts/train_dpo.py --local` runs the same
  function in-process for tests/dev (will only complete on a machine with a
  big enough GPU; primarily for `--local --dry-run` smoke testing).

Open: if Modal billing turns out to be a blocker, the same `train` function
runs unchanged on RunPod via their template — only the orchestration layer
changes. Don't write the RunPod path until/unless it's needed.

## HFGenerator Contract

`src/brainstormlm/model/hf.py::HFGenerator`

```python
class HFGenerator:  # implements model.interface.Generator
    def __init__(
        self,
        *,
        base_model_id: str | None = None,
        adapter_path: Path | None = None,
        device: str | None = None,    # auto-detect cuda > mps > cpu
        dtype: str = "bf16",
    ) -> None: ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str: ...

    def generate_with_meta(self, ...) -> tuple[str, bool]: ...
```

- Lazy-loads weights on first `generate` (heavy import; CLI startup must stay
  fast for the no-adapter path).
- Applies the Qwen chat template (`tokenizer.apply_chat_template`) so the
  prompt format matches Together's serving exactly.
- Cache: same disk cache layer as `TogetherGenerator` keyed by
  `(base_model_id, adapter_path, system, prompt, temperature, max_tokens)`.
  Cache hits short-circuit before any model load — `brainstorm ask` against a
  warm cache stays fast even with `--adapter` set.
- The `Generator` protocol stays unchanged; consumers (CLI, feedback loop,
  eval harness) remain blind to which implementation backs them.

`model/__init__.py` re-exports both:
```python
from brainstormlm.model.base import TogetherGenerator
from brainstormlm.model.hf import HFGenerator
```

`model/factory.py::make_generator(adapter_path: Path | None) -> Generator`
returns `HFGenerator` if `adapter_path is not None`, else `TogetherGenerator`.
This is the single switch point used by CLI and scripts.

## Ablation Eval Harness Contract

`src/brainstormlm/eval/harness.py::run_ablation`

Cells: 4 arms × 50 held-out prompts.

| Arm | Generator | Memory |
|---|---|---|
| `base` | `TogetherGenerator` | off |
| `base+mem` | `TogetherGenerator` | on (k_success=3, k_failure=2) |
| `trained` | `HFGenerator(adapter_path=…)` | off |
| `trained+mem` | `HFGenerator(adapter_path=…)` | on (k_success=3, k_failure=2) |

For each arm × prompt: generate → score → record `creativity`, `logic`,
`creativity + logic`, `memory_context_ids`, `arm`, `adapter_path`,
`generator_cached`, `evaluator_cached`. Persist to
`data/eval/ablation_<run_id>.jsonl` (NOT `trajectories.jsonl` — held-out
prompts must never enter the training pool, same rule as plan-03).

Aggregate output (printed table + saved as `ablation_<run_id>_summary.json`):

```
              creativity  logic  total  Δ vs base   p (paired bootstrap)
base                 3.4    3.6    7.0           —                    —
base+mem             3.6    3.7    7.3        +0.3                 0.07
trained              3.7    3.7    7.4        +0.4                 0.04
trained+mem          3.9    3.8    7.7        +0.7                <0.01
```

Paired bootstrap (10k resamples) on per-prompt deltas, one-sided
`H1: arm > base`. Reuses the bootstrap helper in `scripts/eval_memory.py`
(extract to `eval/stats.py` during this work to share between memory eval and
ablation eval).

`scripts/eval_ablation.py` is the CLI entrypoint:
`--adapter <path> --k-success 3 --k-failure 2 --seed 42
 --out data/eval/ablation_<run_id>.jsonl`.

## Config Additions

```python
# src/brainstormlm/config.py
hf_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
adapters_dir: Path = Field(default=Path("./data/adapters"),
                           alias="BRAINSTORMLM_ADAPTERS_DIR")
default_adapter_path: Path | None = Field(default=None,
                           alias="BRAINSTORMLM_DEFAULT_ADAPTER")
preferences_path: Path = Field(default=Path("./data/preferences.jsonl"),
                           alias="BRAINSTORMLM_PREFERENCES_PATH")
hf_token: SecretStr | None = Field(default=None, alias="HF_TOKEN")
modal_workspace: str | None = Field(default=None, alias="MODAL_WORKSPACE")
```

## CLI & Script Wiring

**`brainstorm ask`** (extend `cli.py`):
```
brainstorm ask "<prompt>" --adapter data/adapters/<run_id>/ \
    --use-memory --k-success 3 --k-failure 2
```
- `--adapter <path>` → switch to `HFGenerator` via `make_generator`.
- `--no-adapter` → force `TogetherGenerator` even if
  `default_adapter_path` is set in config.
- All existing flags (`--temperature`, `--max-tokens`, `--system`,
  `--no-cache`, `--use-memory`, `--k-success`, `--k-failure`,
  `--show-lessons`) remain unchanged.

**`scripts/run_feedback_loop.py`** (extend):
- Add `--adapter <path>`. When set, swap to `HFGenerator` via
  `make_generator`. Trajectory records the adapter path in a new optional
  `adapter_path: str | None = None` field — additive schema change, default
  `None` for back-compat with existing trajectories.

**`scripts/sample_siblings.py`** (new): described above.

**`scripts/train_dpo.py`** (new): Modal-decorated; described above.

**`scripts/eval_ablation.py`** (new): described above.

## Tests (TDD per CLAUDE.md)

All tests must pass without GPUs and without network calls.

1. **`test_preferences_pairs.py`** — pure logic.
   - `mine_pairs` with a hand-built sibling list returns the expected pair;
     confirm `chosen` has top score, `rejected` has bottom score.
   - Below-`min_gap` siblings → returns `[]`.
   - Tied responses (cache-collision case) → returns `[]`.
   - Deterministic `id` from `(prompt, chosen, rejected)`.
2. **`test_preferences_io.py`** — JSONL round-trip.
   - Write 10 pairs → read back → 10 identical objects.
   - Append discipline preserved (no rewrites).
3. **`test_dpo_config.py`** — config builder, no training.
   - `build_lora_config(cfg)` returns a `LoraConfig` with rank=16,
     target_modules matching the spec.
   - `dataset_from_pairs` with 5 pairs + a stub tokenizer that records calls →
     resulting dataset has columns `["prompt","chosen","rejected"]`, length 5,
     and the chat template was applied to each row.
4. **`test_hf_generator.py`** — protocol conformance.
   - Monkeypatch `transformers.AutoModelForCausalLM.from_pretrained` and
     `tokenizer.apply_chat_template` with stubs that return canned tokens.
   - `HFGenerator(...).generate("hi", system="s")` returns the canned string.
   - With `adapter_path` set: the stub's `PeftModel.from_pretrained` is called
     once on first generate; subsequent calls reuse the loaded model.
   - Cache hit on identical args skips both the model load and the generate
     call (verified via spy on `from_pretrained`).
5. **`test_eval_harness.py`** — ablation plumbing.
   - Stub `Generator` and stub scorer driven from a fixed dict; run
     `run_ablation` over a 3-prompt fixture; assert the output has 4 rows × 3
     prompts = 12 records and the summary table picks the right means.
   - Bootstrap helper returns `p ≤ 0.05` on a hand-rigged "arm clearly beats
     base" fixture.

Each test file ships in the same commit as the module it covers, per
`CLAUDE.md`.

## Implementation Sequence

The work fans out in three parallelizable strands once the schema and
config are in place. Land Step 1 on the main thread, then dispatch Steps 2a /
2b / 2c, then integrate.

**Step 1 — main thread (sequential, ~25 min):**
1. Add deps to `pyproject.toml` (`transformers`, `torch`, `peft`, `trl`,
   `accelerate`, `datasets`, `modal`); `uv sync`. Note: `torch` is heavy;
   keep it as a top-level dep so the no-adapter CLI path doesn't try to
   import it (HF generator imports torch lazily inside its module).
2. Add config fields (`hf_base_model`, `adapters_dir`,
   `default_adapter_path`, `preferences_path`, `hf_token`,
   `modal_workspace`).
3. Define `PreferencePair` schema and JSONL helpers in
   `training/preferences.py` (mirror `feedback/trajectories.py`'s `append`
   and `read_all` shape).
4. Add `eval/stats.py` and lift the paired-bootstrap helper out of
   `scripts/eval_memory.py` (light refactor; `eval_memory.py` imports the
   shared helper).
5. Stub `training/__init__.py` re-exporting `PreferencePair`.

**Step 2 — fan out three teammates in parallel:**

- Teammate **`pair-mining`** owns:
  - `training/preferences.py::sample_and_score_siblings`, `mine_pairs`.
  - `scripts/sample_siblings.py` (full CLI, cost guardrails, `--resume`).
  - `tests/test_preferences_pairs.py`, `tests/test_preferences_io.py`.
  - Constraint: never hits the network in tests; uses
    `feedback/{prompter,evaluator,pricing}` unchanged.

- Teammate **`hf-generator`** owns:
  - `model/hf.py::HFGenerator` (lazy weight load, chat template, cache).
  - `model/factory.py::make_generator`.
  - `model/__init__.py` re-exports.
  - `tests/test_hf_generator.py` (monkeypatched HF + PEFT).
  - Constraint: tests pass without GPU and without network — all HF/PEFT
    calls stubbed.

- Teammate **`dpo-trainer`** owns:
  - `training/dpo.py` (`DPOConfig`, `build_lora_config`,
    `dataset_from_pairs`, `build_dpo_trainer`).
  - `scripts/train_dpo.py` (Modal entrypoint + local fallback).
  - `tests/test_dpo_config.py`.
  - Constraint: training tests cover config + dataset construction only; the
    actual `.train()` call is exercised only via the Modal job. No GPU code
    in test path.

**Step 3 — main thread (sequential, integration):**
1. Wire `--adapter` / `--no-adapter` into `cli.py::ask`; route through
   `make_generator`.
2. Wire `--adapter` into `scripts/run_feedback_loop.py`; add
   `adapter_path` to `Trajectory` (additive).
3. Implement `eval/harness.py::run_ablation` and `scripts/eval_ablation.py`,
   reusing `eval/stats.py`.
4. Add `tests/test_eval_harness.py` with stub generator + stub scorer.
5. Run the verification gates below.
6. Commit per `CLAUDE.md` (imperative mood; tests ship with their code;
   one logical change per commit — likely 5-6 commits: deps+config+schema,
   pair mining + sibling sampling, HF generator + factory, DPO trainer,
   ablation harness + integration).

## Verification

All gates must pass before closing plan-04:

1. `uv run pytest tests/` — all old + new tests green (≥10 new tests).
2. **Sibling sampling smoke**:
   `uv run python scripts/sample_siblings.py --n-prompts 5 --siblings 6
    --max-spend-usd 0.20` produces 3–5 pairs in
   `data/preferences.jsonl` with valid schema. Second invocation with
   `--resume` is fully cache-served (cost ≈ $0, no new pairs).
3. **HFGenerator smoke** (CPU-only, OK to be slow):
   `uv run brainstorm ask "Quick test" --adapter <small-fake-adapter>` —
   if no real adapter exists yet, this gate is exercised by
   `tests/test_hf_generator.py` instead and noted as deferred to gate 5.
4. **Full pair run**:
   `uv run python scripts/sample_siblings.py --n-prompts 300 --siblings 6
    --max-spend-usd 4.00` produces ≥1500 pairs. Inspect the gap
   distribution by hand; if the median gap is <2.5 the rubric is too noisy
   and pair-mining tuning is an open item before training.
5. **DPO training run**:
   `modal run scripts/train_dpo.py --pairs-path data/preferences.jsonl
    --run-id <ulid> --max-modal-usd 25` completes inside the budget,
   produces `data/adapters/<run_id>/` with `adapter_model.safetensors` and
   `manifest.json`. `train_metrics.csv` shows DPO loss decreasing and
   reward margin (chosen vs rejected) increasing.
6. **End-to-end ablation (the Phase 2 exit gate)**:
   `uv run python scripts/eval_ablation.py --adapter
    data/adapters/<run_id>/ --k-success 3 --k-failure 2 --seed 42`
   prints the 4-row table. Required outcomes:
   - `trained` row beats `base` row, paired-bootstrap p < 0.10.
   - `trained+mem` row beats `base+mem` row (the plan-03 result),
     paired-bootstrap p < 0.10.
   - The two memory rows beat their no-memory counterparts (regression
     check on plan-03's gain — should still hold).
   If gate 6 fails: log the result and iterate on either the LoRA rank
   (try 32) or the pair-mining `--min-gap` (try 3); do not relax exit
   criteria.
7. Re-running `eval_ablation.py` is fully cache-served — total cost reads
   $0.00, wall-clock under a few seconds.
8. `data/eval/ablation_<run_id>.jsonl` exists and is disjoint from both
   `data/trajectories.jsonl` and `data/preferences.jsonl` (held-out
   prompts must never enter training).

## Non-Goals

- **No PPO, no online RL, no learned reward model.** Master plan rules these
  out for the project as a whole; DPO is the only training algorithm.
- **No SFT warm-up.** TRL's DPOTrainer trains directly on the base
  instruction-tuned Qwen; an SFT pass on chosen-only completions is a
  separate experiment if DPO alone underperforms.
- **No multi-turn preference data.** Phase 1–4 are single-turn brainstorms.
- **No vLLM serving.** Plan-05 may swap `HFGenerator` for a vLLM-backed
  variant; plan-04's `HFGenerator` is the simplest correct path.
- **No automatic retraining.** Continuous training lives in plan-06.
- **No adapter merging into base weights.** Adapters stay as PEFT layers for
  swap-friendly ablation; merging is trivial when/if production needs it.
- **No mixing real human feedback into pairs.** None exists yet (plan-06).
- **No training on `HELD_OUT` prompts.** Mechanically enforced by sourcing
  prompts from `prompter.generate_prompt` over `DOMAINS` only.

## Open Items (resolve during implementation)

1. **Pair yield empirical check.** Run a 20-prompt pilot before the full
   sibling-sampling job; if pair yield < 60% at `min_gap=2`, drop to
   `min_gap=1` rather than raising N (cheaper).
2. **DPO β.** Default 0.1; if the trained model collapses to base outputs
   (zero gradient signal) or diverges (degenerate text), sweep
   {0.05, 0.1, 0.2}.
3. **LoRA rank.** Default 16; if gate 6 fails by a small margin, try 32
   before changing pair construction.
4. **Modal vs RunPod.** Default Modal; switch to RunPod if Modal credits run
   out or the A100 queue is blocking. Code path is identical; only the
   orchestration script changes.
5. **HFGenerator on Apple Silicon.** Likely too slow for `eval_ablation.py`
   on M-series even with MPS; if so, run the eval on Modal too via a thin
   `eval_ablation_modal.py` wrapper. Defer until measured.
6. **Adapter path conventions.** First run uses ULID `<run_id>`; consider a
   `latest` symlink at `data/adapters/latest` after gate 6 passes so the CLI
   can default to it.

## Handoff to plan-05

Plan-05 (steering integration) consumes:
- The `Generator` protocol unchanged — it adds a third implementation
  (`SteeredGenerator`) without touching training or eval code.
- The full ablation harness from plan-04 — plan-05 adds rows for
  `steered`, `steered+mem`, `steered+trained`, `steered+trained+mem`.
- The trained adapter from plan-04 — plan-05 either applies it to the
  steered base or retrains a fresh adapter on steered-base trajectories;
  that decision belongs to plan-05.
