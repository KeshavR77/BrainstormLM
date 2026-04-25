# BrainstormLM — Plan of Plans

_Status: finalized locally 2026-04-24. Ultraplan cloud refinement was skipped
(repo had no GitHub remote at the time); proceeding directly to implementation._

_Progress (2026-04-24):_
- **plan-01 (foundation) — complete.** Commit `48e66da`. `brainstorm ask` wired
  end-to-end against `Qwen/Qwen2.5-7B-Instruct-Turbo` on Together, cached-call
  pipeline working (cold 3.6s → warm sub-ms lookup), `--no-cache` verified,
  tests 3/3 green. See `docs/plans/plan-01-foundation.md`.
- **Judge LLM swap:** Anthropic → **OpenAI**. Plan-02 will pin
  **`gpt-4o-mini`** for both prompter and evaluator. Cost: ~$0.09 per 200-
  trajectory run. Rationale: 15× cheaper than `gpt-4o` at plan-04 scale
  (~$3 vs ~$50 for 8k eval calls) with rubric-following good enough for MVP.
- **Generator backend:** **Together stays pinned through plan-02** (per-run
  cost ~$0.04). Migrate to vLLM-on-Unicorn at Phase 3 when steering requires
  direct activation access — no earlier, to avoid a day of CUDA/SSH setup
  while the feedback loop is still being built.

## Context

BrainstormLM is a creative-brainstorming LLM that uses **steering vectors** to push a
base open-source model toward more creative outputs. Today the repo at
`/Users/keshavrangan/BrainstormLM` is a green field (only `CLAUDE.md` exists — no
code, no git repo, no dependencies).

This document is a **master plan** that sequences the build into phases, each of
which will get its own sub-plan document before implementation. The phasing is
intentionally MVP-first: the synthetic feedback loop and a retrieval-based
improvement mechanism come online before any training or steering work. Steering
vectors are deliberately deferred — the feedback loop is built against a stock
open-source base model so the steered variant can drop in later without touching
the learning pipeline.

The ultimate outcome is a system that:
1. Serves brainstorming prompts to end users with a steered, creativity-boosted LLM.
2. Collects (eventually real, for now synthetic) feedback on every trajectory.
3. Uses that feedback to improve itself continuously via **both** an inference-time
   memory (ReasoningBank) and an offline preference-learning loop (DPO).

All sub-plan documents will live at `BrainstormLM/docs/plans/plan-NN-*.md` and be
version-controlled alongside the code.

---

## North-Star Architecture (end state)

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐
│  User (or LLM-  │─────▶│  BrainstormLM    │◀─────│  ReasoningBank     │
│  simulated      │ prmt │  (steered OSS    │ ctx  │  memory (lessons   │
│  prompter)      │      │  model + LoRA)   │      │  from past trials) │
└─────────────────┘      └────────┬─────────┘      └────────────────────┘
                                  │ response              ▲
                                  ▼                       │
                          ┌───────────────┐               │
                          │  Creativity   │──── lesson ───┘
                          │  Evaluator    │      extraction
                          │  (Claude API) │
                          └───────┬───────┘
                                  │ (prompt, response, score, rationale)
                                  ▼
                          ┌───────────────┐
                          │  Trajectory   │──── periodic ───▶  DPO
                          │  store (JSONL │     preference        trainer
                          │  + vector DB) │     mining            (LoRA)
                          └───────────────┘
```

Components:
- **Base model**: a small open-source LLM (e.g. Llama-3.2-3B-Instruct or
  Qwen2.5-7B-Instruct) loaded via HuggingFace `transformers`. Chosen in plan-01.
- **Steered model**: same base + steering vectors. Drops into the pipeline in
  Phase 3 with no API change.
- **Prompter (synthetic)**: closed-source LLM (Claude) that generates diverse
  brainstorming prompts across a seeded domain taxonomy.
- **Evaluator**: closed-source LLM (Claude) prompt-engineered with a rubric
  scoring **creativity** and **logical soundness** separately, plus rationale.
- **ReasoningBank memory**: vector store of _extracted reasoning lessons_ (not
  raw trajectories) derived from both successful and failed trials, per the
  Google paper. Retrieved at inference and injected into the system prompt.
- **Preference store / DPO trainer**: mines preference pairs from scored
  trajectories, trains a LoRA adapter over the base model.

---

## Guiding Principles

1. **Swappable base model.** All code paths treat the generator as an interface
   so the steered variant replaces the base without touching eval/training code.
2. **CLI-first.** Per `CLAUDE.md`, external APIs and core flows are exposed as
   CLI commands before any UI. Web UI only arrives in Phase 4.
3. **Cache every closed-source API call.** Claude/GPT calls are the dominant
   cost. Cache by `sha256(prompt + model + params)` to a local JSON store.
4. **Trajectories are append-only JSONL.** Never mutate; all downstream tools
   (memory extraction, preference mining, eval) read from the same log.
5. **Fail loudly at system boundaries only.** Trust internal code; validate
   model outputs, API responses, and user input — nothing else.
6. **DRY across phases.** Model loading, Claude client, trajectory schema, and
   config are shared modules, not copy-pasted.
7. **Cost guardrails.** Every script that calls a paid API takes `--max-calls`
   and `--max-spend-usd`. Abort cleanly when exceeded.

---

## Repository Layout (end-of-Phase-4 target)

```
BrainstormLM/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── .env.example
├── .gitignore
├── src/brainstormlm/
│   ├── config.py                  # Pydantic settings, env loading
│   ├── model/
│   │   ├── base.py                # HF transformers generator
│   │   ├── steered.py             # Phase 3: steering vector wrapper
│   │   └── interface.py           # Generator protocol (swap point)
│   ├── clients/
│   │   └── anthropic.py           # Cached Claude client
│   ├── feedback/
│   │   ├── prompter.py            # LLM-generated brainstorm prompts
│   │   ├── evaluator.py           # Creativity + logic rubric scorer
│   │   └── trajectories.py        # JSONL append + schema
│   ├── memory/
│   │   ├── lessons.py             # Lesson-extraction LLM chain
│   │   ├── store.py               # Vector store (FAISS or Chroma)
│   │   └── retrieve.py            # k-NN + prompt augmentation
│   ├── training/
│   │   ├── preferences.py         # Build (chosen, rejected) pairs
│   │   └── dpo.py                 # TRL DPOTrainer wrapper
│   ├── eval/
│   │   ├── harness.py             # Held-out prompt set + metrics
│   │   └── fixtures.py            # Frozen eval prompts
│   └── cli.py                     # Typer entrypoints
├── scripts/
│   ├── run_feedback_loop.py       # One-shot: N trajectories → store
│   ├── train_dpo.py               # Offline training job
│   └── rebuild_memory.py          # Re-extract lessons from trajectories
├── docs/plans/
│   ├── plan-01-foundation.md
│   ├── plan-02-synthetic-feedback.md
│   ├── plan-03-reasoningbank.md
│   ├── plan-04-rlhf-dpo.md
│   ├── plan-05-steering-integration.md
│   └── plan-06-production.md
├── data/                          # gitignored
│   ├── trajectories.jsonl
│   ├── cache/                     # Claude response cache
│   └── memory/                    # vector store artifacts
└── tests/
```

---

## Phase Overview

| Phase | Scope | Sub-plans | Gates |
|-------|-------|-----------|-------|
| **1 — MVP** | Greenfield → end-to-end synthetic feedback loop with ReasoningBank retrieval. No training. | plan-01, plan-02, plan-03 | Retrieval-augmented model beats baseline on held-out evaluator scores. |
| **2 — RLHF** | Offline DPO training on mined preference pairs; LoRA adapter served at inference. | plan-04 | Trained model beats Phase 1 retrieval-only baseline on held-out prompts. |
| **3 — Steering** | Swap in the real steered model. Regression-test the full loop. | plan-05 | Steered model passes Phase 1 & 2 evals without pipeline changes. |
| **4 — Production** | Real users, web UI, continuous training, observability, deployment. | plan-06 | End-to-end live system with ≥1 non-Keshav user completing a session. |

Dependencies are strict: each phase assumes all prior phase exit criteria have
been met. Sub-plans within a phase can be executed sequentially.

---

## Phase 1 — MVP: Synthetic Feedback Loop + ReasoningBank

**Why this is the MVP.** Everything else needs a working evaluator and a
trajectory log to stand on. ReasoningBank is inference-time-only (no GPU
training), so it's the cheapest-to-build improvement mechanism and it produces
the data that Phase 2 will train on.

### Sub-plan 1.1 — `plan-01-foundation.md`

Scope:
- `git init`, `.gitignore`, `pyproject.toml` (uv or poetry), `.env.example`.
- Dependencies: `transformers`, `torch`, `accelerate`, `bitsandbytes` (for
  quantization), `anthropic`, `typer`, `pydantic-settings`, `python-dotenv`.
- `src/brainstormlm/model/base.py`: wrap HF `AutoModelForCausalLM` +
  `AutoTokenizer`. Single `generate(prompt: str, **gen_kwargs) -> str` method.
- `src/brainstormlm/model/interface.py`: `Generator` protocol so Phase 3 can
  swap implementations without downstream code changes.
- `src/brainstormlm/clients/anthropic.py`: cached Claude client. Cache keyed by
  sha256 of (model, system, messages, temperature).
- `src/brainstormlm/config.py`: env-loaded settings (API keys, model id,
  quantization flag, data dir).
- `src/brainstormlm/cli.py`: `brainstorm ask "<prompt>"` — loads model, prints
  response. Smoke test that the base model works end-to-end.

Decisions to resolve in this sub-plan (not here):
- Exact base model id (gated on available VRAM).
- 4-bit vs 8-bit vs full-precision inference.
- `uv` vs `poetry` for dep management.

### Sub-plan 1.2 — `plan-02-synthetic-feedback.md`

**Pinned decisions (2026-04-24):**
- Prompter model: **`gpt-4o-mini`** via `clients/openai.py`.
- Evaluator model: **`gpt-4o-mini`** via `clients/openai.py`, `response_format=json_schema`.
- Generator: `TogetherGenerator` (Qwen2.5-7B-Instruct-Turbo) — unchanged from plan-01.
- Evaluator style: absolute rubric for MVP; pairwise preference mining deferred to plan-04.

Scope:
- `feedback/prompter.py`: OpenAI-based prompt generator. Input = domain
  taxonomy seed (science, art, business, writing, product, personal, …) +
  diversity instructions. Output = a brainstorming prompt. One prompt per call.
- `feedback/evaluator.py`: OpenAI call with structured-output JSON schema:
  `{creativity: 1-5, logic: 1-5, rationale: str, failure_modes: [str]}`.
  Rubric written once, versioned, pinned to `gpt-4o-mini`.
- `feedback/trajectories.py`: Pydantic schema + append-only JSONL writer.
  Schema fields: `id`, `timestamp`, `prompt`, `response`, `generator_version`,
  `prompter_model`, `evaluator_model`, `creativity`, `logic`, `rationale`,
  `failure_modes`, `memory_context_ids`.
- `scripts/run_feedback_loop.py`: generate N prompts → N responses → score all
  → append to JSONL. Respects `--max-calls` and `--max-spend-usd`. Per-run
  cost target: <$0.15 at `--n 200` (empirical baseline: $0.09 @ gpt-4o-mini
  + $0.04 Together).
- Held-out eval seed set (~50 frozen prompts) checked into `eval/fixtures.py`.
  These are NEVER used for training/retrieval — only for measurement.

### Sub-plan 1.3 — `plan-03-reasoningbank.md`

Scope (mirrors the Google ReasoningBank paper):
- `memory/lessons.py`: Claude chain that consumes a scored trajectory (prompt,
  response, scores, rationale) and extracts 1–3 **abstracted reasoning
  lessons** — generalizable patterns, not the raw trajectory. Handles
  successes AND failures (failure lessons phrased as anti-patterns).
- `memory/store.py`: vector store (start with FAISS + sentence-transformers
  embeddings; schema fields: `lesson`, `kind` ∈ {success, failure}, `domain`,
  `source_trajectory_id`, `embedding`).
- `memory/retrieve.py`: given a new prompt, retrieve top-k lessons (mix of
  success + failure) and inject into the system prompt with clear labeling.
- Wire into `cli.py`: `brainstorm ask "<prompt>" --use-memory`.
- Backfill job: `scripts/rebuild_memory.py` re-extracts lessons from all
  trajectories on demand (lesson-extraction prompt will iterate).

### Phase 1 Exit Criteria

1. `brainstorm ask` produces a response for any prompt.
2. `run_feedback_loop.py --n 200` populates `trajectories.jsonl` with 200
   scored trajectories inside budget.
3. `rebuild_memory.py` produces a non-empty memory bank.
4. On the 50 held-out eval prompts, **`--use-memory` beats the no-memory
   baseline** on mean `creativity + logic` by a statistically meaningful
   margin (paired bootstrap, p < 0.1 is acceptable for MVP).
5. All closed-source API calls cached; re-running the eval costs $0.

---

## Phase 2 — RLHF Training (DPO)

**Why DPO, not PPO.** DPO trains directly on `(prompt, chosen, rejected)`
triples without a separate reward model. Given that our "reward" is a noisy
LLM-judge score, collapsing that noise into pairwise preferences and training
with DPO is simpler, more stable on small compute, and well-supported by TRL.

### Sub-plan 2.1 — `plan-04-rlhf-dpo.md`

Scope:
- `training/preferences.py`: preference-pair miner. For each prompt, sample
  N∈{4,8} responses from the current model at varied temperatures, score all
  with the evaluator, emit `(prompt, chosen, rejected)` pairs where chosen's
  score materially exceeds rejected's (configurable threshold).
- `training/dpo.py`: thin wrapper around `trl.DPOTrainer` with LoRA
  (`peft.LoraConfig`). Trains an adapter; base weights frozen.
- `scripts/train_dpo.py`: run training from a preference-pairs dataset.
  Logs to WandB (optional) or local CSV.
- Inference integration: `model/base.py` loads the LoRA adapter when present;
  `brainstorm ask --adapter <path>` and `brainstorm ask --adapter none`.
- Extend `eval/harness.py` with regression test: baseline vs retrieval vs
  trained vs trained+retrieval on held-out prompts. Ablations matter — we
  need to know ReasoningBank and DPO improvements don't cannibalize.

Decisions to resolve:
- Absolute-threshold vs top-vs-bottom pair construction.
- LoRA rank / target modules.
- GPU: local consumer card vs rented A100/H100 (cost vs iteration speed).

### Phase 2 Exit Criteria

1. DPO training completes on 1k+ mined preference pairs without OOM.
2. On held-out eval, **trained-model-with-retrieval beats Phase-1
   retrieval-only baseline** on mean `creativity + logic`.
3. Ablation table produced: (base, base+mem, trained, trained+mem).

---

## Phase 3 — Steering Vector Integration

**Why this is Phase 3, not Phase 1.** You explicitly said the steered model is
swap-in-later. All prior phases must work against the plain base model first
so that when steering lands, any quality delta is attributable to steering
alone — not confounded with pipeline changes.

### Sub-plan 3.1 — `plan-05-steering-integration.md`

Scope:
- `model/steered.py`: steering-vector implementation behind the `Generator`
  protocol from Phase 1. Library choice (nnsight / repeng / transformer_lens)
  is a sub-plan decision.
- Training/derivation of the steering vector itself (likely contrast between
  "creative response" and "generic response" activations at target layers).
- Config flag: `brainstorm ask --steered` routes through the steered model.
- Regression: re-run Phase 1 and Phase 2 evals with `--steered`; numbers must
  not regress. Ideally they improve.

### Phase 3 Exit Criteria

1. Steered model loads and generates end-to-end.
2. Feedback loop (`run_feedback_loop.py --steered`) runs clean.
3. DPO training runs on trajectories from the steered model without code
   changes outside `model/`.
4. Steered-model eval matches or exceeds equivalent base-model eval.

---

## Phase 4 — Production System

**Why last.** Building real user infrastructure before the learning loop is
validated risks shipping an improvement mechanism that doesn't actually improve
anything. Phases 1–3 produce the evidence that the loop works; Phase 4 exposes
it to humans.

### Sub-plan 4.1 — `plan-06-production.md`

Scope (this sub-plan may itself split when authored):
- **Serving**: FastAPI backend wrapping the same `Generator` interface;
  responses streamed; memory retrieval happens server-side.
- **UI**: minimal web frontend (Next.js or Streamlit) with:
  - Prompt box, streaming response
  - Thumbs up/down + optional text critique per response
  - Session history
- **Human feedback ingestion**: user ratings land in a new
  `feedback_human.jsonl` with the same schema shape as synthetic trajectories
  (plus `source: human|synthetic`). Training mixes both.
- **Continuous training**: scheduled job (cron / GitHub Actions / Modal) that
  re-mines preferences, retrains DPO, and promotes the adapter behind a
  feature flag after eval passes.
- **Observability**: structured logs, latency metrics, evaluator agreement
  rate (human vs LLM judge) as a drift signal.
- **Safety**: basic prompt-injection resistance for the memory context; rate
  limiting; content policy on user prompts.
- **Deployment**: GPU inference on Modal / Replicate / a fixed box;
  frontend on Vercel or similar.
- **A/B testing**: two adapters live; route users randomly; compare human
  feedback rates.

### Phase 4 Exit Criteria

1. A non-Keshav user completes a full session (prompt → response → feedback).
2. A scheduled training job ran at least once end-to-end from fresh human
   feedback to a promoted adapter.
3. Evaluator-vs-human agreement measured and logged; threshold set.
4. Incident runbook exists for "model is outputting garbage" and "training
   job is looping on bad data".

---

## Cross-Cutting Concerns (revisited in every sub-plan)

- **Config & secrets**: single `config.py`, never scattered `os.environ`.
- **Caching**: Claude calls cached from day 1 (plan-01). No exceptions.
- **Cost guardrails**: every paid-API script has `--max-calls` /
  `--max-spend-usd` and aborts cleanly.
- **Evaluator calibration**: once real human data exists (Phase 4), compute
  agreement between the LLM judge and humans; recalibrate the rubric if low.
- **Trajectory schema**: additive-only changes. Version the schema.
- **Reproducibility**: pin model ids (base model, Claude model for prompter,
  Claude model for evaluator); log them on every trajectory.

---

## Non-Goals

- Multi-user auth, orgs, billing (even in Phase 4 — it's a research system).
- Training the base model from scratch or full-parameter fine-tuning.
- Supporting arbitrary base models in Phase 1; one model id, pinned.
- Serving at high QPS; Phase 4 targets small user counts.
- Multi-turn conversations; Phase 1–4 are single-turn brainstorm requests.
  Multi-turn is explicitly out of scope.
- PPO, RLAIF with a learned reward model, or online RL — DPO only.

---

## Verification Strategy (end-to-end, per phase)

Each sub-plan will restate this for its scope; the master contract:

- **Phase 1**: `pytest tests/` green + `run_feedback_loop.py --n 50
  --eval-set held_out` shows memory beats baseline.
- **Phase 2**: same eval, with the DPO adapter row added; trained row beats
  retrieval-only row.
- **Phase 3**: same eval with `--steered`; no regression; ideally improved.
- **Phase 4**: live system responds to a real user; continuous-training job
  has run once from real feedback; dashboard shows trajectory count climbing.

---

## Critical Files to Create (by phase)

- **Phase 1**: `pyproject.toml`, `src/brainstormlm/{config,cli}.py`,
  `src/brainstormlm/model/{interface,base}.py`,
  `src/brainstormlm/clients/anthropic.py`,
  `src/brainstormlm/feedback/{prompter,evaluator,trajectories}.py`,
  `src/brainstormlm/memory/{lessons,store,retrieve}.py`,
  `src/brainstormlm/eval/{harness,fixtures}.py`,
  `scripts/{run_feedback_loop,rebuild_memory}.py`.
- **Phase 2**: `src/brainstormlm/training/{preferences,dpo}.py`,
  `scripts/train_dpo.py`.
- **Phase 3**: `src/brainstormlm/model/steered.py`.
- **Phase 4**: `src/brainstormlm/api/` (FastAPI), `frontend/` (web UI),
  `scripts/continuous_train.py`, deployment manifests.

No existing files to reuse — everything is new.

---

## Open Questions (resolve during sub-plan authoring, not now)

1. Base model id and quantization (plan-01).
2. Claude model ids for prompter and evaluator (plan-02).
3. Vector store backend: FAISS vs Chroma vs simple in-memory (plan-03).
4. Preference-pair construction rule (plan-04).
5. LoRA rank and target modules (plan-04).
6. Steering library: nnsight vs repeng vs transformer_lens (plan-05).
7. GPU host for training: local, Modal, Replicate, RunPod (plan-04 / plan-06).
8. Frontend stack: Next.js vs Streamlit (plan-06).
