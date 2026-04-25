# Plan 02 — Synthetic Feedback Loop

_Parent plan: `~/.claude/plans/come-up-with-a-magical-wren.md` (Phase 1, Sub-plan 1.2)._
_Builds on: `plan-01-foundation.md` (complete as of commit `48e66da`)._
_Authored: 2026-04-24._

## Context

Plan-01 left us with a working `brainstorm ask` CLI that produces Qwen2.5-7B
responses through a cached generator. Nothing scores those responses and nothing
gets logged. Plan-02 closes that loop: an OpenAI **prompter** that synthesizes
diverse brainstorming prompts, an OpenAI **evaluator** that scores each
`(prompt, response)` on creativity + logic with structured JSON output, a
**trajectory store** that appends every trial to a versioned JSONL, and a
**run_feedback_loop.py** script that orchestrates N trials end-to-end.

This is the MVP's data-production engine. Plan-03 (ReasoningBank) consumes the
trajectory JSONL to extract lessons. Plan-04 (DPO) consumes the same JSONL plus
sibling-sample batches to mine preference pairs. Both depend on this sub-plan's
schema being stable and additive-only from here forward.

No training happens in plan-02. No retrieval either. We are producing labeled
data and the frozen held-out eval set that every future phase will measure
against.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Prompter model | **`gpt-4o-mini`** | Cheap ($0.15/$0.60 per Mtok), rubric-follows well enough for prompt generation. |
| Evaluator model | **`gpt-4o-mini`** | Same model + `response_format=json_schema` for strict JSON. ~15× cheaper than `gpt-4o` at plan-04 scale. |
| Evaluator style | **Absolute 1–5 rubric** | Matches master-plan's MVP default. Pairwise comparison deferred to plan-04 preference mining. |
| Generator | **Unchanged from plan-01** | `TogetherGenerator(Qwen/Qwen2.5-7B-Instruct-Turbo)` — swap point is the `Generator` protocol. |
| Trajectory store | **Append-only JSONL** at `data/trajectories.jsonl` | One row per trial. Never mutated. Schema versioned. |
| Eval fixtures | **50 frozen prompts** in `eval/fixtures.py` | Hand-curated across domain taxonomy; never used for training/retrieval — measurement only. |
| Cost accounting | **Pre-flight estimate + running tally** | Budget-estimate from token counts before the loop starts; abort mid-loop if `--max-spend-usd` exceeded. |

## Deliverables

Files to **create**:

```
src/brainstormlm/feedback/__init__.py
src/brainstormlm/feedback/prompter.py          # OpenAI prompt generator
src/brainstormlm/feedback/evaluator.py         # OpenAI JSON-schema scorer
src/brainstormlm/feedback/trajectories.py      # Pydantic schema + JSONL writer
src/brainstormlm/feedback/pricing.py           # Per-model $/Mtok table + cost math
src/brainstormlm/eval/__init__.py
src/brainstormlm/eval/fixtures.py              # 50 frozen held-out prompts
scripts/__init__.py
scripts/run_feedback_loop.py                   # orchestrate N trajectories
tests/test_trajectories.py                     # schema + JSONL round-trip
tests/test_pricing.py                          # token→cost arithmetic
tests/test_evaluator_parse.py                  # JSON-schema parsing (no network)
```

Files to **extend**:

```
src/brainstormlm/clients/openai.py             # add complete_json() for structured output + token-usage return
src/brainstormlm/config.py                     # add prompter_model, evaluator_model, trajectories_path
.env.example                                   # no new vars (keys already in plan-01)
```

No new runtime deps. `tqdm` gets added for progress bars (dev group).

## Trajectory Schema (v1)

```python
# src/brainstormlm/feedback/trajectories.py
class Trajectory(BaseModel):
    schema_version: Literal[1] = 1
    id: str                    # uuid4
    timestamp: str             # ISO 8601 UTC
    domain: str                # seed domain label, e.g. "science.biology"
    prompt: str                # what the model was asked
    response: str              # what the generator returned
    generator_model: str       # "Qwen/Qwen2.5-7B-Instruct-Turbo"
    generator_temperature: float
    prompter_model: str        # "gpt-4o-mini"
    evaluator_model: str       # "gpt-4o-mini"
    rubric_version: int        # bump whenever evaluator rubric text changes
    creativity: int            # 1-5
    logic: int                 # 1-5
    rationale: str             # evaluator's written justification
    failure_modes: list[str]   # evaluator-tagged weaknesses, possibly empty
    memory_context_ids: list[str] = []  # reserved for plan-03 (empty in plan-02)
```

Rule: schema changes are **additive only**. To remove or rename a field, bump
`schema_version` and write a migration script — never mutate existing rows.

## Domain Taxonomy (seed)

Hard-coded list in `feedback/prompter.py`. The loop cycles through it with
`random.shuffle` so 200 prompts span the space evenly. Initial set (revisit
once we see evaluator scores cluster by domain):

```
science.physics, science.biology, science.chemistry, science.astronomy,
art.visual, art.music, art.writing_fiction, art.writing_poetry, art.film,
business.startup, business.marketing, business.ops,
product.consumer, product.b2b, product.hardware,
personal.productivity, personal.relationships, personal.hobbies,
tech.software, tech.ai, tech.hardware,
education.teaching, education.learning,
everyday.food, everyday.travel, everyday.home
```

## Step-by-Step Implementation

1. **Extend `clients/openai.py`** with two additions:
   - `complete_json(system, messages, model, schema, ...) -> tuple[dict, Usage]`
     — wraps Chat Completions with `response_format={"type": "json_schema", ...}`,
     returns parsed dict + `Usage(prompt_tokens, completion_tokens)`.
   - Existing `complete()` gets a second return value (tokens used) so the
     pricing layer can tally cost. Callers that don't care ignore it.
   - Cache layer already handles both paths — cache key includes schema hash for `complete_json`.

2. **`feedback/pricing.py`** — `PRICES: dict[str, tuple[float, float]]` mapping
   model id to `($/Mtok_in, $/Mtok_out)`. Covers `gpt-4o-mini`, `gpt-4o`,
   `Qwen/Qwen2.5-7B-Instruct-Turbo`. Functions:
   - `cost_of(model, prompt_tokens, completion_tokens) -> float`
   - `estimate_run_cost(n: int) -> float` — rough pre-flight using plan-02 token
     budget assumptions (prompter 300/80, generator 100/500, evaluator 1200/250).
     Abort the loop if estimate > `--max-spend-usd`.

3. **`feedback/prompter.py`** — `generate_prompt(domain: str) -> str`. System
   prompt pins the "one brainstorming prompt, return only the prompt string,
   specific-enough-to-answer, open-ended-enough-to-explore" contract.
   Temperature 1.2 for diversity. Model = `settings.prompter_model`.

4. **`feedback/evaluator.py`** —
   `score(prompt: str, response: str) -> EvaluatorResult`. Wraps
   `clients/openai.complete_json` with a pinned rubric system prompt and the
   JSON schema above. `rubric_version = 1` for plan-02. Bumping the rubric
   text means bumping `rubric_version` so old rows stay interpretable.

5. **`feedback/trajectories.py`** — append-only JSONL writer:
   - `append(t: Trajectory, path: Path) -> None` — one JSON object per line,
     `json.dumps(t.model_dump(), sort_keys=True)`.
   - `read_all(path: Path) -> Iterator[Trajectory]` — for downstream consumers.
   - Never re-opens in truncate mode. Mkdir parent as needed.

6. **`eval/fixtures.py`** — `HELD_OUT: list[tuple[str, str]]` = 50 hand-picked
   `(domain, prompt)` tuples spanning the taxonomy. Hard-coded. These are the
   only prompts the feedback loop is forbidden to generate from (prompter
   filters them out). They're the yardstick plan-03 and plan-04 measure against.

7. **`scripts/run_feedback_loop.py`** — Typer CLI, single command:
   ```
   python scripts/run_feedback_loop.py \
       --n 200 \
       --temperature 1.0 \
       --max-calls 1000 \
       --max-spend-usd 1.00 \
       --out data/trajectories.jsonl
   ```
   Flow:
   1. Pre-flight: compute `estimate_run_cost(n)`. If > `max_spend_usd`, abort.
   2. Shuffle domain taxonomy, cycle through with `n` draws.
   3. For each draw: prompter → generator → evaluator → append Trajectory.
   4. Running tally: tokens in/out, cumulative cost, call count. Abort if either
      cap exceeded. Always flush JSONL line before moving on (crash-resilient).
   5. Final summary: N successful, total cost, mean creativity/logic, cost breakdown.

8. **Cache discipline.** All three LLM calls (prompter, generator, evaluator)
   go through the existing sha256 cache. Re-running the same `--n 200` with
   same seed → zero API calls, zero cost. Critical for the Phase-1 exit
   criterion "re-running the eval costs $0".

9. **Tests.**
   - `test_trajectories.py`: write 3 Trajectories → read_all → equal; verify
     schema rejects extra fields; verify JSONL is one object per line.
   - `test_pricing.py`: `cost_of("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75`,
     `estimate_run_cost(200) < 0.15` (guards against regressions that would
     blow past our cost target).
   - `test_evaluator_parse.py`: feed a canned JSON response through the parser;
     assert EvaluatorResult fields match. No network.

10. **Verification gate: small live run.** `python scripts/run_feedback_loop.py
    --n 5 --max-spend-usd 0.05` must produce 5 scored rows in `data/trajectories.jsonl`.
    Second invocation with same args is cache-served → cost ≈ $0, runtime < 2s.

11. **Extend `.gitignore`.** `data/trajectories.jsonl` already caught by
    `data/`. No change needed; confirm during implementation.

12. **Commit** after verification per `CLAUDE.md` good-git-practices.

## Verification

All gates must pass before closing plan-02:

1. `uv run pytest tests/` — all old + new tests green.
2. `uv run python scripts/run_feedback_loop.py --n 5 --max-spend-usd 0.05`
   produces 5 well-formed trajectory rows in `data/trajectories.jsonl`;
   each row has valid `creativity`/`logic` in [1,5] and non-empty `rationale`.
3. Re-running the same command is cache-served — wall-clock < 2s, final
   "total cost" line reads $0.00.
4. `estimate_run_cost(200) < 0.15` (asserted by `test_pricing.py`).
5. Token caps behave: run with `--max-calls 3` aborts after 3 trajectories,
   JSONL partial-write is valid (no torn lines).
6. `from brainstormlm.eval.fixtures import HELD_OUT` imports 50 entries;
   prompter never draws from this set (add a sanity `assert` inside
   `run_feedback_loop.py`).

## Non-Goals for This Sub-Plan

- No retrieval, no memory, no `--use-memory` flag — plan-03.
- No preference-pair mining, no DPO, no LoRA — plan-04.
- No steering — plan-05.
- No human feedback — plan-06.
- No cost-tracking dashboard — stdout summary is sufficient.
- No retries on evaluator JSON parse failures. With `response_format=json_schema`
  this effectively cannot fail; if it does, we want the loud error.
- No concurrency. Sequential API calls keep budgeting simple and the cache
  race-free. Plan-04 can parallelize once costs scale.
- No Prometheus/structured logs. Print to stdout.

## Open Items (resolve during implementation)

1. **Rubric prompt wording.** First draft in the implementing commit; iterate
   once we see ~50 scored trajectories. Every wording change bumps
   `rubric_version`.
2. **Domain weighting.** Uniform for MVP. If certain domains produce
   consistently bad Qwen responses, re-weight in plan-03 or plan-04.
3. **JSON-schema strict mode vs. tolerant.** Start strict (`additionalProperties: false`).
   Relax only if we hit false rejections in practice.
4. **Seed file for reproducibility.** `run_feedback_loop.py --seed 42` seeds
   `random.shuffle` on the taxonomy; same seed → same domain order → fully
   deterministic run (combined with the cache).

## Handoff to plan-03

Once plan-02 is verified, `plan-03-reasoningbank.md` will consume
`data/trajectories.jsonl` and produce a memory of extracted reasoning lessons.
It will:
- Add `memory/lessons.py` — lesson extraction from a scored trajectory.
- Add `memory/store.py` — FAISS-backed vector store (sentence-transformers
  embeddings).
- Add `memory/retrieve.py` — k-NN lesson retrieval, system-prompt injection.
- Extend `cli.py` with `brainstorm ask "<prompt>" --use-memory`.
- Extend `run_feedback_loop.py` with `--use-memory`, so Phase-1 exit-criterion
  4 ("memory beats baseline on held-out") can be measured.

Plan-03 reuses, unchanged: `clients/openai.py`, `feedback/trajectories.py`,
`feedback/evaluator.py`, `eval/fixtures.py`, `model/interface.py`, `model/base.py`.
