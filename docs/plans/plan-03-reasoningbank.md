# Plan 03 — ReasoningBank Memory

_Parent plan: `docs/plans/master-plan.md` (Phase 1, Sub-plan 1.3)._
_Builds on: `plan-02-synthetic-feedback.md` (complete as of commit `f237e62`)._
_Authored: 2026-04-25._

## Context

Plan-02 left BrainstormLM with a working synthetic feedback loop: `gpt-4o-mini`
prompter + Qwen2.5-7B generator (via Together) + `gpt-4o-mini` JSON-schema
evaluator, all writing scored trials to `data/trajectories.jsonl`. Nothing yet
*uses* those scored trials to make the model better.

Plan-03 closes that gap by implementing **ReasoningBank** (per the Google
paper): a memory of *abstracted reasoning lessons* — not raw trajectories —
extracted from both successful and failed trials. At inference, the top-k most
relevant lessons (mix of success + failure) are retrieved by embedding
similarity and injected into the system prompt with clear labeling. The
`Trajectory` schema already reserves `memory_context_ids: list[str]` for this.

This is the last sub-plan in Phase 1. After it lands, we can satisfy
**Phase 1 Exit Criterion 4**: on the 50 held-out prompts, `--use-memory` must
beat the no-memory baseline on mean `creativity + logic` by a paired-bootstrap
margin of p < 0.1.

No training, no LoRA, no steering — those live in plan-04 / plan-05.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Vector store backend | **In-memory numpy + JSONL persistence** | Lesson count at MVP scale = hundreds-to-low-thousands; numpy `argpartition` over a (N, 1536) matrix is microseconds. No FAISS/Chroma deps. |
| Embedding model | **`text-embedding-3-small` (OpenAI)** | $0.02/Mtok, integrates with the existing `clients/openai.py` cache layer. Re-embedding all lessons is a cache hit on the second invocation, so reload is free. |
| Lesson extractor model | **`gpt-4o-mini`** | Consistent with plan-02's OpenAI-only stack. ~$0.0005/trajectory under the budgeted token usage. JSON-schema strict via existing `complete_json`. |
| Lesson persistence | **Append-only JSONL** at `data/memory/lessons.jsonl` | Mirrors the trajectory pattern. Re-extracting bumps `extraction_version` so old rows stay interpretable. |
| Embedding persistence | **None — re-derived from sha256 cache on load** | The cache layer already deduplicates; no second source of truth. |
| Retrieval mix | **k_success=3, k_failure=2** by default | Per the paper, anti-patterns matter as much as patterns. Tunable via `--k-success`/`--k-failure`. |

## Deliverables

Files to **create**:

```
src/brainstormlm/memory/__init__.py
src/brainstormlm/memory/lessons.py          # Lesson schema + LLM extraction
src/brainstormlm/memory/store.py            # Numpy-backed lesson store + JSONL
src/brainstormlm/memory/retrieve.py         # Top-k retrieval + system-prompt formatting
scripts/rebuild_memory.py                   # Re-extract lessons from trajectories.jsonl
scripts/eval_memory.py                      # Held-out paired eval: --use-memory vs baseline
tests/test_lessons_parse.py                 # Pure-parse tests for extraction JSON
tests/test_memory_store.py                  # Add/save/load + top-k correctness (mocked embed)
tests/test_memory_retrieve.py               # Mix of success+failure, system-prompt formatting
tests/test_openai_embed.py                  # Cache discipline for embed (no network)
```

Files to **extend**:

```
src/brainstormlm/clients/openai.py          # add embed(texts, model) -> np.ndarray, cached per text
src/brainstormlm/config.py                  # add embedding_model, lesson_extractor_model, memory_dir, lessons_path
src/brainstormlm/cli.py                     # add --use-memory / --k-success / --k-failure to `ask`
scripts/run_feedback_loop.py                # add --use-memory flag; populate memory_context_ids
pyproject.toml                              # add numpy dep
```

No changes to `feedback/trajectories.py` — `memory_context_ids` already exists.

## Lesson Schema (v1)

```python
# src/brainstormlm/memory/lessons.py
class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str                          # sha256(lesson + kind + source_trajectory_id)
    lesson: str                      # 1-3 sentence abstracted pattern
    kind: Literal["success", "failure"]
    domain: str
    source_trajectory_id: str
    extractor_model: str             # "gpt-4o-mini"
    extraction_version: int          # bump when extractor system prompt changes
    extracted_at: str                # ISO 8601 UTC
```

`id` is a deterministic content hash so re-extraction is idempotent and the
store can dedupe on append.

Rule: schema changes are **additive only** (same discipline as `Trajectory`).

## Extraction Contract

`memory/lessons.py::extract_lessons(t: Trajectory) -> list[Lesson]`
- Calls `openai_client.complete_json` with `model=settings.lesson_extractor_model`.
- System prompt explains the ReasoningBank framing: distill 1–3 generalizable
  reasoning patterns; success-kind for what worked; failure-kind for
  anti-patterns. Lessons must be domain-relevant but not bound to the prompt
  specifics.
- JSON schema:
  ```json
  {"lessons": [{"lesson": "string", "kind": "success|failure"}]}
  ```
  with `additionalProperties: false`, `1 ≤ len(lessons) ≤ 3`.
- The LLM gets `(prompt, response, creativity, logic, rationale, failure_modes)`
  as user-message context and decides which lessons are successes vs failures.
- `EXTRACTION_VERSION = 1`. Any wording change in the system prompt bumps it.

## Store Contract

`memory/store.py::LessonStore`
- `__init__(lessons_path: Path)` — does not load yet.
- `load() -> None` — read JSONL → `self.lessons: list[Lesson]`; call
  `openai_client.embed([l.lesson for l in lessons])` → `self.embeddings:
  np.ndarray (N, D)` L2-normalized.
- `add(lesson: Lesson) -> bool` — JSONL-append unless `lesson.id` already
  present; embed-and-stack the new row. Returns whether a new row was added.
- `add_many(lessons: list[Lesson]) -> int` — bulk; one batched embed call.
- `search(query: str, k: int, *, kinds: tuple[str, ...] | None = None) ->
  list[tuple[Lesson, float]]` — cosine via dot-product on L2-normalized
  vectors; optional `kinds` filter; returns descending-score top-k.

## Retrieval & Prompt Injection

`memory/retrieve.py`
- `retrieve(store: LessonStore, query: str, *, k_success: int = 3,
  k_failure: int = 2) -> list[Lesson]` — runs two `search` calls (one per kind)
  and concatenates. If a kind has fewer than k available, take what exists.
- `format_memory_block(lessons: list[Lesson]) -> str` — emits a labeled block:
  ```
  Relevant patterns from past brainstorms:

  Things that worked (apply when relevant):
    - <success lesson 1>
    - <success lesson 2>

  Things to avoid (anti-patterns from past failures):
    - <failure lesson 1>
  ```
- `augmented_system(base_system: str | None, lessons: list[Lesson]) -> str` —
  prepends the memory block to whatever base system was passed in (None →
  the memory block alone is the system prompt). Empty `lessons` → return
  `base_system` unchanged.

## OpenAI Client Extension

Add to `clients/openai.py`:

```python
def embed(
    texts: list[str],
    *,
    model: str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, Usage]:
    """Return (N, D) float32 L2-normalized embeddings; cached per (model, text)."""
```
- Cache key: `cache_key("openai_embed", model, text)` per individual text.
- Misses are batched into one API call; hits short-circuit.
- Returns matrix in input order, plus aggregated `Usage`.

## Config Additions

```python
embedding_model: str = "text-embedding-3-small"
lesson_extractor_model: str = "gpt-4o-mini"
memory_dir: Path = Field(default=Path("./data/memory"), alias="BRAINSTORMLM_MEMORY_DIR")
lessons_path: Path = Field(default=Path("./data/memory/lessons.jsonl"), alias="BRAINSTORMLM_LESSONS_PATH")
```

## CLI & Script Wiring

**`brainstorm ask`** (extend):
```
brainstorm ask "<prompt>" --use-memory --k-success 3 --k-failure 2
```
On `--use-memory`: instantiate `LessonStore(settings.lessons_path)`, `.load()`,
call `retrieve(...)`, build augmented system, pass to generator. Print the
response. Optional `--show-lessons` flag prints retrieved lessons to **stderr**
so stdout stays response-only.

**`scripts/run_feedback_loop.py`** (extend):
- Add `--use-memory` (default off) and `--k-success` / `--k-failure`.
- When on: load the store once before the loop; per trial, retrieve lessons,
  build augmented system, pass to `generator.generate_with_meta(..., system=...)`,
  store retrieved lesson `id`s in `Trajectory.memory_context_ids`.
- Trajectory schema unchanged — `memory_context_ids` is already a list field.

**`scripts/rebuild_memory.py`** (new):
```
python scripts/rebuild_memory.py \
    --trajectories data/trajectories.jsonl \
    --out data/memory/lessons.jsonl \
    --max-trajectories 1000 \
    --max-spend-usd 0.50 \
    --reset
```
- `--reset` truncates `lessons.jsonl` before extraction (otherwise: append).
- Per trajectory: `extract_lessons(t)` → `store.add_many(lessons)`.
- Idempotent without `--reset`: `add` skips already-known `lesson.id`s.
- Pre-flight cost estimate using `pricing.cost_of` with budgeted token counts;
  abort if estimate > `--max-spend-usd`. Running tally aborts mid-loop.

**`scripts/eval_memory.py`** (new — satisfies Phase 1 exit criterion 4):
```
python scripts/eval_memory.py --k-success 3 --k-failure 2 --seed 42
```
- For each of the 50 `HELD_OUT` prompts:
  1. Generate a response with no memory.
  2. Generate a response with `--use-memory`.
  3. Score both with the evaluator.
- Aggregate: mean `creativity + logic` for each arm.
- Paired bootstrap (10k resamples) on the per-prompt deltas; print the
  observed mean delta and a one-sided p-value for `H1: memory > baseline`.
- Append both arms' trajectories to a separate file
  (`data/eval/held_out_eval.jsonl`) so they don't pollute the training pool
  (per master-plan: held-out prompts are NEVER used for training/retrieval).

## Tests (TDD per CLAUDE.md)

1. **`test_openai_embed.py`** — pure cache test, no network.
   - Monkeypatch the OpenAI client's embeddings call; assert the second
     invocation with the same text hits the cache and skips the API.
   - Assert mixed hit/miss batches request only the missing texts.
2. **`test_lessons_parse.py`** — pure parse, no network.
   - Canned `{"lessons": [...]}` payload → `list[Lesson]` with stable IDs.
   - Schema rejects extras, rejects bad `kind`.
   - ID is deterministic given (lesson, kind, source_trajectory_id).
3. **`test_memory_store.py`** — uses a monkeypatched `embed()` that returns
   deterministic vectors per text (e.g. hash → seeded RNG).
   - Add 5 lessons → JSONL has 5 lines → reload → 5 Lessons + (5, D) matrix.
   - Re-adding the same lesson is a no-op (dedup on id).
   - `search(query, k=2, kinds=("failure",))` returns only failures.
   - Top-k order matches hand-computed cosine similarities.
4. **`test_memory_retrieve.py`**
   - `retrieve(k_success=2, k_failure=1)` returns 2 successes + 1 failure when
     both kinds are available; degrades gracefully when one kind is empty.
   - `format_memory_block` includes both labeled sections; empty lists yield
     an empty string (caller decides).

Each new module ships with its tests in the same commit per `CLAUDE.md`.

## Implementation Sequence (parallelizable via teammates)

The shared types (`Lesson` schema + `embed`) are dependencies for both halves
of the work, so they land on the main thread first; then two teammates fan
out in parallel; then the main thread integrates.

**Step 1 — main thread (sequential, ~10 min):**
1. Add `numpy` dep to `pyproject.toml`; `uv sync`.
2. Add config fields (`embedding_model`, `lesson_extractor_model`, `memory_dir`,
   `lessons_path`).
3. Implement `clients/openai.py::embed` with sha256 cache (per-text key,
   batched misses) and `tests/test_openai_embed.py`.
4. Define the `Lesson` Pydantic model + deterministic-id helper in
   `memory/lessons.py` (the extraction function is added by the extractor
   teammate).
5. Stub `memory/__init__.py` re-exporting `Lesson`, `LessonStore`, `retrieve`.

**Step 2 — fan out two teammates in parallel:**

- Teammate **`lessons-extractor`** owns:
  - `memory/lessons.py::extract_lessons` (system prompt, JSON schema, parse →
    `list[Lesson]` with deterministic IDs).
  - `tests/test_lessons_parse.py`.
  - Constraints: must use `extraction_version=1`; must pass
    `(prompt, response, creativity, logic, rationale, failure_modes)` to the
    extractor LLM; must not call the network in tests.

- Teammate **`memory-store`** owns:
  - `memory/store.py::LessonStore` (load/add/add_many/search).
  - `memory/retrieve.py::retrieve` + `format_memory_block` + `augmented_system`.
  - `tests/test_memory_store.py`, `tests/test_memory_retrieve.py`.
  - Constraints: monkeypatch `clients.openai.embed` in tests; never hit the
    network; preserve append-only JSONL discipline.

Both teammates import the already-landed `Lesson` schema, so their work is
truly independent.

**Step 3 — main thread (sequential, integration):**
1. Wire `--use-memory` / `--k-success` / `--k-failure` into `cli.py::ask`.
2. Wire same flags + `memory_context_ids` capture into
   `scripts/run_feedback_loop.py`.
3. Implement `scripts/rebuild_memory.py` (idempotent, cost-capped, `--reset`).
4. Implement `scripts/eval_memory.py` (paired bootstrap on 50 held-out
   prompts; writes to `data/eval/held_out_eval.jsonl`).
5. Run the verification gates below.
6. Commit per `CLAUDE.md` (imperative mood; tests ship with the code they
   cover; one logical change per commit — likely 4-5 commits: numpy+config,
   embed, lessons, store+retrieve, integration).

## Verification

All gates must pass before closing plan-03:

1. `uv run pytest tests/` — all old + new tests green (≥10 new tests).
2. **Live extraction smoke**: `uv run python scripts/rebuild_memory.py
   --max-trajectories 5 --max-spend-usd 0.05` produces 5–15 lessons in
   `data/memory/lessons.jsonl`. Second invocation is cache-served (cost ≈ $0,
   no new lessons because IDs already exist).
3. **`brainstorm ask` with memory**: `uv run brainstorm ask "How can a small
   team move faster without burning out?" --use-memory` runs end-to-end and
   prints a response (cache-served on the second call).
4. **Held-out eval (the Phase 1 exit gate)**: `uv run python
   scripts/eval_memory.py --k-success 3 --k-failure 2` reports a positive
   mean delta on `creativity + logic` for the memory arm, with paired
   bootstrap p < 0.10. If the gate fails, log the result and iterate on the
   extractor system prompt (bumping `extraction_version`) — landing the
   plumbing is enough to close this plan; passing the gate closes Phase 1.
5. Re-running `eval_memory.py` is fully cache-served — wall-clock under a few
   seconds, total cost reads $0.00.
6. `data/eval/held_out_eval.jsonl` exists and is disjoint from
   `data/trajectories.jsonl` (held-out prompts must never enter the training
   pool).

## Non-Goals

- No FAISS, Chroma, or sqlite — numpy is enough at MVP scale.
- No re-ranking / MMR — top-k cosine is sufficient for plan-03.
- No multi-turn — one prompt → one retrieval → one response.
- No incremental embedding updates across processes; re-embed on load is
  cheap and cache-served.
- No automated extractor-prompt iteration; if the held-out gate fails, the
  human (Keshav) iterates the system prompt and bumps `extraction_version`.
- No human feedback ingestion — that's plan-06.

## Open Items (resolve during implementation)

1. Lesson-extractor system prompt wording — first draft in the implementing
   commit; iterate once we see ~50 lessons. Every wording change bumps
   `extraction_version`.
2. Whether to pre-filter trajectories by score before extraction (e.g. skip
   middling rows where neither pattern is clear). Default: extract from all,
   let the LLM judge per-lesson kind. Revisit if the resulting bank is noisy.
3. Embedding model upgrade path — if recall is poor, swap to
   `text-embedding-3-large` via config; bumping the model invalidates cache
   but lessons.jsonl is unchanged.

## Handoff to plan-04

Once plan-03 is verified, `plan-04-rlhf-dpo.md` will consume the same
`data/trajectories.jsonl` (plus on-demand sibling samples) to mine
`(prompt, chosen, rejected)` preference pairs and train a LoRA via DPO. The
ReasoningBank from plan-03 stays live at inference time so plan-04 can
ablate (base, base+mem, trained, trained+mem) cleanly.
