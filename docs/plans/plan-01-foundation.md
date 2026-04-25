# Plan 01 — Foundation

_Parent plan: `~/.claude/plans/come-up-with-a-magical-wren.md` (Phase 1, Sub-plan 1.1)._
_Authored: 2026-04-24._

## Context

The `BrainstormLM` repo is effectively empty only `CLAUDE.md` + a fresh `git
init`). Phase 1 MVP needs a minimal working system: a `brainstorm ask "<prompt>"`
CLI that produces text from a pinned open-source base model, with a cached
closed-source LLM client wired up for use by later sub-plans. No feedback loop,
no memory, no training yet — just the scaffolding everything else layers onto.

This sub-plan deliberately **departs from the master plan's `transformers` +
local inference assumption**. We pin the base model behind a hosted OSS API
(Together AI) so Phase 1 has zero GPU/quantization setup. The `Generator`
protocol means Phase 3's steered local model drops in without touching
downstream code.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Base model host | **Together AI** | Has Qwen2.5 with simple REST API; no local GPU, no HF download. |
| Base model id | **`Qwen/Qwen2.5-7B-Instruct-Turbo`** | Strong creative-writing for its size; Apache-2.0; hosted on Together. |
| Python tooling | **uv** | Fast, single-tool venv + lockfile + install. |
| Judge LLM | **OpenAI** (GPT-4o family) | Strong rubric following + first-class `response_format=json_schema`. Specific model id pinned in plan-02. |
| Config loader | **`pydantic-settings`** | Type-safe env loading, one source of truth. |
| CLI framework | **Typer** | Minimal ceremony for small CLIs. |

## Deliverables

Files to create (all under `/Users/keshavrangan/BrainstormLM/`):

```
pyproject.toml                              # uv project + dependencies
README.md                                    # minimal, links to plan
.gitignore                                   # Python + data dirs
.env.example                                 # required env vars, no secrets
.python-version                              # pin Python 3.11
src/brainstormlm/__init__.py
src/brainstormlm/config.py                   # pydantic-settings Settings
src/brainstormlm/clients/__init__.py
src/brainstormlm/clients/cache.py            # sha256-keyed JSON cache
src/brainstormlm/clients/openai.py           # cached OpenAI client
src/brainstormlm/clients/together.py         # cached Together client
src/brainstormlm/model/__init__.py
src/brainstormlm/model/interface.py          # Generator protocol (swap point)
src/brainstormlm/model/base.py               # TogetherGenerator(Generator)
src/brainstormlm/cli.py                      # `brainstorm ask`
tests/__init__.py
tests/test_cache.py                          # cache hit/miss behavior
```

Dependencies (pyproject.toml):
- Runtime: `openai`, `together`, `typer`, `pydantic`, `pydantic-settings`, `python-dotenv`
- Dev: `pytest`, `pytest-asyncio`, `ruff`

**Explicitly NOT added yet** (deferred to later sub-plans):
- `transformers`, `torch`, `accelerate`, `bitsandbytes` → deferred to Phase 3 (steering) or Phase 2 (DPO training).
- `faiss-cpu`, `sentence-transformers`, `chromadb` → plan-03 (ReasoningBank).
- `trl`, `peft`, `datasets` → plan-04 (DPO).

## Step-by-Step Implementation

1. **Initialize the Python project.**
   - `uv init --package --python 3.11 BrainstormLM` is avoided because the repo already exists; instead, write `pyproject.toml` by hand, then `uv sync`.
   - Add `.python-version` with `3.11`.

2. **`.gitignore`** — standard Python + `.env` + `data/` + `__pycache__` + `.venv/`.

3. **`.env.example`:**
   ```
   OPENAI_API_KEY=sk-...
   TOGETHER_API_KEY=...
   BRAINSTORMLM_DATA_DIR=./data
   BRAINSTORMLM_CACHE_DIR=./data/cache
   ```

4. **`config.py`** — a single `Settings(BaseSettings)` class with env-loaded
   fields (API keys, data/cache dirs, base model id, generation defaults).
   One global `get_settings()` accessor; no scattered `os.environ`.

5. **`clients/cache.py`** — tiny helper: `cache_key(*parts) -> str` (sha256
   hexdigest over stable JSON-serialized parts) + `read_cache(key) -> str |
   None` + `write_cache(key, value)`. Backing store: one JSON file per key
   under `data/cache/<first-two-hex>/<rest>.json`. Simple, inspectable,
   git-ignorable.

6. **`clients/openai.py`** — thin wrapper around the `openai` SDK:
   ```python
   def complete(system: str, messages: list[dict], model: str,
                max_tokens: int = 1024, temperature: float = 1.0) -> str
   ```
   Before each call: compute cache key over all args, return cached value if
   hit. On miss: call the API, write cache, return.

7. **`clients/together.py`** — same shape, wrapping the `together` SDK's
   `ChatCompletions.create`. Same cache discipline. Accepts a `model` arg,
   defaulted via config.

8. **`model/interface.py`** — `Generator` protocol:
   ```python
   class Generator(Protocol):
       def generate(self, prompt: str, *, system: str | None = None,
                    temperature: float = 1.0, max_tokens: int = 1024) -> str: ...
   ```
   This is the swap point. Phase 3 will add `SteeredGenerator`.

9. **`model/base.py`** — `TogetherGenerator(Generator)` backed by
   `clients/together.py`. Takes a model id in its constructor; defaults to
   config.

10. **`cli.py`** — Typer app:
    - `brainstorm ask "<prompt>"` → prints response
    - `brainstorm ask "<prompt>" --temperature 1.3` → override
    - `brainstorm ask "<prompt>" --no-cache` → bypass cache

11. **Register the CLI** — add `[project.scripts] brainstorm = "brainstormlm.cli:app"`
    to `pyproject.toml`.

12. **`tests/test_cache.py`** — one test: same args → cache hit, different
    args → cache miss. No network calls.

13. **First git commit** — after the above lands and `brainstorm ask` runs
    end-to-end, commit with a message per `CLAUDE.md`'s "good git practices".

## Verification

All of these must pass before closing plan-01:

1. `uv sync` succeeds on a fresh clone.
2. `uv run pytest tests/` exits 0.
3. `uv run brainstorm ask "name three unusual uses for a stapler"` returns
   non-empty text from Qwen2.5 in under ~10 seconds.
4. **Cache proof**: re-running the exact same command prints the cached
   response with zero Together API calls (verified by inspecting
   `data/cache/` populated after the first call, and by the second call
   completing in <100ms).
5. `uv run brainstorm ask "test" --no-cache` bypasses the cache.
6. `brainstorm --help` shows the subcommand cleanly.

## Non-Goals for This Sub-Plan

- No evaluator, no prompter, no trajectories — those are plan-02.
- No retrieval, no memory — plan-03.
- No training — plan-04.
- No local model inference — plan-05.
- No web UI — plan-06.
- No streaming responses — MVP is non-streaming; can add in plan-06.
- No rate limiting, no `--max-spend-usd` guardrails — those matter at
  feedback-loop scale (plan-02) where we loop hundreds of calls. For single
  CLI calls, the cache is sufficient cost protection.

## Open Items (resolve during implementation)

1. **Exact Together model slug** — `Qwen/Qwen2.5-7B-Instruct-Turbo` is the
   first candidate; if Together has renamed or deprecated it by 2026-04-24,
   pick the nearest replacement and note the swap in the commit message.
2. **Default generation params** — start with `temperature=1.0`,
   `max_tokens=1024`, `top_p=0.95`. Revisit once we have evaluator feedback
   in plan-02.
3. **Cache schema evolution** — cache key currently includes all gen params,
   so changing defaults invalidates history. Acceptable for MVP; revisit if
   cache hit rate matters.

## Handoff to plan-02

Once plan-01 is verified, plan-02 (`plan-02-synthetic-feedback.md`) will:
- Add `feedback/prompter.py` — OpenAI generates brainstorm prompts by domain.
- Add `feedback/evaluator.py` — OpenAI scores (creativity, logic) with rubric (JSON schema).
- Add `feedback/trajectories.py` — append-only JSONL store.
- Add `scripts/run_feedback_loop.py` — orchestrate N trajectories.
- Add `eval/fixtures.py` — frozen held-out eval prompts.

plan-02 reuses, unchanged: `config.py`, `clients/openai.py`,
`model/interface.py`, `model/base.py`.
