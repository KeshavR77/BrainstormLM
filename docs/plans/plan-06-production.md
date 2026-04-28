# Plan 06 — Production System

_Parent plan: `docs/plans/master-plan.md` (Phase 4, Sub-plan 4.1)._
_Builds on: `plan-05-steering-integration.md` (steering plumbing landed at the
current `main` head, plus plan-04's adapter + ablation harness)._
_Authored: 2026-04-27._

## Context

Phases 1–3 are closed. The pipeline now runs end-to-end against synthetic
prompts: a Together-served base, a Qwen2.5-7B HF generator with optional PEFT
LoRA adapter (plan-04), and a steering-vector wrapper (plan-05) — all behind a
single `Generator` protocol, all routed by
`make_generator(adapter_path, steering_path=...)`
(`src/brainstormlm/model/factory.py:15`). The 4-arm ablation harness
(`src/brainstormlm/eval/harness.py:80`) reports paired-bootstrap p-values
across `base / base+mem / trained / trained+mem`, with a `--steered` toggle
that renames every arm in place.

What is *not* there yet: a way for a real human to use the model, leave
feedback, and have that feedback feed the next training run. Every piece of
the master plan's Phase 4 — serving, UI, human feedback ingestion, continuous
training, observability, safety — is missing.

Plan-06 closes that gap. It ships:

1. A FastAPI server wrapping the existing `Generator` protocol with streaming
   `/ask`, structured `/feedback`, and `/healthz`/`/metrics` endpoints. No
   model-loading code is duplicated — the server calls
   `make_generator(adapter_path, steering_path=...)` per request, gated by an
   A/B router.
2. A **human-feedback JSONL** (`data/feedback_human.jsonl`) that mirrors the
   `Trajectory` schema with `source: "human" | "synthetic"` and a
   `human_rating` block, so the existing trajectory consumers (lesson
   extractor, preference miner) can read it transparently.
3. **A/B routing** — deterministic per-session split between two adapters
   (`adapter_a`, `adapter_b`) so the live system can compare two trained
   variants on real human feedback rates without forking the harness.
4. **Continuous training** — a scheduled `scripts/continuous_train.py` that
   re-mines preference pairs from the union of synthetic + human feedback,
   triggers a Modal DPO run (reusing `scripts/train_dpo.py`), runs the 4-arm
   ablation, and **promotes the new adapter behind a flag** by atomically
   updating a `data/adapters/promoted.json` pointer iff the held-out gate
   passes. This is the file equivalent of a feature flag — no
   external feature-flag service.
5. **Safety + observability** — a memory-context scrubber that strips
   delimiter-like sequences out of retrieved lessons before they hit the
   system prompt (the only attacker-controlled-ish text path), an in-process
   token-bucket rate limiter, and structured per-request logs (latency,
   arm, cache hits, evaluator score when available). Evaluator-vs-human
   agreement is computed by `scripts/agreement.py` over rows that have both a
   human rating and an evaluator score.
6. A `brainstorm serve` CLI command + a thin static frontend in `frontend/`
   (vanilla HTML/JS, no build step) so a non-Keshav user can exercise the
   loop end-to-end against a locally-running or Modal-hosted server. Vercel /
   Modal-Web deployment is a deferred live gate.

After plan-06 lands, the master plan's Phase 4 exit criteria become
**operational** — every piece of plumbing exists and is unit-tested. The
**live** outcomes (a non-Keshav user finishing a session, a scheduled
retraining run end-to-end on real feedback) are deferred gates the same way
plan-04 deferred its Modal A100 training run and plan-05 deferred its live
steering-vector derivation. This plan ships the plumbing + tests; the
operational milestones are one-shot Keshav-driven runs once the plumbing is
green.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Web framework | **FastAPI + uvicorn** | Async-native, `StreamingResponse` is a one-liner, Pydantic-native (we're already on Pydantic v2). Avoids the heavier Streamlit footprint; Streamlit can sit on top of the API later if Keshav wants a dashboard. |
| Frontend | **Static HTML/JS in `frontend/`**, served by FastAPI on `/` | One-page, ~150 lines, no Node toolchain. Calls the same `/ask` SSE endpoint Keshav would call from `curl`. Vercel/Next.js is a master-plan suggestion, not a contract; this avoids a JS build environment for plan-06. |
| Human-feedback persistence | **Append-only JSONL** at `data/feedback_human.jsonl`, schema = `Trajectory` superset with `source: Literal["human","synthetic"]` (default `"synthetic"`) and `human_rating: HumanRating \| None` | Matches the master plan's "feedback_human.jsonl with the same schema shape as synthetic trajectories" line item. Additive change to the existing `Trajectory` schema (same discipline as plan-04's `adapter_path` and plan-05's `steering_path`) — old rows still validate. |
| Human rating shape | **`HumanRating(thumbs: Literal["up","down"], critique: str \| None, rated_at: str)`** with optional 1–5 numeric `rating` | Thumbs are the cheapest UI-friendly signal; the optional 5-star + free-text critique gives preference-mining and rubric-recalibration enough information without overloading the user. |
| Preference mining over human feedback | **Pair within prompt only**: pick top-thumbs-up vs top-thumbs-down per prompt where both arms exist; if not, fall back to the existing synthetic miner (plan-04) on that prompt | Keeps the new code small, lets plan-04's `mine_pairs` stay unchanged for synthetic data. Human pairs ride the same `PreferencePair` schema (no `source` field needed; the `sibling_run_id` is set to `human-<feedback_id>`). |
| Continuous-training cadence | **On-demand via `scripts/continuous_train.py`** — operator runs it; a cron is the deployment concern | The master plan suggested cron / GitHub Actions / Modal; we keep it operator-triggered for plan-06 so the Phase 4 exit criterion ("scheduled training job ran at least once end-to-end") is closed by a documented manual run. Modal `schedules.cron` is a one-line follow-up if it is needed. |
| Adapter promotion gate | **Held-out total ≥ current `promoted.json` total + 0.10** AND **paired-bootstrap p < 0.20** | Bigger margin than plan-03/04's eval gate (p < 0.10) because we're promoting a live-traffic artifact, not just measuring. The +0.10 hard floor prevents promoting a statistically-significant-but-tiny gain. |
| Promotion atomicity | **Write `promoted.json.tmp` then `os.replace`** | POSIX-atomic on the same filesystem; no half-promoted state if the operator Ctrl-Cs mid-write. |
| A/B split | **Deterministic by `sha256(session_id) mod 100`** with operator-set integer `traffic_a_pct` (0..100) in `data/ab_config.json` | Stable per session — the same user sees the same arm across requests until config changes — and trivially reproducible from logs. Defaults to 50/50 when both adapters are set; falls back to single-arm when only one is. |
| Rate limiting | **In-process token bucket** keyed by client IP, default 30 requests / 60s | Adequate for "small user counts" per master-plan non-goals. Operators behind a real load balancer should defer to the LB's limiter; the bucket is a guardrail, not a security boundary. |
| Memory-context safety | **Strip lessons containing the chat-template control tokens or repeated `system:`-style prefixes**; truncate each lesson to 280 chars | Lessons are LLM-generated by a model we control, not arbitrary user input, but they ride into the system prompt; treating them as semi-trusted is cheap insurance against an injected prompt that someone might craft via the human-critique field that later gets re-extracted into a lesson. |
| Content policy on user prompts | **Allowlist by length (≤2000 chars) + reject control characters** | Bare minimum — we are a brainstorming sandbox, not a chatbot. Aggressive filtering would defeat the purpose. |
| Structured logging | **JSON-Lines to `data/logs/server.jsonl`** with `request_id, ts, route, latency_ms, arm, generator_cached, status` | Plain JSON over a real logging stack — analysis happens in `scripts/agreement.py` and ad-hoc `jq`. Master plan's "Prometheus" line is a non-goal for plan-06. |
| Evaluator-vs-human agreement | **`scripts/agreement.py`** computes Pearson + signed agreement (thumbs-up ↔ creativity+logic ≥ 7) over rows with both signals | Reuses the existing evaluator; one script, one summary JSON. Closes Phase 4 exit criterion 3. |
| Schema additivity | **`Trajectory` gets `source` and `human_rating` defaulted to `"synthetic"` / `None`**; old rows still validate | Same discipline as plan-04 / plan-05. `schema_version` stays at 1. |

## Deliverables

Files to **create**:

```
src/brainstormlm/api/__init__.py
src/brainstormlm/api/server.py                # FastAPI app + endpoints
src/brainstormlm/api/routing.py               # A/B router + ABConfig schema
src/brainstormlm/api/safety.py                # Memory scrubber + content policy + rate limiter
src/brainstormlm/api/logging_.py              # Structured-JSONL request logger
src/brainstormlm/api/schemas.py               # Request/response Pydantic models
src/brainstormlm/feedback/human.py            # HumanRating schema + append helper
src/brainstormlm/training/promotion.py        # promoted.json read/write + gate logic
frontend/index.html                           # Static UI (single file, no build)
frontend/app.js                               # SSE client + thumbs/critique submit
frontend/style.css                            # Minimal styling
scripts/serve.py                              # Convenience uvicorn launcher
scripts/continuous_train.py                   # Re-mine → train → eval → promote
scripts/agreement.py                          # Evaluator-vs-human agreement metric
docs/runbooks/garbage_output.md               # Incident runbook 1
docs/runbooks/training_loop_on_bad_data.md    # Incident runbook 2
tests/test_api_server.py                      # Endpoint plumbing (TestClient)
tests/test_api_routing.py                     # A/B determinism + edge cases
tests/test_api_safety.py                      # Scrub + rate limit + content policy
tests/test_human_feedback.py                  # HumanRating + JSONL round-trip
tests/test_promotion.py                       # Atomic promote + gate logic
tests/test_continuous_train.py                # Pipeline orchestration with stubs
tests/test_agreement.py                       # Pearson + signed agreement math
```

Files to **extend**:

```
src/brainstormlm/config.py                   # add api_host, api_port, ab_config_path,
                                             # promoted_adapter_path, feedback_human_path,
                                             # rate_limit_*, server_log_path
src/brainstormlm/cli.py                      # add `brainstorm serve` command
src/brainstormlm/feedback/trajectories.py    # add source: Literal["human","synthetic"]
                                             # = "synthetic", human_rating: HumanRating | None = None
src/brainstormlm/training/preferences.py     # add mine_human_pairs(rows, ...) -> list[PreferencePair]
.env.example                                 # document new BRAINSTORMLM_* vars
pyproject.toml                              # add fastapi, uvicorn[standard], httpx (test dep)
```

No changes to `feedback/evaluator.py`, `feedback/prompter.py`, `memory/*`,
`model/*`, `eval/harness.py`, or `eval/stats.py`. They're consumed unchanged.

## Trajectory Schema (additive, plan-06)

```python
# src/brainstormlm/feedback/trajectories.py (extend)

class HumanRating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thumbs: Literal["up", "down"]
    rating: int | None = Field(default=None, ge=1, le=5)  # optional 5-star
    critique: str | None = None
    rated_at: str  # ISO 8601 UTC

class Trajectory(BaseModel):
    # ... existing fields unchanged ...
    source: Literal["human", "synthetic"] = "synthetic"
    human_rating: HumanRating | None = None
```

Default `"synthetic"` / `None` keeps every plan-02..plan-05 row valid on
re-load. `schema_version` stays at 1.

## HumanFeedback Persistence

`src/brainstormlm/feedback/human.py`:

```python
def append_human(t: Trajectory, path: Path | None = None) -> None: ...
def read_all_human(path: Path | None = None) -> Iterator[Trajectory]: ...
def attach_rating(
    trajectory_id: str,
    rating: HumanRating,
    *,
    path: Path | None = None,
) -> Trajectory:
    """Find the row with matching id, write a new row with the rating attached,
    and return it. Original row is NOT mutated — append-only discipline."""
```

Storage: append-only JSONL at `data/feedback_human.jsonl`. The same file is
the unified human-feedback log; an in-memory index keyed by `id` handles
"attach a rating to the trajectory we just generated" without rewriting the
file.

## API Contracts

### `GET /healthz`

Returns `{"ok": true, "promoted_adapter": "<path|null>", "promoted_steering": "<path|null>"}`.

### `POST /ask` (Server-Sent Events)

Request body:

```json
{
  "prompt": "string",
  "session_id": "string",
  "use_memory": true,
  "k_success": 3,
  "k_failure": 2,
  "temperature": 1.0,
  "max_tokens": 600
}
```

Response: `text/event-stream`. The server emits, in order:

1. `event: meta\ndata: {"trajectory_id": "...", "arm": "a|b", "adapter_path": "..."}\n\n`
2. One `event: token\ndata: {"text": "..."}\n\n` per chunk (the underlying
   `Generator.generate` is non-streaming today; the server simulates streaming
   by chunking the final response into ~50-token windows so the UI feels
   alive — switching to true streaming is a `model/hf.py` follow-up that
   doesn't change the wire protocol).
3. `event: done\ndata: {"trajectory_id": "...", "creativity": null, "logic": null}\n\n`
   (creativity/logic are filled by an async evaluator pass when
   `evaluate_inline=true`; default off so /ask stays fast).

The trajectory is persisted to `data/feedback_human.jsonl` immediately on
`done`, with `source="human"`, `human_rating=None`, and the chosen arm's
`adapter_path` / `steering_path` populated.

### `POST /feedback`

Request body:

```json
{
  "trajectory_id": "string",
  "thumbs": "up" | "down",
  "rating": 1..5 | null,
  "critique": "string | null"
}
```

The server calls `attach_rating(trajectory_id, HumanRating(...))` and returns
`{"ok": true}`. Idempotent: re-submitting the same `(trajectory_id, thumbs)`
appends a new row but logs a warning so duplicate-feedback cases are
inspectable.

### `GET /metrics`

Returns aggregated counts read from the JSONL log: total requests, requests
in last 60s, mean latency by arm, cache hit rate. Plain JSON, not Prometheus.

## A/B Routing Contract

`src/brainstormlm/api/routing.py`

```python
class ABConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adapter_a: Path | None = None
    adapter_b: Path | None = None
    steering_a: Path | None = None
    steering_b: Path | None = None
    traffic_a_pct: int = Field(default=50, ge=0, le=100)

class ABResolution(BaseModel):
    arm: Literal["a", "b"]
    adapter_path: Path | None
    steering_path: Path | None

def load_ab_config(path: Path | None = None) -> ABConfig: ...

def resolve_arm(session_id: str, cfg: ABConfig) -> ABResolution: ...
```

Behavior:

- Hash: `int(sha256(session_id).hexdigest()[:8], 16) % 100`. Stable per
  session.
- `< traffic_a_pct` → arm `a`; otherwise arm `b`.
- If only `adapter_a` is configured, the router degrades to single-arm —
  every request gets `a`, regardless of the percentage. Same for
  `adapter_b`-only.
- If neither is configured, both arms resolve to `(None, None)` so the
  request flows through `TogetherGenerator` (Phase-1 default).

`load_ab_config` reads JSON from `data/ab_config.json` (configurable via
`settings.ab_config_path`); a missing file means "default config" (no
adapters, both arms = base).

## Promotion Gate Contract

`src/brainstormlm/training/promotion.py`

```python
class Promoted(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adapter_path: str | None = None
    steering_path: str | None = None
    held_out_total: float | None = None
    promoted_at: str | None = None
    run_id: str | None = None

def read_promoted(path: Path) -> Promoted: ...

def write_promoted_atomic(promoted: Promoted, path: Path) -> None:
    """Writes to <path>.tmp then os.replace(<path>.tmp, <path>)."""

def promote_if_better(
    *,
    candidate_run_id: str,
    candidate_adapter: Path,
    candidate_summary: dict,   # output of eval/harness.summarize
    current_promoted: Promoted,
    min_improvement: float = 0.10,
    max_p_value: float = 0.20,
) -> tuple[bool, str]:
    """
    Return (promoted: bool, reason: str). The reason is a one-line message
    suitable for stdout + log.
    """
```

Decision rule:

1. Compute `candidate_total = summary["trained+mem"].mean_total` (fall back
   to `trained` if `+mem` arm is missing).
2. If `current_promoted.held_out_total is None` → promote unconditionally;
   reason = "first promotion".
3. If `candidate_total - current.held_out_total < min_improvement` → reject;
   reason = "improvement < threshold".
4. If `summary["trained+mem"].p_value > max_p_value` → reject; reason =
   "p-value too high".
5. Otherwise → promote; reason = "passed gate".

## Continuous Training Pipeline

`scripts/continuous_train.py` — single CLI:

```
uv run python scripts/continuous_train.py \
    --max-spend-usd 5.00 \
    --max-modal-usd 30.00 \
    --max-pairs 2000 \
    --skip-mining          # use existing data/preferences.jsonl as-is
    --skip-training        # reuse the most recent adapter for ablation
    --dry-run              # plan-only, no spend
```

Flow:

1. **Aggregate human feedback** into a temporary mining-input file:
   `data/feedback_human.jsonl` filtered to rows with `human_rating not None`.
2. **Mine pairs** from the union (synthetic siblings + human-rated rows):
   - Synthetic side: existing `scripts/sample_siblings.py` is unchanged;
     this script imports and reuses it programmatically when `--n-prompts > 0`.
   - Human side: new `mine_human_pairs(rows)` — for each unique `prompt`,
     pair the highest thumbs-up response (and if multiple, highest `rating`
     and longest critique tiebreak) against the highest thumbs-down response.
   - Merge both pair lists into `data/preferences.jsonl` (append-only,
     dedup by `id`).
3. **Train** by spawning `scripts/train_dpo.py` as a subprocess (or its
   `_run_training` callable when `--local`). Result: `data/adapters/<run_id>/`.
4. **Eval** — call `scripts/eval_ablation.py`'s harness against the new
   adapter, captured into `data/eval/ablation_<run_id>_summary.json`.
5. **Promote-or-not** — `promote_if_better(...)` writes
   `data/adapters/promoted.json` atomically iff the gate passes; otherwise
   logs the reason and exits 0. The `Generator` factory is unaware of
   promotion — the server reads `promoted.json` on each request (cheap; the
   path resolves a few-byte file).
6. **Print summary** — pair counts, training time, ablation table, gate
   decision.

The script is idempotent on `--dry-run` and respects the existing
`--max-spend-usd` / `--max-modal-usd` discipline.

## Evaluator-vs-Human Agreement Contract

`scripts/agreement.py`:

```
uv run python scripts/agreement.py --out data/eval/agreement_<ts>.json
```

For every row in `data/feedback_human.jsonl` with both an evaluator score
(non-null `creativity` + `logic`) and a `human_rating`:

- **Signed agreement**: `thumbs_up_iff_score_high` — % of rows where
  `(thumbs == "up") == (creativity + logic >= 7)`.
- **Pearson** between `creativity + logic` and a numeric encoding of the
  rating (`thumbs_up=1, thumbs_down=0`; if `rating` is set, use it).
- **Sample size** + per-domain breakdown.

Output is JSON; the script also prints a one-line summary to stdout. This
closes Phase 4 exit criterion 3.

## Safety Contract

`src/brainstormlm/api/safety.py`:

```python
def scrub_lesson(text: str, *, max_len: int = 280) -> str:
    """Remove role-prefix-looking lines and chat-template control tokens; truncate."""

def scrub_lessons(lessons: list[Lesson]) -> list[Lesson]:
    """Return a new list with each lesson.lesson scrubbed."""

def validate_user_prompt(prompt: str, *, max_len: int = 2000) -> None:
    """Raise ValueError on policy violations: too long, control chars, empty."""

class TokenBucket:
    def __init__(self, capacity: int, refill_per_sec: float) -> None: ...
    def take(self, key: str) -> bool: ...

class RateLimitMiddleware:
    """FastAPI middleware that 429s on bucket exhaustion."""
```

Scrubber rules (per-line `text.splitlines()` over each lesson):

1. Drop lines that start with case-insensitive `system:`, `assistant:`,
   `user:`, `<|im_start|>`, `<|im_end|>`, `<|system|>`, etc.
2. Replace any of the above tokens appearing inline with their literal
   visible representation (`[SYSTEM]`).
3. Truncate to `max_len` chars; the lesson keeps its original `id`/`kind`/etc.

The user-prompt validator is intentionally minimal: master plan flags
"content policy on user prompts" as a Phase-4 line item; for a sandbox we
keep the bar at "no obviously broken inputs" and let the evaluator + the
user's good faith do the rest.

## Tests (TDD per CLAUDE.md)

All tests must pass without GPUs and without network calls.

1. **`tests/test_api_server.py`** — uses `fastapi.testclient.TestClient` with
   stubbed generators (override the `make_generator` dependency).
   - `GET /healthz` returns `200` with `ok: true` and the current promoted
     adapter (None when `data/adapters/promoted.json` is missing).
   - `POST /ask` with a small fake generator yields a meta event, ≥1 token
     event, and a done event in order; the `trajectory_id` is the same in
     meta and done.
   - `POST /ask` writes a row to `data/feedback_human.jsonl` with
     `source="human"`, `human_rating is None`.
   - `POST /feedback` for an existing trajectory id appends a new row whose
     `human_rating.thumbs == "up"` (or "down").
   - `POST /feedback` for a missing trajectory id returns `404`.
   - `POST /ask` with a 5000-char prompt returns `400` (`validate_user_prompt`).

2. **`tests/test_api_routing.py`** — pure-Python A/B logic.
   - `resolve_arm` is deterministic: same `session_id` → same arm across 100
     calls.
   - At `traffic_a_pct=0`, every session goes to `b`; at `100`, every session
     goes to `a`.
   - Only `adapter_a` configured → every session resolves to arm `a`
     regardless of percentage (degraded mode).
   - Empty config → both arms resolve to `(None, None, None, None)`.
   - Distribution: 1000 random session ids at `traffic_a_pct=30` → arm `a`
     fraction is within `[0.25, 0.35]` (not statistically tight, just sanity).

3. **`tests/test_api_safety.py`**
   - `scrub_lesson` strips `system:` and `<|im_start|>`-prefixed lines;
     truncates over-long input; preserves benign text unchanged.
   - `scrub_lessons` is a list-of-lessons wrapper; `id`/`kind`/`domain`
     unchanged; only `lesson` text is rewritten.
   - `validate_user_prompt` raises on empty, on `>2000` chars, and on a
     prompt containing `\x00`/`\x07`.
   - `TokenBucket` allows N back-to-back takes at capacity N, then refuses
     the (N+1)th, then allows again after enough refill ticks (use a
     monkeypatched clock).

4. **`tests/test_human_feedback.py`**
   - Round-trip: `append_human(t)` then `read_all_human()` returns equal
     objects.
   - `attach_rating` with a missing id raises `KeyError`.
   - `attach_rating` writes a NEW row (file line count increments by 1; the
     original row is not rewritten — we open in append mode only).
   - Rating-attached row has `source="human"`, `human_rating.thumbs == "up"`.

5. **`tests/test_promotion.py`**
   - Round-trip: `write_promoted_atomic(p, path)` then
     `read_promoted(path)` returns equal `Promoted`.
   - Atomicity: monkeypatch `os.replace` to raise mid-write; assert the old
     `promoted.json` is unchanged.
   - `promote_if_better` against `current.held_out_total = None` always
     promotes (first-run case).
   - Below threshold (`+0.05 < 0.10`) → does not promote.
   - Above threshold + low p-value → promotes.
   - Above threshold + high p-value → does not promote.

6. **`tests/test_continuous_train.py`** — orchestration with stubs (no
   network, no Modal).
   - Stub the four phases (`mine_human_pairs`, `sample_siblings`,
     `train_dpo._run_training`, `run_ablation`) via dependency injection;
     assert each is called once in order.
   - `--skip-mining` skips both miners; `--skip-training` reuses the most
     recent adapter pointer; `--dry-run` makes none of them get called.
   - The promoted-pointer is updated iff the (stubbed) ablation summary
     clears the gate.

7. **`tests/test_agreement.py`** — pure math.
   - 5 hand-crafted rows: 3 thumbs-up with creativity+logic ≥ 7, 2
     thumbs-down with creativity+logic < 7 → agreement = 1.0.
   - 4 mismatched rows → agreement = 0.0.
   - Pearson on a perfectly correlated fixture returns ≈ 1.0.

8. **`tests/test_trajectories.py`** (extend the existing file with one
   test): a row with `source="human"` and a non-trivial `HumanRating`
   round-trips through `append`/`read_all`; an old row without these fields
   still validates (back-compat).

Each test file ships in the same commit as the module it covers, per
`CLAUDE.md`.

## Implementation Sequence

The work fans out in three groups after the schema lands. Land Step 1
sequentially; the three substeps in Step 2 can be done in any order; Step 3
integrates.

**Step 1 — schemas + config + safety primitives (~25 min, sequential):**

1. Add deps to `pyproject.toml` (`fastapi`, `uvicorn[standard]`, `httpx`
   in dev). Run `uv sync`.
2. Add config fields (`api_host`, `api_port`, `feedback_human_path`,
   `ab_config_path`, `promoted_adapter_path`, `server_log_path`,
   `rate_limit_capacity`, `rate_limit_refill_per_sec`). Defaults below.
3. Extend `feedback/trajectories.py` with `HumanRating`, `source`, and
   `human_rating` fields (additive defaults). Extend
   `tests/test_trajectories.py` with the round-trip.
4. Implement `feedback/human.py` (`append_human`, `read_all_human`,
   `attach_rating`). Add `tests/test_human_feedback.py`.
5. Implement `api/safety.py` (`scrub_lesson`, `validate_user_prompt`,
   `TokenBucket`, `RateLimitMiddleware`). Add `tests/test_api_safety.py`.

**Step 2 — three independent strands:**

- **Strand `serving`** owns:
  - `api/schemas.py` (request/response Pydantic models).
  - `api/server.py` (`FastAPI` app, `/healthz`, `/ask`, `/feedback`,
    `/metrics`, dependency-overridable `make_generator` factory).
  - `api/logging_.py` (structured per-request logger).
  - `api/routing.py` (`ABConfig`, `resolve_arm`, `load_ab_config`).
  - `tests/test_api_server.py`, `tests/test_api_routing.py`.
  - Constraint: every test uses dependency-overrides on the FastAPI app to
    stub the generator; no real model loads.

- **Strand `promotion`** owns:
  - `training/promotion.py` (`Promoted`, atomic write, gate logic).
  - `training/preferences.py::mine_human_pairs` extension.
  - `tests/test_promotion.py`, extension to `tests/test_preferences_*.py`
    if needed (likely just a new test file `tests/test_preferences_human.py`
    — author judges).
  - Constraint: tests run with `tmp_path`, no real adapters touched.

- **Strand `pipeline-and-agreement`** owns:
  - `scripts/continuous_train.py` (orchestrator with dependency injection
    for the four stub-able phases).
  - `scripts/agreement.py` (Pearson + signed agreement, JSON output).
  - `tests/test_continuous_train.py`, `tests/test_agreement.py`.
  - Constraint: orchestration tests stub everything heavy; agreement test
    is pure math over hand-crafted rows.

**Step 3 — main thread (sequential, integration):**

1. Wire `brainstorm serve` into `cli.py` (uvicorn launcher; reads
   `settings.api_host`/`api_port`).
2. Add `scripts/serve.py` as the explicit `python -m`-friendly entrypoint.
3. Author `frontend/{index.html,app.js,style.css}` — single page, fetches
   `/ask` via `EventSource`, posts to `/feedback`. Mount via FastAPI
   `StaticFiles` at `/`.
4. Write the two runbooks under `docs/runbooks/`.
5. Run all verification gates below. Commit per `CLAUDE.md` (imperative
   mood; tests ship with code; ~5 commits: schemas+human-store, safety,
   server+routing, promotion+continuous, frontend+CLI+runbooks).

## Verification

All gates must pass before closing plan-06:

1. `uv run pytest tests/` — all old + new tests green (≥18 new tests).
2. **Schema additivity smoke**: load every existing
   `data/trajectories.jsonl` row with the extended `Trajectory`. Each row
   reports `source == "synthetic"`, `human_rating is None`. Pre-existing
   rows MUST NOT need a migration.
3. **API smoke (TestClient, stubbed generator)**:
   `tests/test_api_server.py` covers — no extra manual run needed.
4. **API smoke (live, optional)**:
   `uv run brainstorm serve --port 8000` then
   `curl -N -X POST :8000/ask -d '{"prompt":"design a coffee cup", "session_id":"keshav"}'`
   yields meta → token(s) → done events; the trajectory id appears in
   `data/feedback_human.jsonl`. This is operator-driven and **not part of
   the automated suite** because it spawns a uvicorn server.
5. **Frontend smoke**: open `http://localhost:8000/` in a browser, type a
   prompt, watch tokens stream in, click a thumbs button — a new row with
   `human_rating` lands in the JSONL. Operator-driven.
6. **Continuous-train dry-run**:
   `uv run python scripts/continuous_train.py --dry-run` exits 0 and prints
   the planned phases without spending a cent.
7. **Continuous-train live (deferred — Phase 4 exit criterion 2)**:
   one operator-driven run from real human feedback through Modal training
   to a promoted adapter pointer, with the ablation table archived under
   `data/eval/`. Same shape as plan-04's gate-5 deferral. The plumbing +
   tests that ship in plan-06 are sufficient to close the plan; the live
   run is a one-shot follow-up that produces the artifact.
8. **Agreement smoke**: `uv run python scripts/agreement.py` over a
   hand-injected fixture in `data/feedback_human.jsonl` reports a numeric
   Pearson + signed agreement. Operator-driven; tested via
   `tests/test_agreement.py` for math.
9. **Re-running the test suite is fully deterministic** — no network, no
   GPU, no temp files left behind.
10. `data/feedback_human.jsonl` is gitignored (existing `data/` rule
    catches it).
11. The two runbooks exist and pass a sanity read.

## Non-Goals

- **No multi-user auth, orgs, or billing.** Master plan rules these out
  for the project end-to-end. We use `session_id` (a client-generated
  uuid stored in `localStorage`) as identity.
- **No multi-turn brainstorming.** Phase 1–4 are single-turn; the API
  takes one prompt, returns one response.
- **No real streaming token-by-token from the model**. The wire protocol
  is SSE, but the underlying generator is non-streaming today; the
  server chunks the final string into windows. True token streaming is a
  `model/hf.py` upgrade (HF `streamer=...`) that we defer because it
  doesn't change the wire contract or the human-feedback shape.
- **No Vercel / Next.js / Modal-Web deployment** in this plan. Master
  plan suggests them; we ship the pieces they would mount on top of and
  defer the deploy step to a separate one-shot.
- **No Prometheus / Grafana / OpenTelemetry.** Master-plan stretch goal,
  not a contract. JSONL logs + a `/metrics` JSON endpoint are enough for
  small user counts.
- **No human-feedback influence on the in-line evaluator.** The
  evaluator's rubric is unchanged; agreement is *measured*, not
  *enforced*. Recalibration would bump `rubric_version` and is its own
  follow-up.
- **No automatic continuous-training cron.** Master plan suggests one;
  we keep the script operator-triggered. Modal `schedules.cron` is a
  one-line wrap around the existing function and is best added once we
  see a few manual runs land cleanly.
- **No A/B-by-domain / persona / time-of-day routing.** Hash by session
  is sufficient.

## Open Items (resolve during implementation)

1. **SSE chunk size.** The first cut chunks the final response into ≤50
   visible tokens per event. If the UI feels jumpy, drop to 20; if it
   feels slow, raise to 80. Live-tunable in `api/server.py`.
2. **Frontend session id.** First cut: client generates `crypto.randomUUID()`
   on first visit, persists to `localStorage`. If we later want stable
   identity across browsers, swap to a server-issued cookie — wire
   contract is unchanged because `session_id` stays opaque.
3. **`promote_if_better` thresholds.** `+0.10` and `p < 0.20` are first
   guesses. After the first three operator-driven retraining runs we
   should look at the gate's reject rate and tighten/loosen accordingly
   (open as a comment in `training/promotion.py`).
4. **Rate-limit defaults.** `30 req / 60s` is a guess; bump up if Keshav
   tests against the live UI. The bucket lives in-process and resets on
   restart, so a generous default is safer than a stingy one.
5. **Memory-context scrubber breadth.** First cut catches the obvious
   patterns. If a real injection slips through during testing, log a
   reproducer and tighten the regex in a follow-up commit; do not
   silently widen the filter.
6. **Frontend on Apple Silicon.** No build step means no toolchain
   incompatibility — the `frontend/` directory is plain HTML/JS that any
   browser eats. If/when we move to Vercel + Next.js, the API contract
   is what survives.
7. **Continuous-train invocation surface.** First cut runs `train_dpo.py`
   as a subprocess. If subprocess-launching gets flaky on Modal, switch
   to importing `_run_training` directly — contract is the same.

## Closeout — no plan-07

Plan-06 closes the master plan. The handoff is **operational**, not to
another sub-plan:

- **Operator** (Keshav) runs `brainstorm serve` against a Modal- or
  fixed-box-hosted model, points a small group of beta users at the URL,
  and lets the human-feedback log fill up. Phase 4 exit criterion 1
  closes when the first non-Keshav user finishes a session.
- **Operator** runs `scripts/continuous_train.py` once after the human-
  feedback log has ≥1 thumbs-up + ≥1 thumbs-down per ~5 prompts. The
  promoted-adapter pointer flips iff the gate clears. Exit criterion 2
  closes on the first such run.
- **Operator** runs `scripts/agreement.py` and archives the JSON.
  Criterion 3 closes here.
- The runbooks at `docs/runbooks/{garbage_output,training_loop_on_bad_data}.md`
  close criterion 4.

Future sub-plans (plan-07+) are out of scope for this document; if/when
the master plan is extended (multi-turn, on-policy RLHF, multi-model
deployment), they get their own files under `docs/plans/`.
