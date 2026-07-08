# Brainstorm API — Buildout Plan (v1)

_Last updated: 2026-05-23_

## Goal

Ship a hosted HTTP API a partner company integrates. Their end-user taps a
**"brainstorm"** button → we take a `prompt` (+ optional `context`) → return
**N diverse text ideas** (`title` + `body`). The company renders any images on
their side. Ideas carry stable IDs + a `session_id` so "edit" and "continue
brainstorming" can come later without a contract change.

## Locked decisions

| Decision | Choice |
|---|---|
| Generation backend | **Persistent Conceptial GPU** (always-on box, model warm) |
| Idea shape | **`title` + `body`** |
| Pipeline depth | **Safety → internal brief-refine (Claude) → Conceptial → select** |
| Auth | API key per company (`Bearer`), SHA-256 hashed at rest |
| Transport | Sync JSON (3 ideas don't need streaming) |
| `n` | default 3, capped 1–10 |
| Diversity score | returned in payload (proof of value) |

## v1 API contract

```
POST /v1/brainstorm
Authorization: Bearer <company_api_key>
Content-Type: application/json

Request:
{
  "prompt":     "string (required)",
  "context":    "string (optional)",
  "n":          3,
  "session_id": "uuid (optional; omit to start fresh)"
}

Response 200:
{
  "request_id": "uuid",
  "session_id": "uuid",
  "ideas": [ { "id": "idea_a1b2", "title": "...", "body": "..." }, ... ],
  "diversity": { "score": 0.82, "method": "vendi" },
  "model": "conceptial-14b"
}

Errors: 400 prompt/safety reject · 401 bad key · 422 validation
        · 429 rate limit · 503 engine unavailable (+ Retry-After)
```

## Deploy topology

Two **separate** services (do not merge):

- **Service A — marketing site + demo** — `api/server.py`, **torch-free**,
  Render. Unchanged. (The repo cut on 2026-05-22 kept this import-light on
  purpose; keep it that way.)
- **Service B — company brainstorm API** — new `api/public.py`, runs on the
  **persistent GPU box**, the *only* process that imports `model/engine_adapter`
  and loads the 14B in-process. Model warm on boot → no cold start.

## ⭐ Parallel-work interface (the de-risking centerpiece)

The model team refines the *sole model*; the API team builds everything else.
Neither blocks the other because they meet at **one frozen interface** plus a
**contract test**.

```python
# src/brainstormlm/pipeline/contracts.py  (FROZEN for v1)
@dataclass
class IdeaCandidate:
    text: str            # MUST contain "Title: <t>\n<body>" (agreed format)
    meta: dict           # seed, creative_layer, alpha, etc.

class BrainstormGenerator(Protocol):
    def generate_candidates(
        self, brief: CreativeBrief, n: int, *, exclude: list[str] = ()
    ) -> list[IdeaCandidate]: ...
```

- **Model team owns** the real implementation against the engine: steering
  quality, the `Title:/body` output format, generation-time diversity
  (anti-mode, iterative-ortho), latency on the GPU. They iterate on weights /
  vectors / prompts **behind this method**.
- **API team owns** HTTP, auth, brief-refine, selection, assembly, deploy,
  metering — built against `MockBrainstormGenerator` (fixed candidates).
- **Integration** = a shared **contract test** (`tests/test_generator_contract.py`):
  given a brief, the impl returns `n` candidates in the agreed format that pass
  a diversity threshold. When the model team's impl is green, swap mock → real
  (one line in the factory). No other API code changes.

## Workstreams (run in parallel)

**Model team — "refine the sole model"**
- [ ] Tune steering operating point on the live brief suite (L17/α schedule).
- [ ] Lock the candidate output format (`Title:` + body) and verify the engine
      emits it reliably.
- [ ] Implement `BrainstormGenerator.generate_candidates` over `EngineGenerator`.
- [ ] Pass `tests/test_generator_contract.py` (format + diversity threshold).
- [ ] Warm-load + `/healthz` on the GPU box; measure p50/p95 latency for N=6–8.

**API team — "build the API"** (Phase 1 below, no GPU needed)

## Phase 1 — buildable NOW (mock generator, no GPU)

TDD per repo convention (Red → Green → Refactor). Engine sits behind the
`Generator`/`BrainstormGenerator` protocol so CI mocks it.

- [ ] `pipeline/contracts.py` — `CreativeBrief`, `IdeaCandidate`,
      `BrainstormGenerator` protocol, `Idea`.
- [ ] `api/public_schemas.py` — public v1 models. `IdeaV1` / `DiversityScoreV1` exist and
      are used by `pipeline/select_api.py`; the request/response wrappers were removed as unused.
- [ ] `api/auth.py` — `require_api_key` dependency: parse `Bearer`, SHA-256
      compare against key store, attach `company_id`; 401 on miss. **Test first.**
- [ ] `refine.py` — `refine_brief(prompt, context) -> CreativeBrief` via Claude
      (Anthropic SDK, prompt-cached). Behind an interface so it's mockable.
- [ ] `pipeline/select.py` — dedup + quality-gate + diversity-coverage (MMR over
      embeddings); compute `diversity.score` (Vendi) on the chosen K. **Test first.**
- [ ] `pipeline/brainstorm.py` — orchestrator: safety → refine → generate N →
      select n → parse `Title:/body` → assemble `Idea[]` with stable IDs +
      `session_id` write to `idea_store`.
- [ ] `api/public.py` — FastAPI app: `POST /v1/brainstorm`, `require_api_key`,
      `RateLimitMiddleware`, error model, OpenAPI; uvicorn entrypoint in
      `pyproject` scripts.
- [ ] `MockBrainstormGenerator` + `tests/test_brainstorm_api.py` end-to-end
      (mock engine) → assert 3 distinct ideas, stable IDs, diversity present,
      auth enforced, safety rejection path, `session_id` round-trips.

**Phase 1 acceptance:** `POST /v1/brainstorm` returns 3 well-formed diverse
ideas against the mock, full suite green, no GPU, no torch in CI.

## Phase 2 — GPU integration (needs the box)

- [ ] Implement + register the real `BrainstormGenerator` (swap mock → engine).
- [ ] Wire `refine.py` to live Claude; tune brief prompt (open_axes preserved).
- [ ] Tune `select` thresholds on real candidates; calibrate N (6–8) → 3.
- [ ] Deploy Service B to the persistent GPU host; warm load; `/healthz`.
- [ ] Per-request metering (count per `company_id`), per-stage latency + diversity
      logging; dashboard.
- [ ] Load test; set per-key rate limit + concurrency cap; 503 + `Retry-After`
      when GPU saturated.

## Readiness checklist ("get ready" = these, not just code)

- [ ] GPU host provisioned, always-on, model + L17/L24 vectors on disk, warm on boot.
- [ ] Secrets: `ANTHROPIC_API_KEY`, company API-key store (**hashes only**).
- [ ] Metering for billing (requests + ideas per key).
- [ ] Versioned `/v1`, OpenAPI doc + copy-paste `curl` for the partner.
- [ ] Timeouts/SLA; graceful 503 path; fallback policy if engine down.
- [ ] Auth + rate limit verified; CORS appropriate for server-to-server (no browser).

## Milestones

1. **M1 (Phase 1 done):** contract + mock API green in CI. Partner can integrate
   against the shape immediately.
2. **M2:** model team passes the contract test; swap to real engine in staging.
3. **M3:** deployed to GPU box, metered, load-tested → partner pilot.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Brief-refine narrows the creative space | keep brief enabling; `open_axes` free; A/B refine-on vs off on diversity |
| 3 returned ideas not distinct enough | over-generate N=6–8, select for coverage (MMR), report Vendi |
| GPU single point of failure | health checks + fallback flag; plan horizontal split post-pilot |
| Output format drift breaks parsing | contract test pins `Title:/body`; parser tolerant + validated |
| torch creeps into Service A | Service B is a separate process/entrypoint; keep `server.py` import-light |
