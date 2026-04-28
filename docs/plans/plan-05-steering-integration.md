# Plan 05 — Steering Vector Integration

_Parent plan: `docs/plans/master-plan.md` (Phase 3, Sub-plan 3.1)._
_Builds on: `plan-04-rlhf-dpo.md` (LoRA + DPO training landed at commit `7c84ffb`)._
_Authored: 2026-04-27._

## Context

Phases 1 and 2 are closed. The pipeline has a Together-served base
generator, a synthetic feedback loop, ReasoningBank retrieval, mined
preference pairs, a DPO+LoRA Modal trainer, and a 4-arm ablation harness.
Every consumer of the model goes through the `Generator` protocol
(`src/brainstormlm/model/interface.py:6`), routed by
`make_generator(adapter_path)` (`src/brainstormlm/model/factory.py:12`).

Plan-05 is the steering-vector swap-in promised by the master plan's
Phase 3. We ship a third `Generator` implementation that derives a
**creative-vs-generic activation contrast vector** on the same Qwen2.5-7B
base used in plan-04, then injects it into the residual stream at
inference via `register_forward_hook`. The CLI gets `--steered` /
`--steering-path` flags; everything downstream (feedback loop, ablation
eval) routes through the new generator without touching training, eval
math, or memory code.

After this plan lands we satisfy the master plan's Phase 3 exit
criteria:

1. The steered model loads and generates end-to-end.
2. `run_feedback_loop.py --steered` runs clean and writes valid
   trajectories with provenance recorded.
3. DPO training (plan-04's `train_dpo.py`) runs on trajectories from the
   steered model without code changes outside `src/brainstormlm/model/`.
4. The steered eval row matches or exceeds the equivalent base row in
   the ablation table.

## Decisions Pinned

| Decision | Choice | Rationale |
|---|---|---|
| Steering library | **Hand-rolled `register_forward_hook` on transformer blocks** | We already pin `torch` + `transformers`. Adding `nnsight`/`repeng`/`transformer_lens` would be a new dep for a few dozen lines of hook code. Per master-plan principle 6, DRY across phases — reuse the existing HF stack. |
| Derivation method | **Contrastive Activation Addition (CAA)**: mean over N paired prompts of `h_creative[L] - h_generic[L]` at the last prompt token | Standard, cheap, reproducible. Doesn't need labeled responses or paired generations — just paired *system framings* over the same domain prompts. |
| Contrast pair source | **`HELD_OUT`-disjoint domain prompts** drawn from `prompter.generate_prompt`, paired with two fixed framings (`creative_system`, `generic_system`) | Reuses the existing prompter and stays inside our domain taxonomy. Master-plan rule: held-out prompts are NEVER touched outside eval. The derivation script asserts disjointness. |
| Default contrast set size | **64 paired prompts** | CAA-style vectors saturate fast; 32 is usually enough, 64 is conservative without wasting tokens. Configurable via `--n-pairs`. |
| Target layers | **Middle third of the stack** (default `layers=[12, 14, 16, 18, 20]` for the 28-layer Qwen2.5-7B) | Empirically the residual stream is most "semantic" mid-stack; early layers carry token-level features, late layers carry next-token logits. Configurable via `--layers`. |
| Default strength | **`+1.0`** (multiplied by per-layer L2 norm of the steering vector before injection) | Norm-rescaled strength keeps the additive intervention proportional to the layer's natural activation magnitude, so the same `--strength` flag transfers across model sizes without re-tuning. |
| Steering vector storage | **`safetensors`** at `data/steering/<run_id>/vector.safetensors` + `manifest.json` | Same on-disk family as plan-04's adapters. `safetensors` is already a transitive dep via `transformers`. Storage is keyed `layer_<idx>` → `(hidden_dim,)` `bf16` tensor. |
| Inference cache discipline | **Cache key includes steering provenance** (`steering_path`, rounded `strength`, sorted layer list) | Same sha256 disk-cache layer as `HFGenerator` — re-running the eval is free. |
| Generator stacking | **`SteeredGenerator` subclasses `HFGenerator`** and accepts an optional `adapter_path` | One generator handles "base + steering", "trained + steering", "steered alone" without proliferating classes. The `Generator` protocol stays unchanged. |
| Schema additivity | **`steering_path: str \| None` + `steering_strength: float \| None` added to `Trajectory`** (additive, defaults `None`) | Same discipline as `adapter_path` from plan-04. Old rows stay valid. `schema_version` stays at 1. |
| Ablation harness | **Extend in place** with optional `--steered` / `--steering-path` toggles that turn each of the 4 arms into its `steered` counterpart | Avoids forking the harness; a separate run produces the steered comparison. Phase 3 exit criterion 4 is "matches or exceeds equivalent base row," not "ships an 8-row table." |

## Deliverables

Files to **create**:

```
src/brainstormlm/model/steering.py          # SteeringVector + derivation math
src/brainstormlm/model/steered.py           # SteeredGenerator(HFGenerator) with hooks
scripts/derive_steering.py                  # Build contrast pairs → derive → save vector
tests/test_steering_derive.py               # Pure-numpy derivation math (no GPU, no network)
tests/test_steered_generator.py             # Hook registration + cache discipline (stubbed torch)
tests/test_steering_factory.py              # make_generator routing for the steered path
```

Files to **extend**:

```
src/brainstormlm/config.py                  # add default_steering_path, default_steering_strength, default_steering_layers, steering_dir
src/brainstormlm/model/factory.py           # accept steering_path / strength / layers; route to SteeredGenerator
src/brainstormlm/model/__init__.py          # re-export SteeredGenerator
src/brainstormlm/cli.py                     # add --steered / --steering-path / --steering-strength flags
src/brainstormlm/feedback/trajectories.py   # add steering_path: str|None, steering_strength: float|None (additive)
src/brainstormlm/eval/harness.py            # accept a single optional steered_generator + arm renaming hook
scripts/run_feedback_loop.py                # add --steered flag; thread steering provenance into Trajectory
scripts/sample_siblings.py                  # add --steered flag; route through make_generator (no schema change to PreferencePair)
scripts/eval_ablation.py                    # add --steered / --steering-path flags
.env.example                                # document BRAINSTORMLM_STEERING_PATH (no new keys)
```

No changes to `feedback/evaluator.py`, `feedback/prompter.py`, `memory/*`,
`training/*`, or `eval/stats.py`. They are consumed unchanged.

No new deps. `torch`, `transformers`, `safetensors` are already pinned
via plan-04.

## Steering Vector Schema

```python
# src/brainstormlm/model/steering.py
class SteeringVector:
    """Per-layer residual-stream steering vector, plus provenance."""

    base_model_id: str                 # "Qwen/Qwen2.5-7B-Instruct"
    layers: tuple[int, ...]            # e.g. (12, 14, 16, 18, 20)
    hidden_dim: int                    # 3584 for Qwen2.5-7B
    vectors: dict[int, "torch.Tensor"] # layer_idx → (hidden_dim,) tensor
    n_contrast_pairs: int              # how many pairs went into the mean
    derived_at: str                    # ISO 8601 UTC
    derivation_version: int            # bump when the derivation prompt changes
```

Persisted as a directory:

```
data/steering/<run_id>/
├── vector.safetensors                # keys: "layer_12", "layer_14", ...
├── manifest.json                     # {base_model_id, layers, hidden_dim, n_contrast_pairs, derived_at, derivation_version}
```

`SteeringVector.save(path)` writes both files; `SteeringVector.load(path)`
validates the manifest's `base_model_id` matches the consumer's
`hf_base_model` and raises a loud `ValueError` on mismatch.

## Derivation Contract

`src/brainstormlm/model/steering.py::derive_from_pairs`

```python
def derive_from_pairs(
    pairs: list[tuple[str, str]],     # (creative_text, generic_text) per pair
    *,
    base_model_id: str,
    layers: Sequence[int],
    device: str | None = None,
    dtype: str = "bf16",
) -> SteeringVector: ...
```

Algorithm (per pair, per layer):

1. Apply the chat template to each side: `tokenizer.apply_chat_template`
   with `add_generation_prompt=True` so the input ends at the assistant
   turn boundary — same render path `HFGenerator` uses for inference.
2. Forward both side strings through the model with
   `output_hidden_states=True`. No grad.
3. For each requested `layer_idx`, take the hidden state at the **last
   prompt token** (`hidden_states[layer_idx][0, -1, :]`).
4. Difference per pair: `delta_L = h_creative[L] - h_generic[L]`.
5. Average deltas across pairs: `v_L = mean_pairs(delta_L)`.

Hidden-state contraction is intentionally last-token-only: it's the
position the next-token prediction is conditioned on, and it matches the
literature's CAA setup. Earlier-position contractions over-reward
prompt-template tokens that are identical between the two sides.

Two **fixed framings** — both hardcoded in `model/steering.py` — are used
to build pairs:

```
CREATIVE_SYSTEM = (
    "You are a wildly inventive brainstorming partner. Embrace surprising "
    "ideas, unconventional analogies, and creative leaps. Quality over "
    "consensus."
)
GENERIC_SYSTEM = (
    "You are a careful, conventional assistant. Give the most standard, "
    "expected answer. Avoid speculation or unusual ideas."
)
```

Bumping either string bumps `derivation_version`. The hardcoding is
deliberate: contrast wording is part of the artifact.

## Pair Construction Contract

`scripts/derive_steering.py` orchestrates:

1. Shuffle `prompter.DOMAINS` by `--seed`.
2. For `i in range(n_pairs)`:
   - `domain = DOMAINS[i % len(DOMAINS)]`
   - `prompt_text, _ = prompter.generate_prompt(domain, salt=i)` (cached!).
   - `assert prompt_text not in HELD_OUT` (loud failure — same rule as
     plan-04's sibling sampling).
   - Render `(creative_text, generic_text)` by composing the two fixed
     framings with the same user prompt.
3. Pass the list to `derive_from_pairs`.
4. `vector.save(steering_dir / run_id)`.
5. Print: `n_pairs`, layers, output path, derivation cost (prompter
   only — the model forward passes are local and free).

Cost guardrails: `--max-spend-usd` covers prompter calls only. Local
forward passes have no $ cost; the budget is wall-clock time, capped by
the script timeout.

## SteeredGenerator Contract

`src/brainstormlm/model/steered.py::SteeredGenerator`

```python
class SteeredGenerator(HFGenerator):
    def __init__(
        self,
        *,
        steering_path: Path,
        steering_strength: float = 1.0,
        steering_layers: Sequence[int] | None = None,  # None → use all from manifest
        base_model_id: str | None = None,
        adapter_path: Path | None = None,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> None: ...
```

- Inherits the lazy-load + cache discipline from `HFGenerator`.
- Cache key is extended:
  `cache_key("hf-steered", base_model_id, adapter_path, steering_path,
   f"{strength:.4f}", ",".join(map(str, sorted(layers))), system, prompt,
   temperature, max_tokens)`. Cache hits short-circuit before any model
   load, so warm runs are free.
- On the first cache miss, after `_ensure_loaded()` (which already
  applies the optional PEFT adapter), the generator:
  1. Loads `SteeringVector` from `steering_path` (cached on `self`).
  2. For each `layer_idx`, picks `model.model.layers[layer_idx]`
     (Qwen/LLaMA-style decoder block) and registers a forward hook that
     adds `strength * v_L * (||h_pre|| / ||v_L||)` to the layer's
     residual output (norm-rescaled).
  3. Wraps `_run_generate` in a try/finally that removes every hook
     handle after generation, so a subsequent unsteered call on the
     same instance is unaffected.
- `_residual_layers()` resolves the block list robustly: it tries
  `model.model.layers` (transformers default for LLaMA/Qwen), falls back
  to `model.transformer.h` (GPT-style), and raises a loud `RuntimeError`
  on neither — meaning we hit a model architecture we haven't handled.

`SteeredGenerator` keeps the `generate` / `generate_with_meta` shape
unchanged so the `Generator` protocol holds.

## Factory + Config Wiring

`src/brainstormlm/model/factory.py::make_generator` is widened:

```python
def make_generator(
    adapter_path: Path | None = None,
    *,
    steering_path: Path | None = None,
    steering_strength: float | None = None,
    steering_layers: Sequence[int] | None = None,
) -> Generator:
    if steering_path is not None:
        return SteeredGenerator(
            steering_path=steering_path,
            steering_strength=steering_strength
                if steering_strength is not None
                else get_settings().default_steering_strength,
            steering_layers=steering_layers,
            adapter_path=adapter_path,
        )
    if adapter_path is not None:
        return HFGenerator(adapter_path=adapter_path)
    return TogetherGenerator()
```

The four resulting routings:

| `adapter_path` | `steering_path` | Result |
|---|---|---|
| None | None | `TogetherGenerator` (Phase 1 default) |
| set | None | `HFGenerator` w/ adapter (Phase 2) |
| None | set | `SteeredGenerator` over base HF (Phase 3) |
| set | set | `SteeredGenerator` over HF + adapter (steered + trained) |

Config additions (`src/brainstormlm/config.py`):

```python
default_steering_path: Path | None = Field(
    default=None, alias="BRAINSTORMLM_STEERING_PATH"
)
default_steering_strength: float = 1.0
default_steering_layers: tuple[int, ...] | None = None
steering_dir: Path = Field(
    default=Path("./data/steering"), alias="BRAINSTORMLM_STEERING_DIR"
)
```

## Trajectory Schema (additive)

```python
# src/brainstormlm/feedback/trajectories.py (add fields, default None)
steering_path: str | None = None
steering_strength: float | None = None
```

`schema_version` stays at 1 (additive change with `None` defaults
preserves backward compatibility — same discipline used for
`adapter_path` in plan-04).

## CLI & Script Wiring

**`brainstorm ask`** (extend `cli.py`):

```
brainstorm ask "<prompt>" \
    --steered \
    --steering-path data/steering/<run_id>/ \
    --steering-strength 1.0 \
    --use-memory --k-success 3 --k-failure 2 \
    --adapter data/adapters/<run_id>/      # all combinations valid
```

- `--steered` (no path) uses `settings.default_steering_path`. Errors
  loudly if neither is set.
- `--steering-path <path>` sets the path explicitly (also implies
  `--steered`).
- `--steering-strength <float>` overrides the default (1.0).
- `--no-steered` forces base/HF generator even if the env var is set.

**`scripts/run_feedback_loop.py`** (extend):

- Add `--steered` / `--steering-path` / `--steering-strength` flags
  matching the CLI.
- Route through `make_generator(adapter, steering_path=...)`.
- Persist `steering_path` and `steering_strength` on each `Trajectory`.

**`scripts/sample_siblings.py`** (extend, minimal):

- Add `--steered` / `--steering-path` to allow plan-04's pair mining to
  run over steered-base outputs (master-plan exit criterion 3 says DPO
  training must work over steered trajectories with no code changes
  outside `model/` — pair mining is `scripts/`, but threading the flag
  here is a one-liner via `make_generator` and avoids forcing later
  refactors).
- `PreferencePair` schema unchanged.

**`scripts/derive_steering.py`** (new):

```
python scripts/derive_steering.py \
    --n-pairs 64 \
    --layers 12,14,16,18,20 \
    --run-id <ulid> \
    --max-spend-usd 0.10 \
    --seed 42
```

Behavior:
1. Pre-flight prompter cost estimate (uses `pricing.cost_of` with the
   prompter token budget, same shape as `pricing.estimate_run_cost`).
   Abort if estimate > `--max-spend-usd`.
2. Build `n_pairs` `(creative_text, generic_text)` pairs from cached
   prompter calls. Assert disjointness from `HELD_OUT`.
3. `derive_from_pairs(...)` (loads model on demand; reuses the same HF
   path `HFGenerator` uses).
4. `vector.save(settings.steering_dir / run_id)`.
5. Print `manifest.json` contents + total prompter cost.

**`scripts/eval_ablation.py`** (extend):

- Add `--steered` / `--steering-path` / `--steering-strength`.
- When set, swap both `base_generator` and `trained_generator` for their
  steered counterparts via `make_generator`. The arm labels in the
  output change in place (`base` → `steered`, `trained+mem` →
  `steered+trained+mem`, etc.) so two consecutive runs produce
  comparable rows for the Phase 3 exit gate.
- The harness accepts an `arm_label_prefix: str = ""` knob to plumb the
  rename without forking the table-format code.

## Tests (TDD per CLAUDE.md)

All tests must pass without GPUs and without network calls.

1. **`tests/test_steering_derive.py`** — pure-Python math.
   - Hand-build a stub model whose `forward(..., output_hidden_states=True)`
     returns deterministic per-layer hidden states keyed by input string
     (so creative-vs-generic differences are predictable).
   - Run `derive_from_pairs` over 3 pairs across 2 layers; assert the
     resulting vectors equal the hand-computed mean of last-token
     differences.
   - `SteeringVector.save/load` round-trip in `tmp_path`; manifest
     mismatch raises `ValueError`; missing layer raises `ValueError`.
   - `derivation_version` is included in the manifest and survives the
     round-trip.

2. **`tests/test_steered_generator.py`** — hook plumbing.
   - Reuse the `_FakeTorch` / `_FakeTokenizer` / `_FakeModel` stubs from
     `tests/test_hf_generator.py` (lift to a shared `tests/conftest.py`
     fixture so both files reference one source).
   - Extend `_FakeModel` with a `model.layers` list of stub blocks, each
     of which records `register_forward_hook(fn)` calls and returns a
     fake handle whose `.remove()` flips a flag.
   - Test 1: Cache hit short-circuits — no hooks registered, no model
     load.
   - Test 2: Cache miss registers hooks on exactly the requested
     layers, runs `model.generate`, and removes every handle in the
     `finally` block (the flags all flip).
   - Test 3: A subsequent unsteered `HFGenerator` call on the same model
     instance does not see leaked hooks (re-use the same fake `_model`
     instance to verify).
   - Test 4: Cache key for `SteeredGenerator` differs from `HFGenerator`
     for the same prompt — verifying provenance is included.

3. **`tests/test_steering_factory.py`** — routing matrix.
   - `make_generator(None, steering_path=tmp)` → `SteeredGenerator`.
   - `make_generator(adapter, steering_path=tmp)` → `SteeredGenerator`
     with `adapter_path` forwarded to the parent class.
   - `make_generator(None, None)` → `TogetherGenerator` (regression).
   - `make_generator(adapter)` → `HFGenerator` (regression).

4. **Trajectory schema regression**: extend `tests/test_trajectories.py`
   (existing) with a roundtrip that includes `steering_path` /
   `steering_strength`. Asserts old rows lacking these fields still
   validate (defaults `None`).

Each test file ships in the same commit as the module it covers, per
`CLAUDE.md`.

## Implementation Sequence

The work is small enough that we don't need teammates. Three sequential
commits, each leaves the tree green:

**Step 1 — derivation core (~20 min):**
1. Add config fields (`default_steering_path`,
   `default_steering_strength`, `default_steering_layers`,
   `steering_dir`).
2. Implement `model/steering.py` — `SteeringVector`,
   `derive_from_pairs`, save/load.
3. Add `tests/test_steering_derive.py`.
4. Verify: `uv run pytest tests/test_steering_derive.py -q` green.

**Step 2 — generator + factory (~30 min):**
1. Implement `model/steered.py::SteeredGenerator` (subclass of
   `HFGenerator`, hook registration, cache key extension).
2. Extend `model/factory.py::make_generator` with the
   `steering_path` / `steering_strength` / `steering_layers` kwargs.
3. Re-export `SteeredGenerator` in `model/__init__.py`.
4. Lift the HF stubs from `tests/test_hf_generator.py` into
   `tests/conftest.py` so `tests/test_steered_generator.py` can reuse
   them; tighten the existing tests to import the shared fixtures.
5. Add `tests/test_steered_generator.py` and
   `tests/test_steering_factory.py`.
6. Verify: full `pytest` green.

**Step 3 — scripts + schema + CLI (~30 min):**
1. Add `steering_path` / `steering_strength` to `Trajectory` (additive
   defaults). Extend `tests/test_trajectories.py` with the roundtrip.
2. Wire `--steered` / `--steering-path` / `--steering-strength` into
   `cli.py::ask`, `scripts/run_feedback_loop.py`,
   `scripts/sample_siblings.py`, and `scripts/eval_ablation.py`.
3. Add `arm_label_prefix` plumbing to `eval/harness.py` (default `""`,
   so existing tests stay green).
4. Implement `scripts/derive_steering.py`.
5. `.env.example` documents `BRAINSTORMLM_STEERING_PATH` /
   `BRAINSTORMLM_STEERING_DIR`.
6. Verify gates below; commit per `CLAUDE.md` (imperative mood; tests
   ship with code; ~3 commits: derivation, generator+factory,
   integration).

## Verification

All gates must pass before closing plan-05:

1. `uv run pytest tests/` — all old + new tests green (≥10 new tests).
2. **Schema additivity smoke**: `from brainstormlm.feedback.trajectories
   import Trajectory; Trajectory.model_validate(<old-row>)` succeeds with
   `steering_path is None`. Confirms plan-02 / plan-03 / plan-04
   trajectories on disk still load.
3. **CLI smoke (no network, cache-hit only)**:
   `uv run brainstorm ask "Quick test" --steered --steering-path
   <prebuilt-fixture>` — exercised in tests via stubs because deriving
   on the dev box would require a real GPU. Documented as a deferred
   live gate (gate 7).
4. **Factory routing**: tests in `test_steering_factory.py` cover all
   four cells of the (adapter_path × steering_path) matrix.
5. **Derivation script (cached)**:
   `uv run python scripts/derive_steering.py --n-pairs 4 --layers
   12,16,20 --max-spend-usd 0.05 --run-id smoke` — on a dev box
   without GPU this is **deferred to a Modal counterpart** (open item
   below). Plan-05 ships the script + tests; the live derivation runs
   on the same A100 used by plan-04.
6. **Steered ablation regression** (target environment: Modal A100, same
   image as plan-04's `train_dpo.py`):
   `uv run python scripts/eval_ablation.py --steered --steering-path
   data/steering/<run_id>/ --adapter data/adapters/<run_id>/` produces
   the 4-row table renamed (`steered`, `steered+mem`, `steered+trained`,
   `steered+trained+mem`) with the master-plan Phase 3 outcomes:
   - `steered` ≥ `base` (plan-04 row), paired-bootstrap not
     significantly worse.
   - Ideally `steered+trained+mem` > `trained+mem` (master plan
     "ideally improve"). Required: not significantly worse.
7. **Live derivation deferred**: a follow-up Modal run produces the
   first real `data/steering/<run_id>/` artifact; this plan ships the
   plumbing and `derive_steering.py`. The derivation itself is a
   one-shot job.
8. Re-running `eval_ablation.py --steered` is fully cache-served — total
   cost reads $0.00, wall-clock under a few seconds.
9. `data/eval/ablation_<steered_run_id>.jsonl` exists and is disjoint
   from `data/trajectories.jsonl` and `data/preferences.jsonl`
   (held-out prompts must never enter training).

## Non-Goals

- **No vLLM-backed steering.** Master-plan Phase 4 footnote suggests a
  vLLM swap; staying on `transformers` keeps the diff small.
- **No multi-vector steering** (e.g. arithmetic combinations of multiple
  contrast directions). One vector per artifact.
- **No prompt-time steering** (system-prompt nudges that mimic the
  contrast). The point of plan-05 is the activation-space
  intervention, not the prompt-only proxy.
- **No automatic strength tuning.** `--steering-strength` is a manual
  knob; if Phase 3 exit criterion 4 fails by a small margin we sweep
  `{0.5, 1.0, 1.5}` by hand. Auto-tuning is a Phase 4 concern.
- **No retraining of the LoRA on steered trajectories** in plan-05. We
  verify that plan-04's adapter still works under steering; a fresh
  steered-base DPO run is its own follow-up.
- **No swap of the Together-served base.** `TogetherGenerator` keeps
  serving the no-adapter / no-steering path so cache-warmed dev work
  stays cheap.
- **No human feedback** — that is plan-06.

## Open Items (resolve during implementation)

1. **Layer choice empirical check.** Default `[12, 14, 16, 18, 20]` for
   28-layer Qwen2.5-7B. If gate 6 underperforms, sweep `{[10..14],
   [16..20], [22..26]}` and pick by held-out total.
2. **Strength rescaling formula.** `strength * v_L * (||h_pre|| /
   ||v_L||)` is the per-token, norm-matched form of additive
   intervention. If outputs degrade (incoherent text), drop to plain
   `strength * v_L` (no rescale). Both behaviors live behind a
   `--rescale/--no-rescale` flag, default rescale on.
3. **Hook attachment site.** Default: hook the residual *output* of each
   target decoder block (`block(...)[0]`). Alternative: hook the input
   to the next block. Equivalent in math, different in clarity. Pick
   one in the implementing commit.
4. **Apple Silicon / CPU smoke.** Forward passes for steering
   derivation on the dev box are slow but feasible at `n_pairs=4`. Real
   derivation runs on Modal. If MPS misbehaves on Qwen2.5-7B, fall back
   to CPU+`fp32` for the smoke test only.
5. **Adapter + steering interaction.** With a PEFT adapter loaded, the
   block list still resolves through `model.model.layers` (PEFT wraps
   modules, doesn't reparent the decoder). Verified in
   `test_steered_generator.py` via the stubbed model topology;
   confirmed live in gate 6.
6. **Symlink convenience.** After gate 7, consider
   `data/steering/latest -> <run_id>/` so the CLI can default without
   a path. Defer until a real artifact exists.

## Handoff to plan-06

Plan-06 (production system) consumes:
- The `Generator` protocol unchanged — the FastAPI server picks
  `make_generator(adapter_path, steering_path=...)` from request /
  feature-flag config.
- A persisted steering vector is just another artifact under `data/`,
  promoted via the same flag-gated mechanism plan-06 uses for adapters.
- The trajectory schema's `steering_path` / `steering_strength` fields
  carry through into human-feedback rows (plan-06 sets `source: human`
  and inherits the steering provenance fields).
- The ablation harness's `arm_label_prefix` knob lets plan-06's
  continuous-training job emit comparable tables when the steered
  variant is promoted.
