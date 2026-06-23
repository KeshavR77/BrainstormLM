# Steering harness — ship a {vector, layer, alpha} package for testing

Serves Qwen2.5-14B with an activation-steering vector injected at one layer, behind an HTTP API.
Point it at a **package** (a small JSON spec) and it applies that steering to every generation —
with a baseline (un-steered) arm for A/B testing. Swapping the shipped config is a config change,
not a code change.

## The package

A package carries one or more **variants** — named steering configs the API caller chooses
between. The shipped `creativity_steering/configs/ship_v3_bracket.json` defaults to `L17` (the
single-cell ship pick) and also offers the `bracket` (L14+L16+L17 together) and each neighbour
individually:

```json
{
  "name": "v3-mid-network",
  "vectors_path": "artifacts/qwen14b_run_v3/vectors/mean_response",
  "default_variant": "L17",
  "variants": {
    "L17":     [{"layer": 17, "alpha": 0.2}],
    "L14":     [{"layer": 14, "alpha": 0.2}],
    "L16":     [{"layer": 16, "alpha": 0.2}],
    "bracket": [{"layer": 14, "alpha": 0.2}, {"layer": 16, "alpha": 0.2}, {"layer": 17, "alpha": 0.2}]
  },
  "rescale": true,
  "model_id": "Qwen/Qwen2.5-14B-Instruct", "device": "cuda", "dtype": "bf16",
  "system": null, "max_new_tokens": 256, "temperature": 0.9, "seed": 0
}
```

- `variants` — each entry is a list of `{layer, alpha}` injections. A request picks one by name
  (`"variant": "bracket"`); omitting it uses `default_variant`.
- Each injection uses the **unit** layer-direction × `alpha`; `rescale=true` matches it to each
  token's activation norm so `alpha` is dimensionless.
- `system: null` — no creative framing, so steered-vs-baseline isolates the vector.
- The single-cell form `ship_L17_a0.2.json` (top-level `layer`/`alpha`, no `variants`) still works.

Pick: `L17` is the primary; **`bracket` injects all three mid-network layers at once** (a stronger,
not-yet-measured combination — the individual `L14`/`L16`/`L17` variants are the proper band check).
To ship a different config, write another package JSON and set `STEERING_PACKAGE` to it.

## Run

```bash
PYTHONPATH=creativity_steering \
STEERING_PACKAGE=creativity_steering/configs/ship_v3_bracket.json PORT=8000 \
python creativity_steering/scripts/serve_steering.py        # loads the 14B once (needs a GPU)
```

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | liveness + the loaded package's provenance |
| GET | `/package` | the active package (default variant + all variants) |
| GET | `/variants` | `{default, variants}` — the steering configs the caller can choose |
| POST | `/generate` | `{prompt, variant?, steered?=true, system?, max_new_tokens?, temperature?, seed?}` → `{text, steered, package, variant, layers, layer, alpha}` |
| POST | `/compare` | same body → `{prompt, baseline, steered, variant, layers, …}` — the A/B pair at identical seed |

`variant` selects which steering to apply (omit → the package default, `L17`). An unknown
variant returns `400`.

```bash
curl localhost:8000/variants
curl -X POST localhost:8000/compare -H 'content-type: application/json' \
     -d '{"prompt":"Give me 5 unconventional uses for a brick.","seed":0}'              # default L17
curl -X POST localhost:8000/compare -H 'content-type: application/json' \
     -d '{"prompt":"Give me 5 unconventional uses for a brick.","variant":"bracket","seed":0}'
```

`/compare` is the formal-testing workhorse: identical prompt + seed, the only difference is the
injection, so the difference in output is attributable to the steering vector.

## Programmatic

```python
from creativity_steering.serving import SteeringPackage, SteeringService, create_app
svc = SteeringService.from_package(SteeringPackage.load("…/ship_v3_bracket.json"))
print(svc.generate("…", variant="L17"))       # default ship pick
print(svc.generate("…", variant="bracket"))   # all three mid-network layers
print(svc.generate("…", steered=False))       # baseline
app = create_app(svc)   # the FastAPI app
```

Tests (offline, MockBackend): `tests/test_serving.py`, `tests/test_serving_api.py`.
