# Steering harness — ship a {vector, layer, alpha} package for testing

Serves Qwen2.5-14B with an activation-steering vector injected at one layer, behind an HTTP API.
Point it at a **package** (a small JSON spec) and it applies that steering to every generation —
with a baseline (un-steered) arm for A/B testing. Swapping the shipped config is a config change,
not a code change.

## The package

`creativity_steering/configs/ship_L17_a0.2.json` (the current ship pick):

```json
{
  "name": "v3-L17-a0.2",
  "vectors_path": "artifacts/qwen14b_run_v3/vectors/mean_response",
  "layer": 17, "alpha": 0.2, "rescale": true,
  "model_id": "Qwen/Qwen2.5-14B-Instruct", "device": "cuda", "dtype": "bf16",
  "system": null, "max_new_tokens": 256, "temperature": 0.9, "seed": 0
}
```

- `vectors_path` — directory with `vectors.npz` + `manifest.json` (the derived DiffMeans set).
- `layer` / `alpha` — inject the **unit** layer-direction × `alpha` at `layer`. `rescale=true`
  matches the injection to each token's activation norm, so `alpha` is dimensionless.
- `system: null` — generation uses no creative framing, so steered-vs-baseline isolates the vector.

To ship a different config, write another package JSON (different layer/alpha/vectors) and set
`STEERING_PACKAGE` to it.

## Run

```bash
PYTHONPATH=creativity_steering \
STEERING_PACKAGE=creativity_steering/configs/ship_L17_a0.2.json PORT=8000 \
python creativity_steering/scripts/serve_steering.py        # loads the 14B once (needs a GPU)
```

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | liveness + the loaded package's provenance |
| GET | `/package` | the active {name, layer, alpha, rescale, model, …} |
| POST | `/generate` | `{prompt, steered?=true, system?, max_new_tokens?, temperature?, seed?}` → `{text, steered, package, layer, alpha}` |
| POST | `/compare` | same body → `{prompt, baseline, steered, …}` — the A/B pair at identical seed |

```bash
curl localhost:8000/health
curl -X POST localhost:8000/compare -H 'content-type: application/json' \
     -d '{"prompt":"Give me 5 unconventional uses for a brick.","seed":0}'
```

`/compare` is the formal-testing workhorse: identical prompt + seed, the only difference is the
injection, so the difference in output is attributable to the steering vector.

## Programmatic

```python
from creativity_steering.serving import SteeringPackage, SteeringService, create_app
svc = SteeringService.from_package(SteeringPackage.load("…/ship_L17_a0.2.json"))
print(svc.generate("…", steered=True), svc.generate("…", steered=False))
app = create_app(svc)   # the FastAPI app
```

Tests (offline, MockBackend): `tests/test_serving.py`, `tests/test_serving_api.py`.
