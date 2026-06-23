#!/usr/bin/env python3
"""Serve a steering package over HTTP for formal testing.

Loads Qwen2.5-14B once, applies the package's {vector, layer, alpha} during generation, and
exposes /generate and /compare (baseline vs steered for the same prompt). Swap the shipped
config by pointing STEERING_PACKAGE at a different package JSON — no code change.

  # default ships the v3 mid-network package: default variant L17 @ alpha=0.2, plus the
  # bracket (L14+L16+L17) the caller can choose via the request's "variant" field:
  STEERING_PACKAGE=creativity_steering/configs/ship_v3_bracket.json \
  PORT=8000 python creativity_steering/scripts/serve_steering.py

  # then:
  curl localhost:8000/variants                       # list L17 / L14 / L16 / bracket
  curl -X POST localhost:8000/compare -H 'content-type: application/json' \
       -d '{"prompt":"Give me 5 uses for a brick.","seed":0}'                 # default L17
  curl -X POST localhost:8000/compare -H 'content-type: application/json' \
       -d '{"prompt":"Give me 5 uses for a brick.","variant":"bracket","seed":0}'

Optional steer -> refine: set REFINER=openai (uses OPENAI_API_KEY; REFINER_MODEL overrides the
default gpt-4o-mini) to enable a concept-frozen OpenAI refiner. Then pass "refine": true on
/generate or /compare to get a `refined` field alongside the raw steered text.
"""
import os

import uvicorn

from creativity_steering.serving.app import create_app
from creativity_steering.serving.package import SteeringPackage
from creativity_steering.serving.refiner import OpenAIRefiner
from creativity_steering.serving.service import SteeringService


def _make_refiner():
    """Optional concept-frozen OpenAI refiner, enabled by REFINER=openai (reads OPENAI_API_KEY)."""
    if os.environ.get("REFINER", "").strip().lower() not in {"openai", "1", "true", "on"}:
        return None
    return OpenAIRefiner(model=os.environ.get("REFINER_MODEL", "gpt-4o-mini"))


def main() -> None:
    pkg_path = os.environ.get(
        "STEERING_PACKAGE", "creativity_steering/configs/ship_v3_bracket.json"
    )
    pkg = SteeringPackage.load(pkg_path)
    refiner = _make_refiner()
    print(f"[serve] package '{pkg.name}': layer={pkg.layer} alpha={pkg.alpha} "
          f"rescale={pkg.rescale} vectors={pkg.vectors_path} model={pkg.model_id}", flush=True)
    print(f"[serve] refiner: {refiner.model if refiner else 'off'} "
          f"(set REFINER=openai to enable steer -> refine)", flush=True)
    print("[serve] loading vectors + model (first load streams the 14B shards)…", flush=True)
    service = SteeringService.from_package(pkg, refiner=refiner)
    app = create_app(service)
    print("[serve] ready. endpoints: GET /health /package · POST /generate /compare", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
