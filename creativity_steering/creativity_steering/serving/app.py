"""FastAPI surface for the steering harness.

Endpoints:
  GET  /health   — liveness + the shipped package's provenance
  GET  /package  — the active package (default variant + all variants)
  GET  /variants — the named steering variants the caller can choose (e.g. L17 vs bracket)
  POST /generate — steered (or baseline) completion; pick a ``variant`` or use the default
  POST /compare  — baseline + steered for the same prompt/seed (the A/B pair for testing)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from creativity_steering.serving.service import SteeringService


class GenRequest(BaseModel):
    prompt: str
    variant: str | None = None     # which steering to apply (default: package default)
    system: str | None = None
    steered: bool = True
    max_new_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None


class GenResponse(BaseModel):
    text: str
    steered: bool
    package: str
    variant: str
    layers: list[dict]             # the (layer, alpha) injections applied
    layer: int | None = None       # convenience: first layer (single-layer variants)
    alpha: float | None = None


class CompareResponse(BaseModel):
    prompt: str
    baseline: str
    steered: str
    package: str
    variant: str
    layers: list[dict]
    layer: int | None = None
    alpha: float | None = None


def create_app(service: SteeringService) -> FastAPI:
    app = FastAPI(title="Creativity-Steering Harness", version="1.1")
    pkg = service.package

    def _resolve(variant: str | None):
        if variant is not None and variant not in pkg.variants:
            raise HTTPException(
                status_code=400,
                detail=f"unknown variant {variant!r}; choose from {pkg.variant_names()}",
            )
        name = variant or pkg.default_variant
        spec = pkg.variants[name]
        return name, spec, spec[0]["layer"], spec[0]["alpha"]

    @app.get("/health")
    def health():
        return {"status": "ok", **service.info()}

    @app.get("/package")
    def package():
        return service.info()

    @app.get("/variants")
    def variants():
        return {"default": pkg.default_variant, "variants": pkg.variants}

    @app.post("/generate", response_model=GenResponse)
    def generate(req: GenRequest):
        name, spec, layer, alpha = _resolve(req.variant)
        text = service.generate(
            req.prompt, variant=req.variant, system=req.system, steered=req.steered,
            max_new_tokens=req.max_new_tokens, temperature=req.temperature, seed=req.seed,
        )
        return GenResponse(text=text, steered=req.steered, package=pkg.name, variant=name,
                           layers=spec, layer=layer, alpha=alpha)

    @app.post("/compare", response_model=CompareResponse)
    def compare(req: GenRequest):
        name, spec, layer, alpha = _resolve(req.variant)
        kw = dict(variant=req.variant, system=req.system, max_new_tokens=req.max_new_tokens,
                  temperature=req.temperature, seed=req.seed)
        baseline = service.generate(req.prompt, steered=False, **kw)
        steered = service.generate(req.prompt, steered=True, **kw)
        return CompareResponse(prompt=req.prompt, baseline=baseline, steered=steered,
                               package=pkg.name, variant=name, layers=spec, layer=layer, alpha=alpha)

    return app
