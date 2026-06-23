"""FastAPI surface for the steering harness.

Endpoints:
  GET  /health   — liveness + the shipped package's provenance
  GET  /package  — the active {vector, layer, alpha} package
  POST /generate — steered (or baseline, ``steered:false``) completion for one prompt
  POST /compare  — baseline + steered for the same prompt/seed (the A/B pair for testing)
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from creativity_steering.serving.service import SteeringService


class GenRequest(BaseModel):
    prompt: str
    system: str | None = None
    steered: bool = True
    max_new_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None


class GenResponse(BaseModel):
    text: str
    steered: bool
    package: str
    layer: int
    alpha: float


class CompareResponse(BaseModel):
    prompt: str
    baseline: str
    steered: str
    package: str
    layer: int
    alpha: float


def create_app(service: SteeringService) -> FastAPI:
    app = FastAPI(title="Creativity-Steering Harness", version="1.0")
    pkg = service.package

    @app.get("/health")
    def health():
        return {"status": "ok", **service.info()}

    @app.get("/package")
    def package():
        return service.info()

    @app.post("/generate", response_model=GenResponse)
    def generate(req: GenRequest):
        text = service.generate(
            req.prompt, system=req.system, steered=req.steered,
            max_new_tokens=req.max_new_tokens, temperature=req.temperature, seed=req.seed,
        )
        return GenResponse(text=text, steered=req.steered, package=pkg.name,
                           layer=pkg.layer, alpha=pkg.alpha)

    @app.post("/compare", response_model=CompareResponse)
    def compare(req: GenRequest):
        kw = dict(system=req.system, max_new_tokens=req.max_new_tokens,
                  temperature=req.temperature, seed=req.seed)
        baseline = service.generate(req.prompt, steered=False, **kw)
        steered = service.generate(req.prompt, steered=True, **kw)
        return CompareResponse(prompt=req.prompt, baseline=baseline, steered=steered,
                               package=pkg.name, layer=pkg.layer, alpha=pkg.alpha)

    return app
