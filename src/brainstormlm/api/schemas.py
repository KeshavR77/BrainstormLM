"""Request/response Pydantic models for the FastAPI server (plan-06)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str = Field(max_length=64)
    thumbs: Literal["up", "down"]
    rating: int | None = Field(default=None, ge=1, le=5)
    critique: str | None = Field(default=None, max_length=2000)


class HealthzResponse(BaseModel):
    ok: bool


class FeedbackResponse(BaseModel):
    ok: bool


class MetricsResponse(BaseModel):
    total_requests: int
    requests_last_60s: int
    mean_latency_ms: float
    cache_hit_rate: float | None = None  # None until response caching is instrumented (not faked)


class BrainstormRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(max_length=4000)
    context: str | None = Field(default=None, max_length=4000)
    n_ideas: int = Field(default=4, ge=1, le=8)
    # Reserved: memory is always project-scoped via project_id (this flag is accepted for the
    # existing client contract but does not currently toggle behaviour).
    use_memory: bool = False
    project_id: str | None = Field(default=None, max_length=128)
    reset_project: bool = False


class MemoryCountResponse(BaseModel):
    project_id: str
    seeded: int
    generated: int
    total: int


class IdeaBranch(BaseModel):
    summary: str
    content: str
    trajectory_id: str


class BrainstormResponse(BaseModel):
    ideas: list[IdeaBranch]
