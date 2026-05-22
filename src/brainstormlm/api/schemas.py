"""Request/response Pydantic models for the FastAPI server (plan-06)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    thumbs: Literal["up", "down"]
    rating: int | None = Field(default=None, ge=1, le=5)
    critique: str | None = None


class HealthzResponse(BaseModel):
    ok: bool
    promoted_adapter: str | None = None
    promoted_steering: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool


class MetricsResponse(BaseModel):
    total_requests: int
    requests_last_60s: int
    mean_latency_ms: float
    cache_hit_rate: float


class BrainstormRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    context: str | None = None
    n_ideas: int = Field(default=4, ge=1, le=8)
    use_memory: bool = False
    project_id: str | None = None
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
