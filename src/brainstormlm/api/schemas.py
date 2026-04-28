"""Request/response Pydantic models for the FastAPI server (plan-06)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    session_id: str
    use_memory: bool = False
    k_success: int = Field(default=3, ge=0, le=20)
    k_failure: int = Field(default=2, ge=0, le=20)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=600, ge=1, le=4096)


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
    mean_latency_ms_by_arm: dict[str, float]
    cache_hit_rate: float
