"""Pydantic models for the public Brainstorm API v1."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BrainstormV1Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(max_length=4000)
    context: str | None = Field(default=None, max_length=4000)
    n: int = Field(default=3, ge=1, le=10)
    session_id: str | None = Field(default=None, max_length=128)


class IdeaV1(BaseModel):
    id: str
    title: str
    body: str


class DiversityScoreV1(BaseModel):
    score: float
    method: str


class BrainstormV1Response(BaseModel):
    request_id: str
    session_id: str
    ideas: list[IdeaV1]
    diversity: DiversityScoreV1
    model: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
