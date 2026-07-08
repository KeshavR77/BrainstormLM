"""Pydantic models for the public Brainstorm API v1."""

from __future__ import annotations

from pydantic import BaseModel


class IdeaV1(BaseModel):
    id: str
    title: str
    body: str


class DiversityScoreV1(BaseModel):
    score: float
    method: str
