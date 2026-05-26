"""Safety primitives for the production API (plan-06).

Three concerns, intentionally minimal:
  1. ``scrub_lesson`` / ``scrub_lessons`` — strip role-prefix lines and
     chat-template control tokens from retrieved memory before they hit
     the system prompt.
  2. ``validate_user_prompt`` — reject obvious garbage at the boundary.
  3. ``TokenBucket`` / ``RateLimitMiddleware`` — in-process per-IP rate
     limit. Adequate for small user counts; LBs handle the real traffic.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from brainstormlm.memory.lessons import Lesson

_ROLE_PREFIX_RE = re.compile(
    r"^\s*(system|assistant|user|developer|tool)\s*:",
    re.IGNORECASE,
)
_CONTROL_TOKEN_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>",
    re.IGNORECASE,
)


def scrub_lesson(text: str, *, max_len: int = 280) -> str:
    """Remove role-prefixed lines + chat-template tokens; truncate.

    Lessons are LLM-generated and not user input, but they ride into the
    system prompt; the cheapest defense against a critique-laundered
    injection is to refuse to render lines that look like role headers.
    """
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        if _ROLE_PREFIX_RE.match(raw_line):
            continue
        cleaned = _CONTROL_TOKEN_RE.sub("[CONTROL]", raw_line)
        kept_lines.append(cleaned)
    out = "\n".join(kept_lines).strip()
    if len(out) > max_len:
        out = out[:max_len].rstrip()
    return out


def scrub_lessons(lessons: list[Lesson], *, max_len: int = 280) -> list[Lesson]:
    """Return a list of lessons with each ``lesson`` field scrubbed.

    Metadata (``id``, ``kind``, ``domain``, etc.) is preserved.
    """
    return [
        lesson.model_copy(update={"lesson": scrub_lesson(lesson.lesson, max_len=max_len)})
        for lesson in lessons
    ]


def validate_user_prompt(prompt: str, *, max_len: int = 2000) -> None:
    """Raise ValueError if the prompt is empty, too long, or has control bytes."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is empty")
    if len(prompt) > max_len:
        raise ValueError(f"prompt exceeds {max_len} chars")
    for ch in prompt:
        if ch == "\n" or ch == "\t":
            continue
        if ord(ch) < 0x20:
            raise ValueError("prompt contains control characters")


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucket:
    """In-process token bucket keyed by an opaque string (e.g. client IP).

    Includes periodic eviction: entries idle for >10 minutes are pruned
    every 256 requests to prevent unbounded memory growth from IP rotation.
    """

    _EVICT_INTERVAL = 256  # check every N calls
    _STALE_SECONDS = 600.0  # evict entries idle for >10 min

    def __init__(
        self,
        capacity: int,
        refill_per_sec: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_sec <= 0:
            raise ValueError("refill_per_sec must be positive")
        self._capacity = float(capacity)
        self._refill = refill_per_sec
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._call_count = 0

    def _refill_bucket(self, b: _Bucket, now: float) -> None:
        elapsed = now - b.last_refill
        if elapsed > 0:
            b.tokens = min(self._capacity, b.tokens + elapsed * self._refill)
            b.last_refill = now

    def _maybe_evict(self, now: float) -> None:
        self._call_count += 1
        if self._call_count < self._EVICT_INTERVAL:
            return
        self._call_count = 0
        stale_keys = [
            k for k, b in self._buckets.items()
            if (now - b.last_refill) > self._STALE_SECONDS
        ]
        for k in stale_keys:
            del self._buckets[k]

    def take(self, key: str) -> bool:
        now = self._clock()
        self._maybe_evict(now)
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self._capacity, last_refill=now)
            self._buckets[key] = b
        else:
            self._refill_bucket(b, now)
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that 429s on bucket exhaustion."""

    def __init__(self, app, *, bucket: TokenBucket) -> None:
        super().__init__(app)
        self._bucket = bucket

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client is not None else "unknown"
        if not self._bucket.take(client):
            return JSONResponse(
                {"error": "rate_limited", "detail": "too many requests"},
                status_code=429,
            )
        return await call_next(request)
