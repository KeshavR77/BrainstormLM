"""Single-judge wrapper with a unified interface across providers.

A ``Judge`` exposes ``name``, ``family``, and ``score(prompt, response) -> JudgeResult``.
``BaseJudge`` holds all the provider-agnostic logic — the bias-mitigation truncation, the
disk cache, retry-with-backoff, and strict JSON parsing — so each provider subclass only has
to implement ``_call(system, user) -> str``.

Required behaviors (from the research synthesis):
  * temperature low (0.0 default) for stability;
  * response truncated to 4,000 chars before judging (EQ-Bench v3 length-bias control);
  * all judge calls cached to disk (Stage 2 is the cost driver);
  * retries with Retry-After-aware backoff on transient failures — critically, 429 rate
    limits, where giving up too soon makes the ensemble silently drop a whole judge.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ..types import DIMENSIONS, JudgeResult
from ..utils.caching import DiskCache
from .rubric import RUBRIC_VERSION, SYSTEM_PROMPT, build_user_message


@runtime_checkable
class Judge(Protocol):
    """Structural type for an ensemble member."""

    name: str
    family: str

    def score(self, prompt: str, response: str) -> JudgeResult: ...


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extract of a server-provided ``Retry-After`` (in seconds) from a provider
    exception. The anthropic/openai SDKs raise ``APIStatusError`` subclasses that carry the
    HTTP response on ``.response``; a 429 includes the rate-limit reset there. Returns None
    when the header is absent or in HTTP-date form, so the caller falls back to backoff."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # HTTP-date form — fall back to exponential backoff


def parse_judge_json(raw: str, judge_name: str) -> JudgeResult:
    """Parse a judge reply into a ``JudgeResult``; clamps each score to the 1-5 integer range.

    Raises ``ValueError`` on non-JSON output or a missing dimension.
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge {judge_name} returned non-JSON: {raw!r}") from exc

    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in data or data[dim] is None:
            raise ValueError(f"Judge {judge_name} missing dimension {dim!r}: {data}")
        scores[dim] = max(1, min(5, int(round(float(data[dim])))))
    reasoning = str(data.get("reasoning", ""))
    return JudgeResult(scores=scores, reasoning=reasoning, judge_name=judge_name)


# ---------------------------------------------------------------------------
# Base judge
# ---------------------------------------------------------------------------


class BaseJudge(ABC):
    def __init__(
        self,
        name: str,
        model: str,
        family: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 6,
        truncate_chars: int = 4000,
        cache: DiskCache | None = None,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        timeout: float = 60.0,
    ):
        self.name = name
        self.model = model
        self.family = family
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Retry budget is sized to ride out a rate-limit window: Anthropic's Tier-1 token
        # buckets reset within ~60s, and the wrapper honors the server's Retry-After, so a
        # handful of retries lets a throttled judge recover rather than being dropped.
        self.max_retries = max_retries
        self.truncate_chars = truncate_chars
        self.cache = cache
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.timeout = timeout

    @abstractmethod
    def _call(self, system: str, user: str) -> str:
        """Provider-specific API call returning the raw text reply."""

    def score(self, prompt: str, response: str) -> JudgeResult:
        truncated = response[: self.truncate_chars]
        key = None
        if self.cache is not None:
            key = DiskCache.make_key(self.model, RUBRIC_VERSION, prompt, truncated)
            hit = self.cache.get(key)
            if hit is not None:
                return JudgeResult(
                    scores={k: int(v) for k, v in hit["scores"].items()},
                    reasoning=hit["reasoning"],
                    judge_name=hit["judge_name"],
                )
        raw = self._call_with_retry(SYSTEM_PROMPT, build_user_message(prompt, truncated))
        result = parse_judge_json(raw, self.name)
        if self.cache is not None and key is not None:
            self.cache.set(
                key,
                {
                    "scores": result.scores,
                    "reasoning": result.reasoning,
                    "judge_name": result.judge_name,
                },
            )
        return result

    def _call_with_retry(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._call(system, user)
            except Exception as exc:  # transient API failures (esp. 429s): back off and retry
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self._backoff_seconds(attempt, exc))
        assert last_exc is not None
        raise last_exc

    def _backoff_seconds(self, attempt: int, exc: Exception) -> float:
        """How long to wait before the next retry. A server ``Retry-After`` (the rate-limit
        reset) wins when present; otherwise capped exponential backoff. Capping at
        ``max_backoff`` keeps a single stuck call from stalling a whole worker thread."""
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, self.max_backoff)
        return min(self.backoff_base * (2**attempt), self.max_backoff)


# ---------------------------------------------------------------------------
# Provider implementations (exempt from unit tests / coverage — they only wrap SDK calls).
# ---------------------------------------------------------------------------


class AnthropicJudge(BaseJudge):
    def __init__(self, model: str = "claude-sonnet-4-5", *, name: str = "claude", **kw):
        super().__init__(name=name, model=model, family="anthropic", **kw)

    def _call(self, system: str, user: str) -> str:
        import anthropic  # type: ignore

        # ``max_retries=0``: BaseJudge._call_with_retry is the single, Retry-After-aware retry
        # layer (so the SDK's own retries don't double up with it); ``timeout`` bounds a
        # stalled connection so it can't hang a worker thread indefinitely.
        client = anthropic.Anthropic(
            api_key=self.api_key, timeout=self.timeout, max_retries=0
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks; tolerate empty/non-text content (refusal, empty response,
        # or thinking-only blocks) instead of IndexError on resp.content[0]. An empty string
        # flows to parse_judge_json, which skips this judge for the item (same as the `or ""`
        # guards on the OpenAI/Google judges).
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class OpenAIJudge(BaseJudge):
    def __init__(self, model: str = "gpt-4o", *, name: str = "gpt4o", **kw):
        super().__init__(name=name, model=model, family="openai", **kw)

    def _call(self, system: str, user: str) -> str:
        from openai import OpenAI  # type: ignore

        # See AnthropicJudge: own the retry in the wrapper, bound the socket with a timeout.
        client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},  # JSON mode
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class GoogleJudge(BaseJudge):
    def __init__(self, model: str = "gemini-2.5-pro", *, name: str = "gemini", **kw):
        super().__init__(name=name, model=model, family="google", **kw)

    def _call(self, system: str, user: str) -> str:
        from google import genai  # type: ignore

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=f"{system}\n\n{user}",
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                # Disable Gemini 2.5 "thinking": its reasoning tokens count against
                # max_output_tokens and, on richer prompts, consume the whole budget before
                # any JSON is emitted (empty response). A judge only needs to emit small JSON,
                # so thinking buys nothing here. Requires a Flash-class model (Pro cannot set 0).
                "thinking_config": {"thinking_budget": 0},
            },
        )
        return resp.text or ""


# Map a provider string (from EvalConfig.JudgeSpec) to its judge class.
PROVIDER_REGISTRY: dict[str, type[BaseJudge]] = {
    "anthropic": AnthropicJudge,
    "openai": OpenAIJudge,
    "google": GoogleJudge,
}
