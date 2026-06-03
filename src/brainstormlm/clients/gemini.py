"""Gemini chat client via Google's OpenAI-compatible endpoint.

Gemini exposes an OpenAI-compatible surface at ``/v1beta/openai/``, so we reuse the
already-installed ``openai`` SDK with a custom ``base_url`` instead of adding the
``google-genai`` dependency. Shape mirrors ``clients/openai.py`` (cache-keyed
``complete`` returning ``(text, Usage)``) so the two are drop-in interchangeable.

Models: ``gemini-2.5-flash-lite`` (default — cheapest), ``gemini-2.5-flash``.
Requires ``GEMINI_API_KEY``.
"""

import json
from dataclasses import dataclass

import openai

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.config import get_settings

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_client: openai.OpenAI | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY is not set — cannot call the Gemini client.")
        _client = openai.OpenAI(
            api_key=settings.gemini_api_key.get_secret_value(),
            base_url=_BASE_URL,
        )
    return _client


def _encode(text: str, usage: Usage) -> str:
    return json.dumps({"text": text, "usage": [usage.prompt_tokens, usage.completion_tokens]})


def _decode(blob: str) -> tuple[str, Usage]:
    parsed = json.loads(blob)
    p, c = parsed["usage"]
    return parsed["text"], Usage(prompt_tokens=p, completion_tokens=c)


def complete(
    system: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    *,
    use_cache: bool = True,
) -> tuple[str, Usage]:
    key = cache_key("gemini", model, system, messages, max_tokens, temperature)
    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            text, usage = _decode(cached)
            return text, Usage(usage.prompt_tokens, usage.completion_tokens, cached=True)

    payload = [{"role": "system", "content": system}, *messages]
    response = _get_client().chat.completions.create(
        model=model,
        messages=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    write_cache(key, _encode(text, usage))
    return text, usage
