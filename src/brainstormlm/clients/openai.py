import openai

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.config import get_settings

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
    return _client


def complete(
    system: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    *,
    use_cache: bool = True,
) -> str:
    key = cache_key("openai", model, system, messages, max_tokens, temperature)
    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            return cached

    payload = [{"role": "system", "content": system}, *messages]
    response = _get_client().chat.completions.create(
        model=model,
        messages=payload,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content
    write_cache(key, text)
    return text
