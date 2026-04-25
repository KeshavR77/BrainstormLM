import together

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.config import get_settings

_client: together.Together | None = None


def _get_client() -> together.Together:
    global _client
    if _client is None:
        settings = get_settings()
        _client = together.Together(api_key=settings.together_api_key.get_secret_value())
    return _client


def complete(
    system: str | None,
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    *,
    use_cache: bool = True,
) -> str:
    key = cache_key("together", model, system, messages, max_tokens, temperature, top_p)
    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            return cached

    payload = list(messages)
    if system is not None:
        payload = [{"role": "system", "content": system}, *payload]

    response = _get_client().chat.completions.create(
        model=model,
        messages=payload,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    text = response.choices[0].message.content
    write_cache(key, text)
    return text
