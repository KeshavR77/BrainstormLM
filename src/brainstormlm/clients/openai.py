import json
from dataclasses import dataclass

import numpy as np
import openai

from brainstormlm.clients.cache import cache_key, read_cache, write_cache
from brainstormlm.config import get_settings

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
        _client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
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
    key = cache_key("openai", model, system, messages, max_tokens, temperature)
    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            text, usage = _decode(cached)
            return text, Usage(usage.prompt_tokens, usage.completion_tokens, cached=True)

    payload = [{"role": "system", "content": system}, *messages]
    response = _get_client().chat.completions.create(
        model=model,
        messages=payload,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.choices[0].message.content
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    write_cache(key, _encode(text, usage))
    return text, usage


def complete_json(
    system: str,
    messages: list[dict],
    model: str,
    schema: dict,
    schema_name: str = "result",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    *,
    use_cache: bool = True,
) -> tuple[dict, Usage]:
    key = cache_key("openai_json", model, system, messages, schema, schema_name, max_tokens, temperature)
    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            text, usage = _decode(cached)
            return json.loads(text), Usage(usage.prompt_tokens, usage.completion_tokens, cached=True)

    payload = [{"role": "system", "content": system}, *messages]
    response = _get_client().chat.completions.create(
        model=model,
        messages=payload,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    text = response.choices[0].message.content
    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    write_cache(key, _encode(text, usage))
    return json.loads(text), usage


def embed(
    texts: list[str],
    *,
    model: str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, Usage]:
    """Return (N, D) float32 L2-normalized embeddings, cached per (model, text)."""
    model = model or get_settings().embedding_model
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), Usage(0, 0, cached=True)

    keys = [cache_key("openai_embed", model, t) for t in texts]
    cached_vectors: dict[int, np.ndarray] = {}
    miss_indices: list[int] = []
    if use_cache:
        for i, k in enumerate(keys):
            blob = read_cache(k)
            if blob is None:
                miss_indices.append(i)
                continue
            cached_vectors[i] = np.asarray(json.loads(blob)["embedding"], dtype=np.float32)
    else:
        miss_indices = list(range(len(texts)))

    prompt_tokens = 0
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        response = _get_client().embeddings.create(model=model, input=miss_texts)
        prompt_tokens = response.usage.prompt_tokens
        for i, datum in zip(miss_indices, response.data, strict=True):
            vec = np.asarray(datum.embedding, dtype=np.float32)
            cached_vectors[i] = vec
            write_cache(keys[i], json.dumps({"embedding": vec.tolist()}))

    dim = next(iter(cached_vectors.values())).shape[0]
    matrix = np.empty((len(texts), dim), dtype=np.float32)
    for i in range(len(texts)):
        matrix[i] = cached_vectors[i]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    matrix = matrix / norms

    return matrix, Usage(prompt_tokens=prompt_tokens, completion_tokens=0, cached=not miss_indices)
