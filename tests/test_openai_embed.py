"""Cache-discipline tests for clients.openai.embed (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from brainstormlm import config
from brainstormlm.clients import openai as openai_client


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORMLM_CACHE_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    monkeypatch.setattr(openai_client, "_client", None, raising=False)
    yield tmp_path
    config.get_settings.cache_clear()


def _fake_response(vectors: list[list[float]], prompt_tokens: int = 10):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _install_fake(monkeypatch, response):
    fake = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: response),
        _calls=[],
    )

    def create(**kwargs):
        fake._calls.append(kwargs)
        return response

    fake.embeddings.create = create
    monkeypatch.setattr(openai_client, "_get_client", lambda: fake)
    return fake


def test_embed_caches_repeated_texts(isolated_cache, monkeypatch):
    fake = _install_fake(
        monkeypatch, _fake_response([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )

    matrix1, usage1 = openai_client.embed(
        ["hello", "world"], model="text-embedding-3-small"
    )
    assert matrix1.shape == (2, 3)
    assert usage1.cached is False
    assert len(fake._calls) == 1

    matrix2, usage2 = openai_client.embed(
        ["hello", "world"], model="text-embedding-3-small"
    )
    assert usage2.cached is True
    assert len(fake._calls) == 1
    np.testing.assert_array_equal(matrix1, matrix2)


def test_embed_partial_hit_only_requests_misses(isolated_cache, monkeypatch):
    _install_fake(monkeypatch, _fake_response([[1.0, 0.0, 0.0]]))
    openai_client.embed(["hello"], model="text-embedding-3-small")

    fake = _install_fake(monkeypatch, _fake_response([[0.0, 1.0, 0.0]]))
    matrix, _usage = openai_client.embed(
        ["hello", "world"], model="text-embedding-3-small"
    )

    assert len(fake._calls) == 1
    assert fake._calls[0]["input"] == ["world"]
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix[0], [1.0, 0.0, 0.0])
    assert np.allclose(matrix[1], [0.0, 1.0, 0.0])


def test_embed_l2_normalizes(isolated_cache, monkeypatch):
    _install_fake(monkeypatch, _fake_response([[3.0, 4.0]]))
    matrix, _ = openai_client.embed(["x"], model="text-embedding-3-small")
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0], atol=1e-6)


def test_embed_empty_list_short_circuits(isolated_cache, monkeypatch):
    fake = _install_fake(monkeypatch, _fake_response([]))
    matrix, usage = openai_client.embed([], model="text-embedding-3-small")
    assert matrix.shape[0] == 0
    assert usage.cached is True
    assert len(fake._calls) == 0


def test_embed_no_cache_always_calls_api(isolated_cache, monkeypatch):
    fake = _install_fake(monkeypatch, _fake_response([[1.0, 0.0]]))
    openai_client.embed(["hello"], model="text-embedding-3-small")
    openai_client.embed(["hello"], model="text-embedding-3-small", use_cache=False)
    assert len(fake._calls) == 2
