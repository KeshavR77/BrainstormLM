from brainstormlm.clients import cache


def test_write_then_read_hits(tmp_path):
    key = cache.cache_key("hello", 1)
    cache.write_cache(key, "value-1", cache_dir=tmp_path)
    assert cache.read_cache(key, cache_dir=tmp_path) == "value-1"


def test_different_args_produce_different_keys():
    assert cache.cache_key("a", 1) != cache.cache_key("b", 1)


def test_missing_key_returns_none(tmp_path):
    assert cache.read_cache("nonexistent_key", cache_dir=tmp_path) is None
