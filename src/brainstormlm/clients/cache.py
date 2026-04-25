import hashlib
import json
from pathlib import Path

from brainstormlm.config import get_settings


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is None:
        cache_dir = get_settings().cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / key[:2] / f"{key[2:]}.json"


def cache_key(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_cache(key: str, cache_dir: Path | None = None) -> str | None:
    path = _cache_path(key, _resolve_cache_dir(cache_dir))
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["value"]


def write_cache(key: str, value: str, cache_dir: Path | None = None) -> None:
    path = _cache_path(key, _resolve_cache_dir(cache_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"value": value}, f)
