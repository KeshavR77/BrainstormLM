"""A tiny JSON disk cache.

Stage 2 (judges) and the Stage-1 reward model are the cost drivers; both are deterministic
given their inputs, so we memoize results to disk keyed on a hash of the inputs. Values must
be JSON-serializable. Writes go to a temp file then atomic-rename, so a crash mid-write can't
leave a half-written cache entry that later reads as valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class DiskCache:
    """Content-addressed JSON cache under ``cache_dir`` (one file per key)."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(*parts: str) -> str:
        """Stable, order-sensitive hash of the parts (NUL-joined so 'ab','c' != 'a','bc')."""
        h = hashlib.sha256()
        h.update("\x00".join(parts).encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        # Hash again so arbitrary keys map to filesystem-safe names.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(value, f)
            os.replace(tmp, path)  # atomic on the same filesystem
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
