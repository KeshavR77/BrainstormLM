"""Structured per-request JSONL logger (plan-06)."""

from __future__ import annotations

import datetime as dt
import json
from collections import deque
from pathlib import Path
from threading import Lock


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RequestLogger:
    """Append JSON lines to a server log + maintain a small in-memory ring."""

    def __init__(self, path: Path, *, ring_size: int = 1024) -> None:
        self.path = path
        self._ring: deque[dict] = deque(maxlen=ring_size)
        self._lock = Lock()

    def log(self, payload: dict) -> None:
        record = {"ts": _utc_now_iso(), **payload}
        with self._lock:
            self._ring.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._ring)
