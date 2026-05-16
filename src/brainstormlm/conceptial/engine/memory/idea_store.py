"""SQLite-backed per-project native-latent idea memory.

V8.2 / Day 3: store last-token hidden states (Qwen2.5-3B L28, 2048-d) keyed by
``(project_id, idea_id)`` so the live IOR repulsion FIFO can be hydrated from disk
on each new request. Trades nothing on the hot decode path — only the
``add`` / ``list_recent_hiddens`` calls touch SQLite, and both are O(N) in the
project's own row count.

**Code-hygiene constraint (from the approved 7-day plan):** zero
``shrooms_llm``-specific imports. Only ``torch``, ``numpy``, ``sqlite3``, stdlib.
This makes the file directly copy-pasteable into
``BrainstormLM/src/brainstormlm/memory/idea_store_native.py`` during the
post-demo consolidation port.

Schema:

.. code-block:: sql

    CREATE TABLE idea_memory (
      project_id   TEXT    NOT NULL,
      idea_id      TEXT    NOT NULL,
      text         TEXT    NOT NULL,
      hidden_f16   BLOB    NOT NULL,  -- numpy float16 bytes, shape (hidden_size,)
      hidden_size  INTEGER NOT NULL,
      source       TEXT    NOT NULL,  -- 'seeded' | 'generated'
      created_at   REAL    NOT NULL,  -- unix epoch seconds
      PRIMARY KEY (project_id, idea_id)
    );
    CREATE INDEX idx_project_created ON idea_memory(project_id, created_at DESC);

Storage cost: 2048 × 2 bytes = 4 KiB per row plus text.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_DB_PATH_ENV = "CONCEPTIAL_IDEA_DB_PATH"
DEFAULT_DB_PATH = "./data/idea_memory.db"
SOURCE_SEEDED = "seeded"
SOURCE_GENERATED = "generated"
_VALID_SOURCES = frozenset({SOURCE_SEEDED, SOURCE_GENERATED})


@dataclass(frozen=True)
class IdeaRow:
    project_id: str
    idea_id: str
    text: str
    source: str
    created_at: float
    hidden: torch.Tensor


class IdeaMemoryStore:
    """File-backed SQLite store. Thread-safe via a single connection guarded by a lock.

    Single-process usage assumes the FastAPI server already serializes generation
    via ``state.gen_lock`` (``src/api/server.py``). Concurrent writes from outside
    the server are not supported — use the seed loader script for bulk loads
    while the server is idle.
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        path = Path(db_path) if db_path is not None else Path(os.environ.get(DEFAULT_DB_PATH_ENV, DEFAULT_DB_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        # ``check_same_thread=False`` is safe because every call acquires ``self._lock`` first.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._lock = threading.Lock()
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS idea_memory (
                  project_id   TEXT    NOT NULL,
                  idea_id      TEXT    NOT NULL,
                  text         TEXT    NOT NULL,
                  hidden_f16   BLOB    NOT NULL,
                  hidden_size  INTEGER NOT NULL,
                  source       TEXT    NOT NULL,
                  created_at   REAL    NOT NULL,
                  PRIMARY KEY (project_id, idea_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_project_created ON idea_memory(project_id, created_at DESC)"
            )
            self._conn.commit()

    def add(
        self,
        project_id: str,
        idea_id: str,
        hidden: torch.Tensor,
        text: str,
        *,
        source: str = SOURCE_GENERATED,
    ) -> None:
        """Insert (or replace) one idea row. ``hidden`` is expected to be a 1D
        unit-normalized tensor; the store does not normalize for you so the
        IOR-side and seed-side hiddens stay on the same manifold."""
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if hidden.ndim != 1:
            raise ValueError(f"hidden must be 1D; got shape {tuple(hidden.shape)}")
        if source not in _VALID_SOURCES:
            raise ValueError(f"source must be one of {_VALID_SOURCES}; got {source!r}")
        h_np = hidden.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy()
        blob = h_np.tobytes()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO idea_memory
                  (project_id, idea_id, text, hidden_f16, hidden_size, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(project_id),
                    str(idea_id),
                    str(text),
                    blob,
                    int(h_np.shape[0]),
                    str(source),
                    float(time.time()),
                ),
            )
            self._conn.commit()

    def add_many(self, rows: list[IdeaRow]) -> int:
        """Bulk insert — used by the seed loader. Returns the count actually written."""
        if not rows:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO idea_memory
                  (project_id, idea_id, text, hidden_f16, hidden_size, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.project_id,
                        r.idea_id,
                        r.text,
                        r.hidden.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy().tobytes(),
                        int(r.hidden.shape[0]),
                        r.source if r.source in _VALID_SOURCES else SOURCE_SEEDED,
                        r.created_at,
                    )
                    for r in rows
                ],
            )
            n = cur.rowcount if cur.rowcount is not None else len(rows)
            self._conn.commit()
            return int(n)

    def count(self, project_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM idea_memory WHERE project_id = ?",
                (str(project_id),),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def count_by_source(self, project_id: str) -> dict[str, int]:
        """Return ``{'seeded': N, 'generated': M}`` for the SteeringTelemetry chip."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, COUNT(*) FROM idea_memory WHERE project_id = ? GROUP BY source",
                (str(project_id),),
            ).fetchall()
        out = {SOURCE_SEEDED: 0, SOURCE_GENERATED: 0}
        for src, n in rows:
            out[str(src)] = int(n)
        return out

    def list_recent_hiddens(self, project_id: str, *, limit: int = 5) -> list[torch.Tensor]:
        """Return up to ``limit`` most-recent hidden tensors for ``project_id``, as
        float32 CPU tensors ready to assign into ``IOREngine._explored``.

        Ordering: most-recent first. The IOR FIFO treats list-index 0 as oldest,
        so the caller should reverse() before assigning. (See ``server.py`` hydrate
        path on Day 3.)
        """
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT hidden_f16, hidden_size FROM idea_memory
                 WHERE project_id = ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (str(project_id), int(limit)),
            ).fetchall()
        out: list[torch.Tensor] = []
        for blob, hsize in rows:
            arr = np.frombuffer(blob, dtype=np.float16, count=int(hsize)).astype(np.float32)
            out.append(torch.from_numpy(arr.copy()))  # copy() to escape the readonly buffer
        return out

    def list_recent(self, project_id: str, *, limit: int = 5) -> list[IdeaRow]:
        """Variant of :meth:`list_recent_hiddens` that returns full :class:`IdeaRow`s
        (text, ids, source, timestamp). Handy for debug endpoints and the seed loader."""
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT project_id, idea_id, text, source, created_at, hidden_f16, hidden_size
                  FROM idea_memory
                 WHERE project_id = ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (str(project_id), int(limit)),
            ).fetchall()
        out: list[IdeaRow] = []
        for pid, iid, text, source, ts, blob, hsize in rows:
            arr = np.frombuffer(blob, dtype=np.float16, count=int(hsize)).astype(np.float32)
            out.append(
                IdeaRow(
                    project_id=str(pid),
                    idea_id=str(iid),
                    text=str(text),
                    source=str(source),
                    created_at=float(ts),
                    hidden=torch.from_numpy(arr.copy()),
                )
            )
        return out

    def delete_project(self, project_id: str) -> int:
        """Remove every row for ``project_id``. Returns the count deleted.

        Used by the demo-reset path and by destructive tests. The seed loader also
        calls this when ``--reset`` is passed so a re-seed produces a clean slate.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM idea_memory WHERE project_id = ?",
                (str(project_id),),
            )
            n = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
            return int(n)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
