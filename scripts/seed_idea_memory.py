#!/usr/bin/env python3
"""Bulk-load a curated seed JSON into the Conceptial IdeaMemoryStore.

Usage::

    python3 scripts/seed_idea_memory.py \\
        --seed-json artifacts/conceptial/seeds/seed_skincare_brand_identity.json

    # Or load every seed at once:
    python3 scripts/seed_idea_memory.py --all

Each seed JSON specifies a ``project_id`` and a list of ``ideas`` with
``summary`` + ``content`` fields. The script writes them into the SQLite
store at ``$CONCEPTIAL_IDEA_DB_PATH`` (default ``./data/idea_memory.db``)
with ``source='seeded'`` so they appear in the chip's seeded count.

The hidden-state slot is filled with a zero-vector placeholder — the demo
streaming path uses the text-level exclusion, not activation-level
repulsion, so the real residual stream isn't required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainstormlm.conceptial.idea_store import (  # noqa: E402  — after sys.path bootstrap
    SOURCE_SEEDED,
    IdeaMemoryStore,
    IdeaRow,
)


def _load_seed(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _zero_hidden(dim: int = 64) -> torch.Tensor:
    return torch.zeros(dim, dtype=torch.float32)


def seed_project(store: IdeaMemoryStore, seed_path: Path, *, reset: bool, hidden_dim: int) -> int:
    data = _load_seed(seed_path)
    project_id = data["project_id"]
    ideas = data["ideas"]
    name = data.get("project_name", project_id)

    if reset:
        deleted = store.delete_project(project_id)
        if deleted:
            print(f"  [reset] cleared {deleted} prior rows for {project_id}")

    rows: list[IdeaRow] = []
    now = time.time()
    for k, idea in enumerate(ideas):
        text = f"{idea['summary']}\n{idea['content']}"
        rows.append(
            IdeaRow(
                project_id=project_id,
                idea_id=f"seed-{uuid.uuid4().hex[:12]}",
                text=text,
                source=SOURCE_SEEDED,
                created_at=now + k * 0.001,
                hidden=_zero_hidden(hidden_dim),
            )
        )

    n = store.add_many(rows)
    print(f"  loaded {n} ideas into project '{project_id}' ({name})")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-json", type=Path, default=None, help="Path to a single seed JSON")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load every JSON under artifacts/conceptial/seeds/",
    )
    parser.add_argument("--reset", action="store_true", help="Clear each project before seeding")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override CONCEPTIAL_IDEA_DB_PATH for this run.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if not args.seed_json and not args.all:
        parser.error("pass --seed-json <path> or --all")

    store = IdeaMemoryStore(db_path=args.db_path) if args.db_path else IdeaMemoryStore()
    total = 0
    try:
        if args.all:
            seeds_dir = ROOT / "artifacts" / "conceptial" / "seeds"
            paths = sorted(seeds_dir.glob("seed_*.json"))
            if not paths:
                print(f"No seed files found under {seeds_dir}", file=sys.stderr)
                return 1
            print(f"Found {len(paths)} seed files")
            for p in paths:
                total += seed_project(store, p, reset=args.reset, hidden_dim=args.hidden_dim)
        else:
            total = seed_project(store, args.seed_json, reset=args.reset, hidden_dim=args.hidden_dim)
    finally:
        store.close()

    print(f"\nDone. Total ideas inserted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
