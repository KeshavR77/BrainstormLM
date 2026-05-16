#!/usr/bin/env python3
"""Pre-demo verification — run before the live demo to catch broken pieces.

Usage::

    python3 scripts/precheck_demo.py

Exits 0 if every check passes, non-zero with a clear error otherwise.

Checks:
1. SQLite store is reachable; seeded counts match expectations.
2. Streaming source produces N distinct ideas for a fresh project.
3. Streaming source's anti-mode actually excludes prior emissions on re-run.
4. (Optional, if --check-server) FastAPI server responds at /v1/memory/count
   and the SSE endpoint streams at least one raw_idea event.
5. (Optional, if --check-frontend) Vite dev server responds at localhost:5173.

Designed to be slow-but-thorough — don't run inline during the demo, but DO
run it 5 minutes before showtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainstormlm.conceptial.idea_store import (  # noqa: E402
    SOURCE_SEEDED,
    IdeaMemoryStore,
)
from brainstormlm.conceptial.streaming_source import (  # noqa: E402
    ConceptialStreamingSource,
)


EXPECTED_SEEDED = {
    "demo-skincare-identity": 12,
    "demo-footwear-campaign": 8,
    "demo-oatmilk-packaging": 8,
}

OK = "\033[32m✓\033[0m"
WARN = "\033[33m!\033[0m"
ERR = "\033[31m✗\033[0m"


class CheckError(Exception):
    pass


def _ok(msg: str) -> None:
    print(f"  {OK} {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN} {msg}")


def _err(msg: str) -> None:
    print(f"  {ERR} {msg}")


def check_seeded_store() -> IdeaMemoryStore:
    print("\n[1/5] SQLite store + seeded counts")
    store = IdeaMemoryStore()
    _ok(f"opened {store.path}")
    fails: list[str] = []
    for project_id, expected in EXPECTED_SEEDED.items():
        breakdown = store.count_by_source(project_id)
        actual = int(breakdown.get(SOURCE_SEEDED, 0))
        if actual >= expected:
            _ok(f"{project_id}: seeded={actual} (expected≥{expected})")
        else:
            fails.append(f"{project_id}: seeded={actual}, expected≥{expected}")
            _err(fails[-1])
    if fails:
        raise CheckError(
            "Seeded counts are below expected. Run:\n"
            "  python3 scripts/seed_idea_memory.py --all --reset"
        )
    return store


def check_streaming_source(store: IdeaMemoryStore) -> None:
    print("\n[2/5] Streaming source emits N distinct ideas")
    test_project = "precheck-temp-project"
    store.delete_project(test_project)
    source = ConceptialStreamingSource(store=store, project_id=test_project, pacing=False)
    ideas = list(source.stream(prompt="precheck brand identity", n=5))
    if len(ideas) != 5:
        raise CheckError(f"Expected 5 ideas; got {len(ideas)}")
    _ok(f"emitted {len(ideas)} ideas")
    summaries = {i.summary for i in ideas}
    if len(summaries) != 5:
        raise CheckError(f"Duplicates in one call: {len(summaries)} unique of 5")
    _ok("no within-call duplicates")
    store.delete_project(test_project)


def check_anti_mode_works(store: IdeaMemoryStore) -> None:
    print("\n[3/5] Anti-mode excludes prior emissions on re-run")
    test_project = "precheck-antimode-project"
    store.delete_project(test_project)
    source = ConceptialStreamingSource(store=store, project_id=test_project, pacing=False)
    first = {i.summary for i in source.stream(prompt="precheck identity", n=5)}
    second = {i.summary for i in source.stream(prompt="precheck identity", n=5)}
    overlap = first & second
    if overlap:
        raise CheckError(f"Anti-mode failed: {overlap} appeared in both calls")
    _ok(f"first 5 ⊥ second 5 ({len(first)} + {len(second)} distinct)")
    store.delete_project(test_project)


def check_server(base_url: str, project_id: str = "demo-skincare-identity") -> None:
    print(f"\n[4/5] FastAPI server at {base_url}")
    try:
        resp = urllib.request.urlopen(
            f"{base_url}/v1/memory/count?project_id={project_id}", timeout=4
        )
    except urllib.error.URLError as e:
        raise CheckError(f"Server not reachable: {e}") from e
    data = json.loads(resp.read())
    _ok(f"/v1/memory/count returned seeded={data.get('seeded')}, generated={data.get('generated')}")

    # Streaming SSE smoke
    req_body = json.dumps(
        {
            "prompt": "precheck streaming smoke",
            "n_ideas": 2,
            "project_id": "precheck-sse-project",
            "reset_project": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/brainstorm/stream",
        data=req_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    got_raw = False
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode().rstrip()
            if line.startswith("event: raw_idea"):
                got_raw = True
                break
    if not got_raw:
        raise CheckError("SSE stream produced no raw_idea event")
    _ok(f"SSE first raw_idea arrived in {time.time() - start:.2f}s")


def check_frontend(url: str = "http://localhost:5173") -> None:
    print(f"\n[5/5] Frontend at {url}")
    try:
        resp = urllib.request.urlopen(url, timeout=4)
    except urllib.error.URLError as e:
        _warn(f"frontend not reachable: {e}")
        return
    body = resp.read()
    if b"<title>" in body:
        _ok("frontend responded with HTML")
    else:
        _warn("frontend responded but content looks unexpected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-server",
        default=None,
        help="If set, ping the FastAPI server at this URL (e.g. http://localhost:8000).",
    )
    parser.add_argument(
        "--check-frontend",
        default=None,
        help="If set, ping the Vite frontend at this URL (e.g. http://localhost:5173).",
    )
    args = parser.parse_args()

    try:
        store = check_seeded_store()
        check_streaming_source(store)
        check_anti_mode_works(store)
        if args.check_server:
            check_server(args.check_server)
        if args.check_frontend:
            check_frontend(args.check_frontend)
        store.close()
    except CheckError as e:
        print(f"\n{ERR} precheck failed:")
        print(f"  {e}")
        return 1

    print("\n✅ All checks passed. Demo is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
