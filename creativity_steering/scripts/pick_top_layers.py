#!/usr/bin/env python3
"""Pick the top-N layers from a Stage-1 novelty sweep result, ranked by best-cell novelty.

Reads a ``sweep_results.json`` (the two-stage sweep output) and prints a comma-separated list
of the N distinct layers whose best (layer, alpha) cell has the highest Stage-1 novelty. Used
by the overnight orchestrator to focus the expensive judged frontier on the layers that the
cheap novelty pass already flagged as promising.

Usage:  pick_top_layers.py <sweep_results.json> <N>

Robust by design: any parse/IO failure exits non-zero with no stdout so the caller's shell
fallback (a hardcoded layer set from prior runs) takes over — the night must not die here.
"""
import json
import sys


def pick(path: str, n: int) -> list[int]:
    data = json.loads(open(path).read())
    # Rank non-baseline cells by Stage-1 novelty; collect distinct layers in that order.
    cells = [c for c in data.get("cells", []) if c.get("layer") is not None]
    cells.sort(key=lambda c: c.get("stage1", c.get("composite", float("-inf"))), reverse=True)
    layers: list[int] = []
    for c in cells:
        L = int(c["layer"])
        if L not in layers:
            layers.append(L)
        if len(layers) >= n:
            break
    if not layers:
        raise ValueError("no layered cells found")
    return layers


if __name__ == "__main__":
    path, n = sys.argv[1], int(sys.argv[2])
    print(",".join(str(L) for L in pick(path, n)))
