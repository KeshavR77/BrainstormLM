#!/usr/bin/env python3
"""Merge several chunked two-stage sweep result JSONs into one combined map.

The overnight novelty sweep is split into per-layer-group chunks (so a crash loses at most one
chunk). This concatenates their ``cells`` into a single sweep-shaped JSON the layer picker can
read. The baseline (cell with no layer) is carried once from the first file that has one.

Usage:  merge_cells.py <out.json> <chunk1.json> [chunk2.json ...]
"""
import json
import sys


def merge(out_path: str, in_paths: list[str]) -> int:
    cells = []
    baseline = None
    evaluator = "two_stage"
    for p in in_paths:
        try:
            d = json.loads(open(p).read())
        except Exception as e:  # a missing/corrupt chunk must not sink the merge
            print(f"WARN skip {p}: {e}", file=sys.stderr)
            continue
        evaluator = d.get("evaluator", evaluator)
        if baseline is None and d.get("baseline"):
            baseline = d["baseline"]
        for c in d.get("cells", []):
            if c.get("layer") is not None:
                cells.append(c)
    out = {"evaluator": evaluator, "baseline": baseline, "cells": cells}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return len(cells)


if __name__ == "__main__":
    out, ins = sys.argv[1], sys.argv[2:]
    n = merge(out, ins)
    print(f"merged {n} cells from {len(ins)} files -> {out}")
