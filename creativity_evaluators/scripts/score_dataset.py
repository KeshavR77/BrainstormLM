#!/usr/bin/env python
"""Batch-score a JSONL of (prompt, response[, category]) records into EvalResults.

Each input line is a JSON object with at least ``prompt`` and ``response``; ``category``
defaults to ``creative``. Output is JSONL, one serialized ``EvalResult`` per line.

Examples
--------
    # Stage 1 only (no API keys / no judge cost):
    python scripts/score_dataset.py --input raw.jsonl --output scored.jsonl --no-stage2

    # Full two-stage scoring with a config file:
    python scripts/score_dataset.py --input raw.jsonl --output scored.jsonl --config cfg.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Make the package importable when this script is run directly (without `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creativity_evaluators.pipeline import evaluate_batch  # noqa: E402
from creativity_evaluators.types import EvalConfig  # noqa: E402


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj.setdefault("category", "creative")
        records.append(obj)
    return records


def load_config(path: Path | None, no_stage2: bool) -> EvalConfig:
    config = EvalConfig.model_validate_json(path.read_text()) if path else EvalConfig()
    if no_stage2:
        config = config.model_copy(update={"enable_stage2": False})
    return config


def serialize(result) -> str:
    # asdict recurses dataclasses; default=str handles the PromptCategory enum.
    return json.dumps(dataclasses.asdict(result), default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input JSONL")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL")
    parser.add_argument("--config", type=Path, default=None, help="EvalConfig JSON (optional)")
    parser.add_argument("--no-stage2", action="store_true", help="skip the LLM-judge ensemble")
    parser.add_argument("--workers", type=int, default=4, help="Stage-2 parallelism")
    args = parser.parse_args()

    records = load_records(args.input)
    config = load_config(args.config, args.no_stage2)
    results = evaluate_batch(records, config, n_workers=args.workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for result in results:
            f.write(serialize(result) + "\n")
    print(f"Scored {len(results)} records -> {args.output}")


if __name__ == "__main__":
    main()
