#!/usr/bin/env python
"""Calibrate the composite scorer against a hand-rated gold set.

Loads a mini gold set (50-100 items, each with a human creativity rating 1-5), runs the full
pipeline, and reports Spearman + Pearson correlation between the composite ``creativity_score``
and the human ratings. It also prints suggested quintile thresholds on the composite score
(for splitting future batches into creativity quintiles).

Acceptance bar: Spearman >= 0.5 on this set. Below that, the pipeline is not ready and the
script prints FAIL and exits non-zero.

Gold-set schema (PLACEHOLDER loader — replace with the real set):
    {"items": [{"prompt": str, "response": str, "category": "creative"|"neutral",
                "human_score": float}]}
A bare JSON list of those item objects is also accepted.

Example
-------
    python scripts/calibrate_thresholds.py --gold tests/fixtures/sample_responses.json --no-stage2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Make the package importable when this script is run directly (without `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from creativity_evaluators.pipeline import evaluate_batch  # noqa: E402
from creativity_evaluators.types import EvalConfig  # noqa: E402

ACCEPTANCE_SPEARMAN = 0.5
DEFAULT_GOLD = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_responses.json"


def load_gold(path: Path) -> list[dict]:
    """Placeholder loader. Returns a list of {prompt, response, category, human_score}."""
    data = json.loads(path.read_text())
    items = data["items"] if isinstance(data, dict) else data
    missing = [i for i, it in enumerate(items) if "human_score" not in it]
    if missing:
        raise ValueError(f"gold items missing 'human_score' at indices {missing}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="gold-set JSON")
    parser.add_argument("--config", type=Path, default=None, help="EvalConfig JSON (optional)")
    parser.add_argument("--no-stage2", action="store_true", help="skip the LLM-judge ensemble")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    config = EvalConfig.model_validate_json(args.config.read_text()) if args.config else EvalConfig()
    if args.no_stage2:
        config = config.model_copy(update={"enable_stage2": False})

    results = evaluate_batch(
        [{"prompt": it["prompt"], "response": it["response"], "category": it["category"]} for it in gold],
        config,
    )
    predicted = np.array([r.creativity_score for r in results], dtype=float)
    human = np.array([float(it["human_score"]) for it in gold], dtype=float)

    rho, _ = spearmanr(predicted, human)
    r, _ = pearsonr(predicted, human)
    quintiles = np.percentile(predicted, [20, 40, 60, 80])

    print("=" * 60)
    print(f"Calibration on {len(gold)} gold items (stage2={'off' if args.no_stage2 else 'on'})")
    print(f"  Spearman: {rho:.3f}   (acceptance bar >= {ACCEPTANCE_SPEARMAN})")
    print(f"  Pearson:  {r:.3f}")
    print(f"  Suggested quintile thresholds (20/40/60/80 pct): {np.round(quintiles, 3).tolist()}")
    passed = rho >= ACCEPTANCE_SPEARMAN
    print(f"  RESULT: {'PASS' if passed else 'FAIL — pipeline not ready'}")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
