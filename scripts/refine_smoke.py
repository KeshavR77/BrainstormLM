#!/usr/bin/env python3
"""Real-API smoke for the refine step + tail-calibration sweep.

Usage::

    # one live refine pass (needs GEMINI_API_KEY for chat, OPENAI_API_KEY for embeddings)
    uv run python scripts/refine_smoke.py "Design a logo for a fintech startup"

    # tail-sampling temperature/probability calibration sweep -> artifacts/ CSV
    uv run python scripts/refine_smoke.py "Name a coffee shop" --sweep

Not part of CI — exercises the real clients, so it is gated on the API keys being set.
Module logic is covered deterministically by tests/test_refine.py and
tests/test_tail_calibration.py (fakes, no keys needed).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainstormlm.config import get_settings  # noqa: E402
from brainstormlm.generation.calibration import sweep_tail_calibration  # noqa: E402
from brainstormlm.refine import make_embedder, make_llm, refine  # noqa: E402

DEFAULT_FLOORS = [0.0, 0.05, 0.1, 0.2, 0.4]
DEFAULT_TEMPS = [0.7, 1.0, 1.3]


def _check_keys(provider: str) -> None:
    """Fail fast with a clear message if required keys are missing."""
    try:
        settings = get_settings()
    except Exception as exc:  # pydantic ValidationError when OPENAI_API_KEY is unset
        sys.exit(
            f"Config error ({exc}).\n"
            "Set OPENAI_API_KEY (embeddings) and GEMINI_API_KEY (chat) in your environment or .env."
        )
    if provider == "gemini" and settings.gemini_api_key is None:
        sys.exit("GEMINI_API_KEY is not set — required for the Gemini chat client.")


def _run_refine(args: argparse.Namespace) -> None:
    llm = make_llm(provider=args.provider, model=args.model)
    embedder = make_embedder()
    result = refine(
        args.prompt,
        context=args.context,
        k=args.k,
        vs_n=args.vs_n,
        modal_floor=args.modal_floor,
        llm=llm,
        embedder=embedder,
    )
    print(f"\n=== BRIEF ===\n{result.brief}\n")
    print("=== ANGLES ===")
    for i, a in enumerate(result.angles):
        print(f"  {i}. {a}")
    print("\n=== IDEAS (one per angle) ===")
    for i, idea in enumerate(result.ideas):
        flag = "  <- regenerated" if i in result.replaced else ""
        print(f"  {i}. {idea}{flag}")
    print(f"\ndistinctness={result.distinctness:.3f}  replaced={result.replaced}")


def _run_sweep(args: argparse.Namespace) -> None:
    llm = make_llm(provider=args.provider, model=args.model)
    embedder = make_embedder()
    result = sweep_tail_calibration(
        args.prompt,
        llm=llm,
        embedder=embedder,
        floors=DEFAULT_FLOORS,
        temperatures=DEFAULT_TEMPS,
        m=args.m,
        vs_n=args.vs_n,
    )

    print(f"\n{'floor':>6} {'temp':>5} {'diversity':>10} {'distinct':>9} {'coherence':>10}")
    for c in result.cells:
        print(
            f"{c.modal_floor:>6.2f} {c.temperature:>5.2f} "
            f"{c.diversity:>10.3f} {c.distinct_count:>9d} {c.coherence:>10.3f}"
        )
    rec = result.recommended
    if rec is not None:
        print(f"\nrecommended: floor={rec.modal_floor} temp={rec.temperature} "
              f"diversity={rec.diversity:.3f} coherence={rec.coherence:.3f}")
    else:
        print("\nno cell cleared the coherence floor")

    out = args.out or (ROOT / "artifacts" / f"tail_calibration_{datetime.now():%Y%m%d_%H%M%S}.csv")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = result.to_rows()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="the brainstorming prompt / brief")
    parser.add_argument("--context", default=None, help="optional background context")
    parser.add_argument("--k", type=int, default=6, help="number of orthogonal angles")
    parser.add_argument("--vs-n", type=int, default=4, dest="vs_n", help="VS candidates per call")
    parser.add_argument("--modal-floor", type=float, default=0.1, dest="modal_floor")
    parser.add_argument("--provider", default=None, help="chat provider (default: config refine_provider)")
    parser.add_argument("--model", default=None, help="chat model (default: config refine_model)")
    parser.add_argument("--sweep", action="store_true", help="run the tail-calibration sweep instead")
    parser.add_argument("--m", type=int, default=40, help="samples per sweep cell")
    parser.add_argument("--out", default=None, help="sweep CSV output path")
    args = parser.parse_args()

    provider = args.provider or get_settings().refine_provider
    _check_keys(provider)

    if args.sweep:
        _run_sweep(args)
    else:
        _run_refine(args)


if __name__ == "__main__":
    main()
