from __future__ import annotations

import argparse

from brainstormlm.conceptial.engine.config import SteeringConfig, SteeringStrategy
from brainstormlm.conceptial.engine.corpius import apply_corpius_1, load_corpius_spec
from brainstormlm.conceptial.engine.core.loader import load_engine


def _build_config(ns: argparse.Namespace) -> SteeringConfig:
    spec = load_corpius_spec()
    return SteeringConfig(
        model_id=ns.model or spec.base_model,
        strategy=SteeringStrategy.SINGLE,
        torch_dtype=ns.dtype,
        preferred_device=ns.device,
        device_map=ns.device_map,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate with Corpius 1 (Qwen2.5-3B + frozen v17 activation steering).",
    )
    p.add_argument("--model", default=None, help="HF model id override; defaults to config/corpius_1.json")
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Inference device when --device-map is omitted (default: cuda→mps→cpu).",
    )
    p.add_argument(
        "--device-map",
        default=None,
        dest="device_map",
        help='Optional HF device_map JSON/dict string, or "auto" (can disk-offload). '
        "Default: entire model on one accelerator (recommended on Mac).",
    )
    p.add_argument("--prompt", required=True, help="User prompt for generation")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--baseline", action="store_true", help="Generate with plain Qwen baseline instead of Corpius steering")

    args = p.parse_args(argv)

    engine = load_engine(_build_config(args))
    if not args.baseline:
        apply_corpius_1(engine)
    text = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_steering=not args.baseline,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
