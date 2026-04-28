"""Derive a creativity steering vector via Contrastive Activation Addition.

For N domain prompts (cached prompter calls), build paired (creative, generic)
chat-template renderings and compute the mean of last-token hidden-state
differences at the configured layers. Saves a `SteeringVector` artifact under
`data/steering/<run_id>/`.

Held-out prompts are mechanically excluded — same rule as plan-04's sibling
sampling. Cost guardrails cover prompter calls only; the local forward passes
are free but slow on a dev box (real derivation runs on Modal).
"""

from __future__ import annotations

import datetime as dt
import random
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from brainstormlm.config import get_settings
from brainstormlm.eval.fixtures import HELD_OUT
from brainstormlm.feedback import pricing, prompter
from brainstormlm.model.steering import derive_from_pairs, render_contrast_pair

app = typer.Typer(add_completion=False, no_args_is_help=False)


DEFAULT_LAYERS = "12,14,16,18,20"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_layers(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--layers must contain at least one integer")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise typer.BadParameter(f"--layers must be comma-separated ints: {e}") from e


def _estimate_prompter_cost(n_pairs: int, prompter_model: str) -> float:
    p_in, p_out = pricing._TOKEN_BUDGET["prompter"]
    return pricing.cost_of(prompter_model, p_in, p_out) * n_pairs


@app.command()
def main(
    n_pairs: Annotated[int, typer.Option(help="Number of (creative, generic) prompt pairs to derive over.")] = 64,
    layers: Annotated[str, typer.Option(help="Comma-separated layer indices to extract from.")] = DEFAULT_LAYERS,
    run_id: Annotated[str | None, typer.Option(help="Identifier for this derivation run (default: random hex).")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Steering output dir (default: data/steering).")] = None,
    seed: Annotated[int, typer.Option(help="Seed for domain shuffle.")] = 42,
    max_spend_usd: Annotated[float, typer.Option(help="Abort if prompter cost estimate exceeds this.")] = 0.10,
    base_model_id: Annotated[str | None, typer.Option(help="Override the base HF model id.")] = None,
    dtype: Annotated[str, typer.Option(help="Model dtype: bf16 / fp16 / fp32.")] = "bf16",
    device: Annotated[str | None, typer.Option(help="Override torch device (auto-detect by default).")] = None,
) -> None:
    settings = get_settings()
    rid = run_id or uuid.uuid4().hex[:12]
    out_root = out_dir or settings.steering_dir
    out_path = out_root / rid
    layer_list = _parse_layers(layers)
    base_model = base_model_id or settings.hf_base_model

    estimate = _estimate_prompter_cost(n_pairs, settings.prompter_model)
    typer.echo(f"Pre-flight prompter estimate for n_pairs={n_pairs}: ${estimate:.4f}")
    if estimate > max_spend_usd:
        typer.echo(
            f"Aborting: estimate ${estimate:.4f} exceeds --max-spend-usd ${max_spend_usd:.4f}.",
            err=True,
        )
        raise typer.Exit(code=2)

    held_out_prompts = {p for _, p in HELD_OUT}

    rng = random.Random(seed)
    domains = list(prompter.DOMAINS)
    rng.shuffle(domains)

    # Lazy-load HF tokenizer + model (heavy; same path HFGenerator uses).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    typer.echo(f"Loading {base_model} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)
    model.eval()

    pairs: list[tuple[str, str]] = []
    total_prompter_cost = 0.0
    for i in tqdm(range(n_pairs), desc="pairs"):
        domain = domains[i % len(domains)]
        salt = i // len(domains)
        prompt_text, p_usage = prompter.generate_prompt(domain, salt=salt)
        assert prompt_text not in held_out_prompts, (
            f"Prompter produced a held-out prompt: {prompt_text!r}"
        )
        if not p_usage.cached:
            total_prompter_cost += pricing.cost_of(
                settings.prompter_model, p_usage.prompt_tokens, p_usage.completion_tokens
            )
        creative, generic = render_contrast_pair(prompt_text, tokenizer)
        pairs.append((creative, generic))

    typer.echo(f"Built {len(pairs)} contrast pairs (prompter cost: ${total_prompter_cost:.4f}).")
    typer.echo(f"Deriving steering vector at layers {layer_list}...")

    sv = derive_from_pairs(
        pairs,
        base_model_id=base_model,
        layers=layer_list,
        derived_at=_utc_now_iso(),
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    sv.save(out_path)

    typer.echo("")
    typer.echo(f"Steering vector saved to: {out_path}")
    typer.echo(f"  base_model_id      : {sv.base_model_id}")
    typer.echo(f"  layers             : {list(sv.layers)}")
    typer.echo(f"  hidden_dim         : {sv.hidden_dim}")
    typer.echo(f"  n_contrast_pairs   : {sv.n_contrast_pairs}")
    typer.echo(f"  derivation_version : {sv.derivation_version}")
    typer.echo(f"  derived_at         : {sv.derived_at}")
    typer.echo(f"  total prompter cost: ${total_prompter_cost:.4f}")


if __name__ == "__main__":
    sys.exit(app() or 0)
