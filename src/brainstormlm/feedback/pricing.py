"""Per-model pricing table and cost arithmetic for the feedback loop."""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
}

_TOKEN_BUDGET = {
    "prompter": (300, 80),
    "generator": (100, 500),
    "evaluator": (1200, 250),
    "extractor": (1500, 300),
}

# Modal on-demand GPU pricing (USD/hour, observed 2026-04). Informational only;
# actual training-cost guardrail is `--max-modal-usd` enforced at the script level.
MODAL_GPU_HOURLY: dict[str, float] = {
    "A100-40GB": 2.78,
    "A100-80GB": 3.92,
    "H100-80GB": 4.56,
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_per_m, out_per_m = PRICES[model]
    return (prompt_tokens / 1_000_000) * in_per_m + (completion_tokens / 1_000_000) * out_per_m


def estimate_run_cost(
    n: int,
    *,
    prompter_model: str = "gpt-4o-mini",
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    evaluator_model: str = "gpt-4o-mini",
) -> float:
    p_in, p_out = _TOKEN_BUDGET["prompter"]
    g_in, g_out = _TOKEN_BUDGET["generator"]
    e_in, e_out = _TOKEN_BUDGET["evaluator"]
    per_trial = (
        cost_of(prompter_model, p_in, p_out)
        + cost_of(generator_model, g_in, g_out)
        + cost_of(evaluator_model, e_in, e_out)
    )
    return per_trial * n


def estimate_extraction_cost(n: int, *, extractor_model: str = "gpt-4o-mini") -> float:
    """Pre-flight estimate for n trajectories through the lesson extractor."""
    e_in, e_out = _TOKEN_BUDGET["extractor"]
    return cost_of(extractor_model, e_in, e_out) * n


def estimate_sibling_run_cost(
    n_prompts: int,
    *,
    siblings: int = 6,
    prompter_model: str = "gpt-4o-mini",
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    evaluator_model: str = "gpt-4o-mini",
) -> float:
    """Pre-flight estimate for `scripts/sample_siblings.py`.

    One prompter call per prompt, plus `siblings` generator + evaluator calls
    per prompt.
    """
    p_in, p_out = _TOKEN_BUDGET["prompter"]
    g_in, g_out = _TOKEN_BUDGET["generator"]
    e_in, e_out = _TOKEN_BUDGET["evaluator"]
    per_prompt = (
        cost_of(prompter_model, p_in, p_out)
        + siblings * cost_of(generator_model, g_in, g_out)
        + siblings * cost_of(evaluator_model, e_in, e_out)
    )
    return per_prompt * n_prompts


def estimate_modal_cost(hours: float, *, gpu: str = "A100-80GB") -> float:
    """Informational training-cost estimate (Modal on-demand)."""
    if gpu not in MODAL_GPU_HOURLY:
        raise KeyError(f"Unknown Modal GPU: {gpu}")
    return MODAL_GPU_HOURLY[gpu] * hours
